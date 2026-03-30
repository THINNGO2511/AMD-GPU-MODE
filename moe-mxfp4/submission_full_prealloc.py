"""
MoE Full Pre-alloc: Call stages directly with all buffers pre-allocated.
Eliminates ALL per-call tensor allocation:
- Sorting buffers (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
- Intermediate buffer (a2)
- Quant output buffers (a1, a1_scale, a2_q, a2_scale) — handled by fused_dynamic_mxfp4_quant_moe_sort
Plus: global use_nt=False, CK kernel injection for E<=64 d<2048
"""
import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils

_patched = False
_buf_cache = {}  # shape key -> pre-allocated buffers

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Global use_nt=False
    fm.use_nt = lambda token, topk, expert: False

    # block_m for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # CK kernel injection
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _get_bufs(M, topk, E, block_m, model_dim, inter_dim, device):
    """Get or create pre-allocated buffers for a given shape."""
    key = (M, topk, E, block_m, model_dim, inter_dim)
    if key not in _buf_cache:
        max_num_tokens_padded = int(M * topk + E * block_m - topk)
        max_num_m_blocks = (max_num_tokens_padded + block_m - 1) // block_m
        _buf_cache[key] = {
            'sorted_ids': torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            'sorted_weights': torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            'sorted_expert_ids': torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            'num_valid_ids': torch.empty(2, dtype=dtypes.i32, device=device),
            'moe_buf': torch.empty((M, model_dim), dtype=dtypes.bf16, device=device),
            'a2': torch.empty((M, topk, inter_dim), dtype=dtypes.bf16, device=device),
        }
    return _buf_cache[key]


def _run_moe(hidden_states, w1, w2, topk_weights, topk_ids,
             w1_scale, w2_scale, hidden_pad, intermediate_pad, config):
    """Run MoE with all buffers pre-allocated."""
    M = hidden_states.shape[0]
    E = w1.shape[0]
    topk = topk_ids.shape[1]

    _, model_dim, inter_dim = fm.get_inter_dim(w1.shape, w2.shape)
    padded_M = fm.get_padded_M(M)
    block_m = fm.get_block_size_M(padded_M, topk, E, inter_dim)

    is_shuffled = getattr(w1, "is_shuffled", False)
    metadata = fm.get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk,
        dtypes.bf16, dtypes.fp4x2, dtypes.fp4x2,
        QuantType.per_1x32, True, ActivationType.Silu, False,
        hidden_pad, intermediate_pad, is_shuffled,
    )
    block_m = int(metadata.block_m) if metadata.block_m else block_m

    device = hidden_states.device
    bufs = _get_bufs(M, topk, E, block_m, model_dim, inter_dim, device)

    # Phase 1: Sorting (into pre-allocated buffers)
    fwd_fn = aiter.moe_sorting_opus_fwd if hasattr(aiter, 'moe_sorting_opus_fwd') else aiter.moe_sorting_fwd
    fwd_fn(
        topk_ids, topk_weights,
        bufs['sorted_ids'], bufs['sorted_weights'],
        bufs['sorted_expert_ids'], bufs['num_valid_ids'], bufs['moe_buf'],
        E, int(block_m), None, None, 0,
    )

    sorted_ids = bufs['sorted_ids']
    sorted_weights = bufs['sorted_weights']
    sorted_expert_ids = bufs['sorted_expert_ids']
    num_valid_ids = bufs['num_valid_ids']
    moe_buf = bufs['moe_buf']

    # Phase 2: Input quantization
    if M <= 1024:
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=block_m,
        )
    else:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        a1, a1_scale = dynamic_mxfp4_quant(hidden_states, scale=None,
                                            quant_dtype=dtypes.fp4x2)
        a1_scale = fp4_utils.moe_mxfp4_sort(
            a1_scale, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, block_size=block_m,
        )

    # Phase 3: Stage 1 GEMM (gate+up + SiLU)
    a2 = bufs['a2']
    w1_scale_v = w1_scale.view(dtypes.fp8_e8m0)
    metadata.stage1(
        a1, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        a2, topk, block_m=block_m,
        a1_scale=a1_scale, w1_scale=w1_scale_v,
        sorted_weights=None,
    )

    # Phase 4: Inter-stage quantization
    a2_flat = a2.view(-1, inter_dim)
    if M <= 1024:
        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m,
        )
    else:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        a2_q, a2_scale = dynamic_mxfp4_quant(a2_flat, scale=None,
                                              quant_dtype=dtypes.fp4x2,
                                              num_rows_factor=topk)
        a2_scale = fp4_utils.moe_mxfp4_sort(
            a2_scale[:M * topk, :].view(M, topk, -1),
            sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, block_size=block_m,
        )
    a2_q = a2_q.view(M, topk, -1)

    # Phase 5: Stage 2 GEMM (down + weighted sum)
    w2_scale_v = w2_scale.view(dtypes.fp8_e8m0)
    metadata.stage2(
        a2_q, w1, w2,
        sorted_ids, sorted_expert_ids, num_valid_ids,
        moe_buf, topk,
        w2_scale=w2_scale_v, a2_scale=a2_scale,
        block_m=block_m,
        sorted_weights=sorted_weights,
    )

    return moe_buf


_warmup_done = set()

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    key = (M, E)

    # On first call per shape, use standard path for warmup (triggers JIT)
    if key not in _warmup_done:
        _warmup_done.add(key)
        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # Subsequent calls: use optimized direct path
    return _run_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        hidden_pad, intermediate_pad, config,
    )

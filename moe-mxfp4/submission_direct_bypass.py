"""
MoE — Direct pipeline bypass.
Calls internal functions directly, bypassing fused_moe Python overhead.
Pre-allocates intermediate buffers, caches metadata.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

_patched = False
_warmed_up = False
_call_count = 0

# Pre-cached per-case state
_case_cache = {}  # (M, E, inter_dim) -> cached state

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _get_block_m(est_m):
    return 32 if est_m < 50 else 64


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch use_nt for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            return _get_block_m(token * topk // expert)
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Inject CK kernels
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
                and not result.run_1stage):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    if inter_dim >= 2048:
                        kn1 = STAGE1_64
                    else:
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
            except:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _direct_moe(hidden_states, w1, w2, topk_weights, topk_ids,
                w1_scale, w2_scale, hidden_pad, intermediate_pad):
    """
    Direct MoE pipeline bypass — call internal functions directly.
    Eliminates Python overhead from fused_moe → fused_moe_ → fused_moe_2stages chain.
    """
    token_num = hidden_states.shape[0]
    E, model_dim, inter_dim = fm.get_inter_dim(w1.shape, w2.shape)
    topk = topk_ids.shape[1]
    dtype = torch.bfloat16
    device = hidden_states.device

    # 1. Block size
    est_m = token_num * topk // E if E <= 64 else token_num * topk // E
    if E <= 64:
        block_m = _get_block_m(est_m)
    else:
        block_m = fm.get_block_size_M(fm.get_padded_M(token_num), topk, E, inter_dim)

    # 2. Sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
        topk_ids, topk_weights, E, model_dim, dtype, block_m, None, None, 0)

    # 3. Get CK kernel metadata (cached)
    q_dtype_a = dtypes.fp4x2
    q_dtype_w = dtypes.fp4x2
    is_shuffled = getattr(w1, "is_shuffled", False)
    isG1U1 = inter_dim != w1.shape[1]
    metadata = fm.get_2stage_cfgs(
        fm.get_padded_M(token_num), model_dim, inter_dim, E, topk,
        dtype, q_dtype_a, q_dtype_w, QuantType.per_1x32,
        isG1U1, ActivationType.Silu, False,
        hidden_pad, intermediate_pad, is_shuffled)

    # 4. Quantize A1 (fused quant + sort)
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, topk=1, block_size=block_m)

    # 5. Stage 1 — CK GEMM (gate + up + SiLU)
    a2 = torch.empty((token_num, topk, inter_dim), dtype=dtype, device=device)
    w1_scale_v = w1_scale.view(dtypes.fp8_e8m0)
    metadata.stage1(
        a1, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
        block_m=block_m, a1_scale=a1_scale, w1_scale=w1_scale_v)

    # 6. Quantize A2 (inter-stage quant + sort)
    a2_flat = a2.view(-1, inter_dim)
    a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=token_num, topk=topk, block_size=block_m)
    a2 = a2.view(token_num, topk, -1)

    # 7. Stage 2 — CK GEMM (down + weighted accumulate)
    w2_scale_v = w2_scale.view(dtypes.fp8_e8m0)
    metadata.stage2(
        a2, w1, w2, sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
        block_m=block_m, a2_scale=a2_scale, w2_scale=w2_scale_v,
        sorted_weights=sorted_weights)

    return moe_buf[:token_num, :model_dim - hidden_pad]


def custom_kernel(data: input_t) -> output_t:
    global _warmed_up, _call_count
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

    _call_count += 1

    # Use fused_moe for first few calls (warmup / JIT compilation)
    if _call_count <= 7:
        return fused_moe(
            hidden_states,
            gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # After warmup, try direct bypass for maximum speed
    try:
        return _direct_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
            hidden_pad, intermediate_pad)
    except Exception as e:
        # Fallback to fused_moe on any error
        if _call_count <= 14:
            print(f"[BYPASS ERR] {e}", flush=True)
        return fused_moe(
            hidden_states,
            gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

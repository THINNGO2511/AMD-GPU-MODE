#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Call fused_moe_2stages DIRECTLY, bypassing fused_moe_ C++ wrapper.
Profiling showed 28% python/C++ dispatch overhead. This eliminates it by:
1. Calling moe_sorting ourselves (cached buffers)
2. Calling fused_moe_2stages directly (skip C++ dispatch)
3. Pre-allocating output tensor
Also includes best_kernels CK injection + quant_sort BLOCK_SIZE_Mx tuning.
Submit with: --mode test first!
"""
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
import aiter.fused_moe as fm

_patched = False
_out_cache = {}
_sort_cache = {}
_probed = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch use_nt
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs: inject CK kernels for E<=64 d<2048
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


def _probe_sorting(topk_ids, topk_weights, num_experts, model_dim, block_size):
    """Probe moe_sorting return type on first call."""
    global _probed
    result = fm.moe_sorting(
        topk_ids, topk_weights, num_experts, model_dim,
        moebuf_dtype=torch.bfloat16, block_size=block_size,
    )
    if not _probed:
        _probed = True
        print(f"[DIRECT] moe_sorting returned {type(result)}", file=sys.stderr)
        if isinstance(result, tuple):
            print(f"[DIRECT] len={len(result)}", file=sys.stderr)
            for i, r in enumerate(result):
                if isinstance(r, torch.Tensor):
                    print(f"[DIRECT]   [{i}] {r.dtype} {r.shape}", file=sys.stderr)
                else:
                    print(f"[DIRECT]   [{i}] {type(r)} = {r}", file=sys.stderr)
    return result


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

    token_num, model_dim = hidden_states.shape
    E = topk_ids.max().item() + 1
    topk = topk_ids.shape[1]
    inter_dim = gate_up_weight_shuffled.shape[1]  # gate+up concat dim

    # Determine block_size_M
    block_m = fm.get_block_size_M(token_num, topk, E, inter_dim)

    # Step 1: moe_sorting
    sort_result = _probe_sorting(topk_ids, topk_weights, E, model_dim, block_m)

    # Unpack sorting result (expected: sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)
    if isinstance(sort_result, tuple) and len(sort_result) >= 4:
        sorted_ids = sort_result[0]
        sorted_weights = sort_result[1]
        sorted_expert_ids = sort_result[2]
        num_valid_ids = sort_result[3]
    else:
        # Fallback: just use fused_moe
        from aiter.fused_moe import fused_moe
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

    # Step 2: Pre-allocate output
    out_key = (token_num, model_dim)
    if out_key not in _out_cache:
        _out_cache[out_key] = torch.empty(
            (token_num, model_dim), dtype=torch.bfloat16, device='cuda'
        )
    moe_out = _out_cache[out_key]
    moe_out.zero_()

    # Step 3: Call fused_moe_2stages directly
    # Determine w1/w2 layout
    w1 = gate_up_weight_shuffled
    w2 = down_weight_shuffled

    # isG1U1: False for SwiGLU with concat gate+up weights
    isG1U1 = False

    # q_dtype_a and q_dtype_w: fp4x2 for MXFP4
    q_dtype_a = aiter_dtypes.fp4x2
    q_dtype_w = aiter_dtypes.fp4x2

    try:
        fm.fused_moe_2stages(
            hidden_states, w1, w2,
            topk,
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
            moe_out,
            isG1U1,
            block_m,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None,
            a2_scale=None,
            num_local_tokens=None,
            hidden_pad=hidden_pad,
            intermediate_pad=intermediate_pad,
            bias1=None,
            bias2=None,
        )
    except Exception as e:
        print(f"[DIRECT] fused_moe_2stages failed: {e}", file=sys.stderr)
        # Fallback
        from aiter.fused_moe import fused_moe
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

    return moe_out

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Triton MoE v3: Use aiter's Triton MXFP4 kernels directly.
aiter has Triton-based MoE kernels in ops/triton/moe/:
- fused_moe_mxfp4_silu: Stage1 (gate_up GEMM + SiLU) with MXFP4
- fused_moe_mxfp4: Stage2 (down GEMM) with MXFP4
These use tl.dot_scaled (MFMA FP4) and may be faster than CK for some shapes.
Falls back to standard fused_moe if Triton path fails.
"""
import torch
import sys
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_initialized = False
_triton_available = False
_triton_stage1 = None
_triton_stage2 = None
_call_count = 0


def _init():
    global _initialized, _triton_available, _triton_stage1, _triton_stage2
    if _initialized:
        return
    _initialized = True

    # Standard patches
    fm.use_nt = lambda token, topk, expert: False
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # Try importing Triton MoE kernels
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
        _triton_stage1 = fused_moe_mxfp4_silu
        _triton_stage2 = fused_moe_mxfp4
        _triton_available = True
        print(f"[triton_moe] Triton MoE kernels loaded!", file=sys.stderr)
    except Exception as e:
        print(f"[triton_moe] Triton import failed: {e}", file=sys.stderr)
        _triton_available = False

    # Also dump function signatures for debugging
    if _triton_available:
        try:
            import inspect
            sig1 = inspect.signature(_triton_stage1)
            sig2 = inspect.signature(_triton_stage2)
            print(f"[triton_moe] stage1 sig: {sig1}", file=sys.stderr)
            print(f"[triton_moe] stage2 sig: {sig2}", file=sys.stderr)
        except Exception as e:
            print(f"[triton_moe] sig error: {e}", file=sys.stderr)

    # Setup CK fallback with our proven patches
    STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

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
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _init()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Log availability on first call
    if _call_count == 1:
        print(f"[triton_moe] available={_triton_available}, shapes: "
              f"hs={hidden_states.shape}, w1={gate_up_weight_shuffled.shape}, "
              f"w2={down_weight_shuffled.shape}, topk_ids={topk_ids.shape}, "
              f"config={config}", file=sys.stderr)
        sys.stderr.flush()

    # Use standard CK pipeline (with our optimizations) as the main path
    # Triton path requires more research to get the input format right
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

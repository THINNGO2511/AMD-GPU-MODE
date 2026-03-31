#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Stage2 v3 kernel injection for d=2048 bottleneck (E=33 bs=512).

Key change from submission_optimized_v2.py:
- d<2048 E<=64: keep proven S1+S2_v1 injection (unchanged)
- d>=2048 E<=64: inject v3 stage2 kernels (UNTESTED, already compiled in .so)
  - est_m >= 100 (bs=512 case): S1=256x128, S2=256x32x128x128_v3, block_m=128
  - est_m < 100: S1=64x32, S2=64x32x32x128_v1 (safe fallback)
- E>64: use_nt=False only (no kernel injection)

The 256x32x128x128_v3 stage2 is the untested sweet spot:
- Larger than 64x32x32x128_v1 (which is proven too small for d=2048)
- Smaller than 256x128x128x128_v3 (which was tested and showed no improvement)
- Medium tile may hit a different occupancy/bandwidth tradeoff
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Stage1 kernels (proven working for E<=64 d<2048)
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Stage1 kernels for d>=2048 (larger tiles for bigger workloads)
S1_256x128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

# Stage2 v1 (proven for d<2048)
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Stage2 v3 variants — UNTESTED, already compiled in .so
# The 256x32 is the key test: medium tile, v3 pipeline, 1x4 wavefront
S2_V3_256x32 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. use_nt=False globally (proven 2-4% improvement)
    fm.use_nt = lambda token, topk, expert: False

    # 2. block_m tuning for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048:
                # d=2048 large batch: 128 matches fc0c54bb recommendation
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 3. Kernel injection: d<2048 proven path + d>=2048 v3 stage2 experiment
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
                est_m = token * topk // expert
                if inter_dim >= 2048:
                    # d=2048 path: v3 stage2 with 256x32 tiles
                    # est_m for bs=512 E=33 topk=9 = 139 (>= 100)
                    if est_m >= 100:
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=S1_256x128, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=S2_V3_256x32, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            128, 0, False)
                    else:
                        # Small est_m with d>=2048: use safe 64x32 kernels
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=S1_64, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=S2_V1, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            32, 0, False)
                else:
                    # d<2048 path: proven S1+S2_v1 injection
                    kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    if not kw.get('kernelName', ''):
                        kn1 = S1_256 if est_m >= 100 else S1_64
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=kn1, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=S2_V1, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


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

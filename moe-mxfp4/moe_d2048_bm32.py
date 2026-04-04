#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — d=2048 CK injection with block_m=32.

Key finding from this session: CK kernels CRASH at block_m>=64 on E=33.
block_m=32 is the ONLY value that works (proven in submission_optimized_v2).

Previous moe_d2048_medtile used block_m=64 → crashed.
This version: 256x32 tiles for d=2048 but with block_m=32 (safe).

d=2048 at 333μs is 2x the other shapes. Even a 20% improvement
drops the geomean significantly.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_p = False
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
S2_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _patch():
    global _p
    if _p: return
    _p = True
    fm.use_nt = lambda t, k, e: False

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 32  # CHANGED: was 128, CK crashes at 128. Use 32.
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    orig_g2s = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk,
                dtype, q_dtype_a, q_dtype_w, q_type,
                use_g1u1, activation, doweight_stage1,
                hidden_pad, intermediate_pad, is_shuffled=True):
        r = orig_g2s(token, model_dim, inter_dim, expert, topk,
                     dtype, q_dtype_a, q_dtype_w, q_type,
                     use_g1u1, activation, doweight_stage1,
                     hidden_pad, intermediate_pad, is_shuffled)
        if expert <= 64 and q_type == QuantType.per_1x32 and not r.run_1stage:
            kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
            if not kw.get('kernelName', ''):
                est_m = token * topk // expert
                if inter_dim >= 2048:
                    # d=2048: 256x32 tiles + block_m=32 (CRASH-SAFE)
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=S1_256, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_256, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
                else:
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
        return r
    fm.get_2stage_cfgs = new_g2s
    fm.cfg_2stages = None

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hs, guw, dw, guws, dws, guw_sh, dw_sh, guws_sh, dws_sh, tw, ti, cfg) = data
    return fused_moe(
        hs, guw_sh, dw_sh, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guws_sh, w2_scale=dws_sh,
        a1_scale=None, a2_scale=None,
        hidden_pad=cfg["d_hidden_pad"]-cfg["d_hidden"],
        intermediate_pad=cfg["d_expert_pad"]-cfg["d_expert"],
    )

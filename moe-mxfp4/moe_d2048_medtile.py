#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — CK injection with medium tiles for d=2048.

d=2048 at 333μs dominates the geomean. The 256x128 tiles CRASHED.
Try the MEDIUM tiles: 256x32x128x128 for both stage1 and stage2.
These are proven for E=257 in the dsv3 CSV.

For d<2048: same as v2 (CK injection with correct block_m).
Block_m for d=2048: use library default (128 for est_m=139).
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
# Medium-tile stage2 for d=2048 (proven for E=257 in dsv3 CSV)
S2_256 = "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _patch():
    global _p
    if _p: return
    _p = True
    fm.use_nt = lambda t, k, e: False
    _orig_bsm = fm.get_block_size_M
    _orig_g2s = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def _g2s(token, model_dim, inter_dim, expert, topk,
             dtype, q_dtype_a, q_dtype_w, q_type,
             use_g1u1, activation, doweight_stage1,
             hidden_pad, intermediate_pad, is_shuffled=True):
        r = _orig_g2s(token, model_dim, inter_dim, expert, topk,
                      dtype, q_dtype_a, q_dtype_w, q_type,
                      use_g1u1, activation, doweight_stage1,
                      hidden_pad, intermediate_pad, is_shuffled)
        if expert <= 64 and q_type == QuantType.per_1x32 and not r.run_1stage:
            kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
            if not kw.get('kernelName', ''):
                est_m = token * topk // expert
                bm = _orig_bsm(token, topk, expert, inter_dim)

                if inter_dim >= 2048:
                    # d=2048: medium tiles (256x32), NOT the 256x128 that crashed
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=S1_256, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_256, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        bm, 0, False)
                else:
                    # d<2048: same as before, small/medium tiles
                    kn1 = S1_256 if est_m >= 100 else S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        bm, 0, False)
        return r
    fm.get_2stage_cfgs = _g2s
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

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — CK injection with CORRECT block_m.

BUG FIX: submission_optimized_v2 hardcoded block_m=32 in MOEMetadata
for ALL CK-injected shapes. For E=33 d=512 bs=512, the library default
block_m=128. We were using 32 = 4x wrong.

FIX: Call original get_block_size_M to get library default block_m,
pass THAT to MOEMetadata instead of hardcoding.

d=2048: NO injection (library default kernels, no risk).
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
        # Only inject for E<=64, d<2048, FP4, 2-stage
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not r.run_1stage and inter_dim < 2048):
            kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
            if not kw.get('kernelName', ''):
                est_m = token * topk // expert
                kn1 = S1_256 if est_m >= 100 else S1_64
                # USE LIBRARY DEFAULT block_m — not hardcoded 32!
                bm = _orig_bsm(token, topk, expert, inter_dim)
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

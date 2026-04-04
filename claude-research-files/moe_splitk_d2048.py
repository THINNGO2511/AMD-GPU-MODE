#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — SplitK=2 for E=33 d=2048.

The d=2048 shape has E=33 with ~15 tokens/expert. At block_m=32,
each expert's tile is half-empty. CUs are underutilized because M is
too small to fill the GPU.

SplitK=2 decomposes the K dimension, doubling the number of thread
groups. More parallelism → better CU utilization for small M.

All other shapes keep splitk=0 (they're fine as-is).
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

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50: return 32
            elif inter_dim >= 2048 and est_m >= 100: return 128
            else: return 64
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
        # CK injection for E<=64 d<2048 (proven)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not r.run_1stage and inter_dim < 2048):
            kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
            if not kw.get('kernelName', ''):
                est_m = token * topk // expert
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
        # SplitK=2 for E<=64 d>=2048 (new experiment)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and inter_dim >= 2048 and not r.run_1stage):
            try:
                return fm.MOEMetadata(r.stage1, r.stage2, r.block_m, 2, False)
            except Exception:
                pass
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

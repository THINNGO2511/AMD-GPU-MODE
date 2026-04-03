#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe ALL available CK kernel names for E=33 d=2048.

We've only used 3 kernel names (S1_64, S1_256, S2_V1) for d<2048.
The runner has 1024+ fmoe .co kernels. There might be BETTER kernels
for E=33 d=2048 (our bottleneck at ~333μs).

This file:
1. Lists all available 2-stage kernel names on the runner
2. Tries the larger tile variants for d=2048 (128x, 256x)
3. Reports what works and what fails

Key kernel naming pattern:
moe_ck2stages_gemm1_{tile}_MulABScaleShuffled_{version}_..._silu_FP4X2_FP4X2_B16
Tiles seen: 64x32x32x128, 256x32x128x128

For d=2048 we need bigger N tiles. Maybe:
- 64x32x64x128 (N=64)
- 256x32x256x128 (N=256)
- or the _Nswizzle variants
"""
import torch
import os
import glob
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_p = False
_call = 0

# Proven kernels for d<2048
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Try these for d=2048 Stage 1 (larger tiles)
S1_D2048_CANDIDATES = [
    # 64-wide tiles with different N
    "moe_ck2stages_gemm1_64x32x64x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "moe_ck2stages_gemm1_64x32x32x256_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    # 256-wide tiles
    "moe_ck2stages_gemm1_256x32x256x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "moe_ck2stages_gemm1_256x32x64x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
]

# Try these for d=2048 Stage 2 (larger tiles)
S2_D2048_CANDIDATES = [
    "moe_ck2stages_gemm2_64x32x64x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "moe_ck2stages_gemm2_256x32x256x128_1x4_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
]


def _patch():
    global _p
    if _p: return
    _p = True

    # List available kernel .co files to find correct names
    try:
        co_files = glob.glob("/home/runner/aiter/hsa/gfx950/fmoe/*.co")
        ck2s_g1 = sorted(set(os.path.basename(f).replace('.co','') for f in co_files if 'ck2stages_gemm1' in f and 'FP4' in f and 'silu' in f))
        ck2s_g2 = sorted(set(os.path.basename(f).replace('.co','') for f in co_files if 'ck2stages_gemm2' in f and 'FP4' in f))
        print(f"[MOE] Available CK 2-stage gemm1 FP4 silu kernels: {len(ck2s_g1)}", flush=True)
        for k in ck2s_g1[:20]:
            print(f"[MOE]   G1: {k}", flush=True)
        print(f"[MOE] Available CK 2-stage gemm2 FP4 kernels: {len(ck2s_g2)}", flush=True)
        for k in ck2s_g2[:20]:
            print(f"[MOE]   G2: {k}", flush=True)
    except Exception as e:
        print(f"[MOE] Failed to list .co files: {e}", flush=True)

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

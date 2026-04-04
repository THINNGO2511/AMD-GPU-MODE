#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Custom AITER_CONFIG_FMOE CSV with E=33 entries for cu_num=256.

KEY FACT: tuned_fmoe.csv has ZERO E=33 entries at cu_num=256.
The runtime falls back to generic heuristics for E=33, which may
select suboptimal kernels.

This provides a CSV with entries specifically for our benchmark shapes:
- E=33 d=512 bs=16/128/512 at cu_num=256
- E=33 d=2048 bs=512 at cu_num=256

Kernel names from our proven CK injection (submission_optimized_v2):
- Stage1 small: 64x32x32x128 (est_m < 50)
- Stage1 large: 256x32x128x128 (est_m >= 100)
- Stage2: 64x32x32x128_v1

We DON'T override E=257 entries (those already work at 14 entries, cu_num=256).
Combined with use_nt=False.

NO monkey-patching of get_2stage_cfgs — let the CSV do the work.
"""
import os
import tempfile
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_p = False

# Known working kernel names
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _build_csv():
    """Build CSV with E=33 entries at cu_num=256."""
    header = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"
    rows = [header]

    # Common: silu activation, bf16 dtype, fp4x2 quant, per_1x32
    # act_type=1 (silu), dtype=bf16
    # q_dtype_a=fp4x2, q_dtype_w=fp4x2, q_type=per_1x32
    # use_g1u1=0, doweight_stage1=0

    # E=33 d=512 shapes (token = bs * topk = bs * 9)
    # bs=16: token=144, est_m=144*9/33=39 -> block_m=32, S1_64
    # bs=128: token=1152, est_m=1152*9/33=314 -> block_m=32, S1_256
    # bs=512: token=4608, est_m=4608*9/33=1257 -> block_m=32, S1_256
    shapes_e33_d512 = [
        (16, 32, S1_64),
        (128, 32, S1_256),
        (512, 32, S1_256),
    ]
    for bs, bm, s1 in shapes_e33_d512:
        rows.append(f"256,{bs},7168,512,33,9,1,bf16,fp4x2,fp4x2,per_1x32,0,0,{bm},0,0.0,{s1},0,0.0,{S2_V1},0,0.0,0,0.0,0.0")

    # E=33 d=2048 shape (bs=512 only)
    # Don't inject d=2048 kernel — let default heuristic handle it
    # (All our d=2048 injection attempts made it worse or crashed)

    return "\n".join(rows) + "\n"


def _patch():
    global _p
    if _p: return
    _p = True

    # Write custom CSV
    csv_content = _build_csv()
    csv_path = "/tmp/custom_e33_fmoe.csv"
    with open(csv_path, 'w') as f:
        f.write(csv_content)

    # Set env var BEFORE any fused_moe imports read the config
    # Use os pathsep to APPEND our CSV to the existing one
    existing = os.environ.get("AITER_CONFIG_FMOE", "")
    if existing:
        os.environ["AITER_CONFIG_FMOE"] = f"{existing}{os.pathsep}{csv_path}"
    else:
        os.environ["AITER_CONFIG_FMOE"] = csv_path

    fm.use_nt = lambda t, k, e: False

    # Clear cached configs so CSV gets re-read
    fm.cfg_2stages = None
    try:
        import functools
        orig = fm.get_2stage_cfgs.__wrapped__
        fm.get_2stage_cfgs = functools.lru_cache(maxsize=2048)(orig)
    except Exception:
        pass


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

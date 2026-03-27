#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Inject optimal configs into CSV for E=33 cases.
Instead of monkey-patching get_2stage_cfgs, directly add rows to the
tuned CSV file so the native config lookup succeeds.
This avoids the overhead of our Python wrapper around get_2stage_cfgs.
"""
import os
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Best CK kernel names for MXFP4 SiLU
KN1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
KN1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
KN2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _inject_csv_configs():
    """Inject optimal configs for E=33 cases into the tuned CSV file."""
    try:
        from aiter.fused_moe import AITER_CONFIGS
        csv_path = AITER_CONFIGS.AITER_CONFIG_FMOE_FILE
    except Exception:
        csv_path = "/tmp/aiter_configs/tuned_fmoe.csv"

    if not os.path.exists(csv_path):
        return

    # Read existing content
    with open(csv_path, 'r') as f:
        content = f.read()

    # Config template for E=33 d=512 cases
    # CSV columns: cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw
    base = "ActivationType.Silu,torch.bfloat16,torch.float4_e2m1fn_x2,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0"

    # E=33 d=512 configs (model_dim=7168, inter_dim=512)
    new_rows = []
    for token in [16, 128, 512]:
        est_m = token * 9 // 33
        kn1 = KN1_256 if est_m >= 100 else KN1_64
        bm = 32 if est_m < 50 else 64
        row = f"256,{token},7168,512,33,9,{base},{bm},0,50.0,{kn1},0.0%,30.0,{KN2_32},0.0%,80.0,0,100.0,5000.0"
        # Only add if not already present
        key = f"256,{token},7168,512,33,9"
        if key not in content:
            new_rows.append(row)

    # E=33 d=2048 configs (model_dim=7168, inter_dim=2048)
    # DON'T inject kernel names for d=2048 — let auto-select handle it
    # But DO set block_m and ksplit=0 to avoid bad configs
    for token in [512]:
        key = f"256,{token},7168,2048,33,9"
        if key not in content:
            # Use empty kernel names so CK auto-selects
            bm = 64  # Optimal from get_block_size_M heuristic
            row = f"256,{token},7168,2048,33,9,{base},{bm},0,100.0,,0.0%,100.0,,0.0%,200.0,0,100.0,5000.0"
            new_rows.append(row)

    if new_rows:
        with open(csv_path, 'a') as f:
            for row in new_rows:
                f.write('\n' + row)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Inject CSV configs BEFORE any fused_moe call
    _inject_csv_configs()

    # Force re-read of CSV
    fm.cfg_2stages = None

    # use_nt=False for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # block_m for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # ksplit=0 for E<=64
    try:
        orig_ksplit = fm.get_ksplit
        fm.get_ksplit = lambda t, k, e, i, m: 0 if e <= 64 else orig_ksplit(t, k, e, i, m)
    except Exception:
        pass


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

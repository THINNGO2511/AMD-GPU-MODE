#!/usr/bin/env python3
"""
MoE Auto-Sweep: Systematically test CK kernel × block_m × ksplit combinations.
Focus on d=2048 (333μs bottleneck).

Usage:
    python3 autosweep_moe.py
"""
import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).parent
MOE_DIR = REPO / "moe-mxfp4"
LOG_FILE = REPO / "auto_research_logs" / "moe_sweep.jsonl"
POPCORN = shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli")

SLEEP_BETWEEN = 620  # ~10 min

# Available CK kernels
S1_KERNELS = {
    "s1_64": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256": "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256x64": "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256x128": "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
}

S2_KERNELS = {
    "s2_v1": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "s2_256": "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
}

BLOCK_M_OPTIONS = [32, 64, 128]

# Generate configs to test for d=2048
SWEEP_CONFIGS = []
for s1_name, s1_kernel in S1_KERNELS.items():
    for s2_name, s2_kernel in S2_KERNELS.items():
        for bm in BLOCK_M_OPTIONS:
            SWEEP_CONFIGS.append({
                "name": f"{s1_name}_{s2_name}_bm{bm}",
                "s1": s1_kernel, "s2": s2_kernel, "bm": bm,
            })

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Auto-sweep MoE: {config_name}"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

S1_KERNEL = "{s1_kernel}"
S2_KERNEL = "{s2_kernel}"
TARGET_BM = {block_m}

# Proven kernels for E<=64 d<2048
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _patch():
    global _patched
    if _patched: return
    _patched = True
    fm.use_nt = lambda token, topk, expert: False
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if inter_dim >= 2048:
                return TARGET_BM  # <-- SWEEP THIS
            elif est_m < 50: return 32
            else: return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh)
        if expert <= 64 and qt == QuantType.per_1x32 and not r.run_1stage:
            try:
                est = token * topk // expert
                if inter_dim >= 2048:
                    # INJECT for d=2048 with sweep config
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1, kernelName=S1_KERNEL, activation=act, quant_type=qt, dtype=dtype, splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=S2_KERNEL, activation=act, quant_type=qt, use_non_temporal_load=False),
                        TARGET_BM, 0, False)
                else:
                    kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {{}}
                    if not kw.get('kernelName', ''):
                        kn = S1_256 if est >= 100 else S1_64
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1, kernelName=kn, activation=act, quant_type=qt, dtype=dtype, splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd, kernelName=S2_V1, activation=act, quant_type=qt, use_non_temporal_load=False),
                            32, 0, False)
            except: pass
        return r
    fm.get_2stage_cfgs = new_g2s
    fm.cfg_2stages = None

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data
    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip)
'''


def generate_submission(config, idx):
    code = SUBMISSION_TEMPLATE.format(
        config_name=config["name"],
        s1_kernel=config["s1"],
        s2_kernel=config["s2"],
        block_m=config["bm"],
    )
    filepath = MOE_DIR / f"sweep_{idx:04d}.py"
    filepath.write_text(code)
    return filepath


def submit(filepath, mode="benchmark"):
    cmd = [POPCORN, "submit", "--gpu", "MI355X",
           "--leaderboard", "amd-moe-mxfp4", "--mode", mode,
           str(filepath), "--no-tui"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def log_result(entry):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    print(f"=== MoE Auto-Sweep Started {datetime.now()} ===")
    print(f"Testing {len(SWEEP_CONFIGS)} configs for d=2048")

    for idx, config in enumerate(SWEEP_CONFIGS, 1):
        filepath = generate_submission(config, idx)
        print(f"\n[{idx}/{len(SWEEP_CONFIGS)}] {config['name']}")

        output = submit(filepath, mode="test")
        if "Rate limit" in output:
            print(f"  Rate limited. Waiting {SLEEP_BETWEEN}s...")
            time.sleep(SLEEP_BETWEEN)
            output = submit(filepath, mode="test")

        passed = "passed" in output.lower() or "success" in output.lower()
        print(f"  {'PASSED' if passed else 'FAILED'}")

        if passed:
            # Submit to benchmark for timing
            print(f"  Benchmarking...")
            time.sleep(SLEEP_BETWEEN)
            bench_output = submit(filepath, mode="benchmark")
            log_result({
                "timestamp": datetime.now().isoformat(),
                "config": config["name"],
                "s1": config["s1"].split("_")[3],  # tile size
                "s2": config["s2"].split("_")[3],
                "bm": config["bm"],
                "passed": True,
                "bench_preview": bench_output[:300],
            })
        else:
            log_result({
                "timestamp": datetime.now().isoformat(),
                "config": config["name"],
                "passed": False,
                "error": output[:300],
            })

        time.sleep(SLEEP_BETWEEN)

    print(f"\n=== MoE Sweep Complete ===")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GEMM Auto-Sweep: Systematically test config combinations for gemm_a16wfp4.
Runs independently of Claude Code. Submit via popcorn-cli.

Usage:
    python3 autosweep_gemm.py

This generates submission files with different configs, submits them,
and logs results. Runs continuously respecting rate limits.
"""
import os
import sys
import json
import time
import subprocess
import shutil
import itertools
from pathlib import Path
from datetime import datetime

REPO = Path(__file__).parent
GEMM_DIR = REPO / "mxfp4-mm"
LOG_FILE = REPO / "auto_research_logs" / "gemm_sweep.jsonl"

# Rate limits
SUBMISSIONS_PER_HOUR = 6
LEADERBOARD_PER_HOUR = 1
SLEEP_BETWEEN = 620  # ~10 min between submissions

# Config space to sweep (for gemm_a16wfp4)
# Focus on K=7168 first (dominates geomean)
SWEEP_CONFIGS = {
    # (M, N, K): list of config dicts to try
    (16, 2112, 7168): [
        # Current best: BM=8 BN=64 BK=512 KS=8 stages=2
        # Try stages=3
        {"BM": 8, "BN": 64, "BK": 512, "KS": 8, "SKBS": 1024, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        # Try KSPLIT=16
        {"BM": 8, "BN": 64, "BK": 512, "KS": 16, "SKBS": 512, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 8, "BN": 64, "BK": 512, "KS": 16, "SKBS": 512, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        # Try BN=128
        {"BM": 8, "BN": 128, "BK": 512, "KS": 8, "SKBS": 1024, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 8, "BN": 128, "BK": 512, "KS": 8, "SKBS": 1024, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        # Try BM=16
        {"BM": 16, "BN": 64, "BK": 512, "KS": 8, "SKBS": 1024, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 16, "BN": 64, "BK": 512, "KS": 8, "SKBS": 1024, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        # Try waves_per_eu=1
        {"BM": 8, "BN": 64, "BK": 512, "KS": 8, "SKBS": 1024, "S": 2, "W": 4, "WPE": 1, "MI": 16},
        # Try BK=256 with more KSPLIT
        {"BM": 8, "BN": 64, "BK": 256, "KS": 16, "SKBS": 512, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 8, "BN": 64, "BK": 256, "KS": 16, "SKBS": 512, "S": 3, "W": 4, "WPE": 2, "MI": 16},
    ],
    (64, 7168, 2048): [
        # Current: default (no custom config)
        # Try various configs
        {"BM": 64, "BN": 128, "BK": 256, "KS": 2, "SKBS": 1024, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 64, "BN": 128, "BK": 256, "KS": 2, "SKBS": 1024, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 64, "BN": 64, "BK": 256, "KS": 4, "SKBS": 512, "S": 2, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 64, "BN": 64, "BK": 256, "KS": 4, "SKBS": 512, "S": 3, "W": 4, "WPE": 2, "MI": 16},
        {"BM": 32, "BN": 128, "BK": 256, "KS": 2, "SKBS": 1024, "S": 2, "W": 4, "WPE": 2, "MI": 32},
        {"BM": 32, "BN": 128, "BK": 256, "KS": 2, "SKBS": 1024, "S": 3, "W": 4, "WPE": 2, "MI": 32},
    ],
    (4, 2880, 512): [
        {"BM": 8, "BN": 128, "BK": 128, "KS": 1, "SKBS": 512, "S": 2, "W": 4, "WPE": 1, "MI": 16},
        {"BM": 8, "BN": 128, "BK": 128, "KS": 1, "SKBS": 512, "S": 3, "W": 4, "WPE": 1, "MI": 16},
        {"BM": 16, "BN": 64, "BK": 128, "KS": 1, "SKBS": 512, "S": 3, "W": 4, "WPE": 1, "MI": 16},
    ],
}

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Auto-sweep: testing config {config_name} for shape ({m},{n},{k})"""
from task import input_t, output_t
import torch
import sys
import time

_ref = None
_raw = None
_bq = None
_y = {{}}
_warmed = False

_CONFIG = {{
    "BLOCK_SIZE_M": {BM}, "BLOCK_SIZE_N": {BN}, "BLOCK_SIZE_K": {BK},
    "GROUP_SIZE_M": 1, "num_warps": {W}, "num_stages": {S},
    "waves_per_eu": {WPE}, "matrix_instr_nonkdim": {MI}, "cache_modifier": None,
    "NUM_KSPLIT": {KS}, "SPLITK_BLOCK_SIZE": {SKBS},
}}

_K7168_DEFAULT = {{
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}}

TARGET_M, TARGET_N, TARGET_K = {m}, {n}, {k}

def _unshuffle(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)

def _prewarm():
    global _warmed
    if _warmed: return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try: dynamic_mxfp4_quant(torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda'))
        except: pass
    torch.cuda.synchronize()

def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _bq
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _ref is not B_scale_sh:
        _ref = B_scale_sh; _raw = _unshuffle(B_scale_sh); _bq = B_q.view(torch.uint8); _prewarm()
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y: _y[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    # Use custom config for target shape, K7168 default for K=7168, None for others
    if m == TARGET_M and n == TARGET_N and k == TARGET_K:
        cfg = _CONFIG
    elif k == 7168:
        cfg = _K7168_DEFAULT
    else:
        cfg = None
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, y=_y[key], config=cfg)
'''


def generate_submission(m, n, k, config, idx):
    """Generate a submission file for a specific config."""
    config_name = f"BM{config['BM']}_BN{config['BN']}_BK{config['BK']}_KS{config['KS']}_S{config['S']}_W{config['W']}_WPE{config['WPE']}_MI{config['MI']}"

    code = SUBMISSION_TEMPLATE.format(
        config_name=config_name, m=m, n=n, k=k,
        BM=config['BM'], BN=config['BN'], BK=config['BK'],
        KS=config['KS'], SKBS=config['SKBS'],
        S=config['S'], W=config['W'], WPE=config['WPE'], MI=config['MI'],
    )

    filepath = GEMM_DIR / f"sweep_{idx:04d}.py"
    filepath.write_text(code)
    return filepath, config_name


def submit(filepath, mode="benchmark"):
    """Submit via popcorn-cli and return output."""
    cmd = [
        shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli"),
        "submit", "--gpu", "MI355X",
        "--leaderboard", "amd-mxfp4-mm",
        "--mode", mode,
        str(filepath),
        "--no-tui",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def log_result(entry):
    """Append result to log file."""
    LOG_FILE.parent.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    print(f"=== GEMM Auto-Sweep Started {datetime.now()} ===")
    print(f"Config space: {sum(len(v) for v in SWEEP_CONFIGS.values())} configs across {len(SWEEP_CONFIGS)} shapes")

    idx = 0
    for (m, n, k), configs in SWEEP_CONFIGS.items():
        for config in configs:
            idx += 1
            filepath, config_name = generate_submission(m, n, k, config, idx)

            print(f"\n[{idx}] Testing M={m} N={n} K={k} {config_name}")
            print(f"    Submitting benchmark...")

            output = submit(filepath, mode="benchmark")

            if "Rate limit" in output:
                wait = 620
                print(f"    Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                output = submit(filepath, mode="benchmark")

            entry = {
                "timestamp": datetime.now().isoformat(),
                "shape": f"{m}x{n}x{k}",
                "config": config_name,
                "config_dict": config,
                "output_preview": output[:500],
            }
            log_result(entry)

            print(f"    Done. Output: {output[:200]}")

            # Respect rate limits
            time.sleep(SLEEP_BETWEEN)

    print(f"\n=== Sweep Complete. {idx} configs tested. ===")
    print(f"Results in: {LOG_FILE}")


if __name__ == "__main__":
    main()

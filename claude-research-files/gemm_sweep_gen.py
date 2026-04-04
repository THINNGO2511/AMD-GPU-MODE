#!/usr/bin/env python3
"""
GEMM Config Sweep Generator
Generates submission files with injected Triton configs for the two bottleneck shapes:
  - Shape 2: M=16, N=2112, K=7168 (currently 13.6μs)
  - Shape 5: M=64, N=7168, K=2048 (currently 14.1μs)

The submission file writes a JSON config to the runner's writable config directory
before the benchmark runs.

Usage: python gemm_sweep_gen.py [--start INDEX] [--shape {k7168,k2048,both}]
"""

import json
import os
import itertools
import sys

REPO = "/home/claude/AMD-GPU-MODE"
SWEEP_DIR = os.path.join(REPO, "mxfp4-mm", "sweep_configs")
os.makedirs(SWEEP_DIR, exist_ok=True)

# Define shapes and their config filenames on the runner
# get_gemm_config looks for: gfx950-GEMM-A16WFP4-N={N}-K={2*K}.json
SHAPES = {
    "k7168": {
        "M": 16, "N": 2112, "K": 7168,
        "config_file": "gfx950-GEMM-A16WFP4-N=2112-K=14336.json",
        "current_best": {"BM": 16, "BN": 64, "BK": 512, "warps": 4, "stages": 2, "waves": 2, "GROUP_M": 1},
    },
    "k2048": {
        "M": 64, "N": 7168, "K": 2048,
        "config_file": "gfx950-GEMM-A16WFP4-N=7168-K=4096.json",
        "current_best": {"BM": 16, "BN": 128, "BK": 512, "warps": 8, "stages": 2, "waves": 4, "GROUP_M": 1},
    },
}

# Sweep parameters — BK is always 512 for MXFP4, KSPLIT always 1 (>1 is dead)
SWEEP_PARAMS = {
    "k7168": {
        "BLOCK_SIZE_M":  [4, 8, 16, 32],
        "BLOCK_SIZE_N":  [32, 64, 128, 256],
        "BLOCK_SIZE_K":  [512],       # always 512 for FP4
        "GROUP_SIZE_M":  [1, 2, 4, 8],
        "num_warps":     [2, 4, 8],
        "num_stages":    [1, 2],       # stages > 2 for K>512 is dead
        "waves_per_eu":  [1, 2, 4],
    },
    "k2048": {
        "BLOCK_SIZE_M":  [4, 8, 16, 32, 64],
        "BLOCK_SIZE_N":  [32, 64, 128, 256],
        "BLOCK_SIZE_K":  [512],
        "GROUP_SIZE_M":  [1, 2, 4, 8],
        "num_warps":     [2, 4, 8],
        "num_stages":    [1, 2],
        "waves_per_eu":  [1, 2, 4],
    },
}


def is_valid_config(cfg, shape_key):
    """Filter out obviously invalid configs."""
    bm = cfg["BLOCK_SIZE_M"]
    bn = cfg["BLOCK_SIZE_N"]
    warps = cfg["num_warps"]

    # Warp count must make sense for tile size
    # Each warp handles a sub-tile. Total threads = warps * 64 (wave_size)
    # Minimum useful work: at least 1 MFMA per warp
    total_threads = warps * 64
    tile_elements = bm * bn

    # Too many warps for small tiles = waste
    if total_threads > tile_elements * 4:
        return False

    # Too few warps for large tiles = underutilization
    if tile_elements > 16384 and warps < 4:
        return False

    # BN=256 with only 2 warps is almost certainly bad
    if bn >= 256 and warps < 4:
        return False

    return True


def generate_configs(shape_key):
    """Generate all valid config combinations for a shape."""
    params = SWEEP_PARAMS[shape_key]
    keys = list(params.keys())
    values = [params[k] for k in keys]

    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        if is_valid_config(cfg, shape_key):
            configs.append(cfg)

    return configs


def prioritize_configs(configs, shape_key):
    """Sort configs: configs closest to current best first, then explore outward."""
    best = SHAPES[shape_key]["current_best"]

    def distance(cfg):
        """Manhattan distance from current best in log-space for sizes, linear for others."""
        d = 0
        d += abs(cfg["BLOCK_SIZE_M"] - best["BM"]) / max(best["BM"], 1)
        d += abs(cfg["BLOCK_SIZE_N"] - best["BN"]) / max(best["BN"], 1)
        d += abs(cfg["num_warps"] - best["warps"]) / max(best["warps"], 1)
        d += abs(cfg["num_stages"] - best["stages"])
        d += abs(cfg["waves_per_eu"] - best["waves"]) / max(best["waves"], 1)
        d += abs(cfg["GROUP_SIZE_M"] - best["GROUP_M"]) / max(best["GROUP_M"], 1)
        return d

    # Sort: exact current best first, then nearby, then far away
    # But skip the exact current best since we already have that result
    configs_with_dist = [(c, distance(c)) for c in configs]
    configs_with_dist.sort(key=lambda x: x[1])

    # Remove exact match with current best (already tested)
    result = []
    for cfg, dist in configs_with_dist:
        if (cfg["BLOCK_SIZE_M"] == best["BM"] and
            cfg["BLOCK_SIZE_N"] == best["BN"] and
            cfg["num_warps"] == best["warps"] and
            cfg["num_stages"] == best["stages"] and
            cfg["waves_per_eu"] == best["waves"] and
            cfg["GROUP_SIZE_M"] == best["GROUP_M"]):
            continue
        result.append(cfg)

    return result


def make_submission(cfg, shape_key, index):
    """Generate a submission .py file that injects the config and runs all 6 shapes."""
    shape = SHAPES[shape_key]
    config_filename = shape["config_file"]

    # Build the JSON config dict (matches aiter's expected format)
    triton_config = {
        "BLOCK_SIZE_M": cfg["BLOCK_SIZE_M"],
        "BLOCK_SIZE_N": cfg["BLOCK_SIZE_N"],
        "BLOCK_SIZE_K": cfg["BLOCK_SIZE_K"],
        "GROUP_SIZE_M": cfg["GROUP_SIZE_M"],
        "num_warps": cfg["num_warps"],
        "num_stages": cfg["num_stages"],
        "waves_per_eu": cfg["waves_per_eu"],
        "matrix_instr_nonkdim": 16,
        "NUM_KSPLIT": 1,
        "SPLITK_BLOCK_SIZE": 1024,
    }

    # The config JSON has M-tier keys like "1", "2", "4", etc.
    # We need ALL M tiers to point to our config
    config_json = {}
    for m_tier in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        config_json[str(m_tier)] = triton_config

    config_json_str = json.dumps(config_json, indent=2)

    # Config tag for logging
    tag = f"{shape_key}_BM{cfg['BLOCK_SIZE_M']}_BN{cfg['BLOCK_SIZE_N']}_W{cfg['num_warps']}_S{cfg['num_stages']}_WV{cfg['waves_per_eu']}_G{cfg['GROUP_SIZE_M']}"

    submission = f'''#!/usr/bin/env python3
"""
GEMM Sweep #{index}: {tag}
Target: {shape_key} (M={shape["M"]}, N={shape["N"]}, K={shape["K"]})
Config: BM={cfg["BLOCK_SIZE_M"]}, BN={cfg["BLOCK_SIZE_N"]}, warps={cfg["num_warps"]}, stages={cfg["num_stages"]}, waves={cfg["waves_per_eu"]}, GROUP_M={cfg["GROUP_SIZE_M"]}
"""
import os
import json
import torch

# Inject config for target shape BEFORE importing aiter
CONFIG_DIR = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
CONFIG_FILE = os.path.join(CONFIG_DIR, "{config_filename}")

config_data = {config_json_str}

# Write config
try:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f)
    print(f"SWEEP: Wrote config to {{CONFIG_FILE}}")
    print(f"SWEEP_TAG: {tag}")
except Exception as e:
    print(f"SWEEP: Failed to write config: {{e}}")

os.environ["OPTIMIZE_EPILOGUE"] = "1"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton.quant import dynamic_mxfp4_quant

# Cache output tensors
_out_cache = {{}}

def custom_kernel(A: torch.Tensor, B_shuffle: torch.Tensor, B_scale_shuffle: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    N = B_shuffle.shape[0]
    
    key = (M, N, K)
    if key not in _out_cache:
        _out_cache[key] = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    out = _out_cache[key]

    if K == 1536:
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4, B_shuffle, A_scale, B_scale_shuffle)

    gemm_a16wfp4(A, B_shuffle, B_scale_shuffle, out)
    return out
'''

    return submission, tag


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start index in config list")
    parser.add_argument("--shape", default="both", choices=["k7168", "k2048", "both"])
    parser.add_argument("--list", action="store_true", help="Just list configs, don't generate files")
    parser.add_argument("--count", type=int, default=None, help="Max configs to generate")
    args = parser.parse_args()

    shapes_to_sweep = ["k7168", "k2048"] if args.shape == "both" else [args.shape]

    for shape_key in shapes_to_sweep:
        configs = generate_configs(shape_key)
        configs = prioritize_configs(configs, shape_key)

        print(f"\n{'='*60}")
        print(f"Shape: {shape_key} — {len(configs)} valid configs")
        print(f"{'='*60}")

        if args.list:
            for i, cfg in enumerate(configs[:20]):
                print(f"  [{i}] BM={cfg['BLOCK_SIZE_M']:>3} BN={cfg['BLOCK_SIZE_N']:>3} "
                      f"W={cfg['num_warps']} S={cfg['num_stages']} "
                      f"WV={cfg['waves_per_eu']} G={cfg['GROUP_SIZE_M']}")
            if len(configs) > 20:
                print(f"  ... and {len(configs)-20} more")
            continue

        end = len(configs) if args.count is None else min(args.start + args.count, len(configs))

        for i in range(args.start, end):
            cfg = configs[i]
            submission, tag = make_submission(cfg, shape_key, i)

            fname = f"sweep_{shape_key}_{i:04d}.py"
            fpath = os.path.join(SWEEP_DIR, fname)
            with open(fpath, "w") as f:
                f.write(submission)

            print(f"  [{i}] {tag} → {fname}")

    if not args.list:
        print(f"\nFiles written to: {SWEEP_DIR}")
        print(f"Total files: {len(os.listdir(SWEEP_DIR))}")


if __name__ == "__main__":
    main()

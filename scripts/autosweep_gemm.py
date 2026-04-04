#!/usr/bin/env python3
"""
GEMM Coordinate Descent Config Sweeper.
Uses gemm_a16wfp4 config= parameter for per-shape tuning.
Each submission tests ALL shapes simultaneously.
Perturbs one shape's config per submission, keeps winners.
Never stops. Respects rate limits.
"""
import os, sys, json, time, re, random, shutil, subprocess, math
from pathlib import Path
from datetime import datetime
from copy import deepcopy

REPO = Path(__file__).parent
GEMM_DIR = REPO / "mxfp4-mm"
LOG_DIR = REPO / "auto_research_logs"
LOG_FILE = LOG_DIR / "gemm_sweep.jsonl"
STATE_FILE = LOG_DIR / "gemm_sweep_state.json"
POPCORN = shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli")

# Shapes ordered by optimization potential (K=512 already optimal with defaults)
# Focus on K=7168 and K=2048 first, then K=512 shapes
SHAPES = [
    (16, 2112, 7168),   # Most impactful — dominates geomean
    (64, 7168, 2048),   # Second most impactful
    (4, 2880, 512),     # Already near-optimal with defaults
    (32, 4096, 512),    # Already near-optimal with defaults
    (32, 2880, 512),    # Already near-optimal with defaults
]

# Config parameter names for gemm_a16wfp4
CONFIG_KEYS = [
    "BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K", "GROUP_SIZE_M",
    "num_warps", "num_stages", "waves_per_eu", "matrix_instr_nonkdim",
    "NUM_KSPLIT", "SPLITK_BLOCK_SIZE",
]

# Search space per parameter
PARAM_SPACE = {
    "BLOCK_SIZE_M": [8, 16, 32, 64],
    "BLOCK_SIZE_N": [32, 64, 128, 256],
    "BLOCK_SIZE_K": [128, 256, 512],
    "GROUP_SIZE_M": [1, 2, 4, 8],
    "num_warps": [2, 4, 8],
    "num_stages": [1, 2, 3],
    "waves_per_eu": [1, 2, 4],
    "matrix_instr_nonkdim": [16, 32],
    "NUM_KSPLIT": [1, 2, 4, 8, 16],
    "SPLITK_BLOCK_SIZE": [256, 512, 1024, 2048],
}

# Known best config for K=7168 (proven)
K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# Default starting configs per shape (will be evolved by coordinate descent)
# Proven configs: K7168 with KSPLIT=8 (our best), None for everything else
# PR #2261 configs DON'T WORK on current runner (old aiter #2156)
DEFAULT_CONFIGS = {
    (4, 2880, 512): None,      # Library default (6.15μs — optimal)
    (16, 2112, 7168): K7168_CONFIG.copy(),  # Proven (14.7μs)
    (32, 4096, 512): None,     # Library default (6.68μs — optimal)
    (32, 2880, 512): None,     # Library default (6.92μs — optimal)
    (64, 7168, 2048): None,    # Library default (14.1μs — starting point for sweep)
}

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM sweep: {sweep_desc}"""
from task import input_t, output_t
import torch

_ref = None
_raw = None
_bq = None
_y = {{}}
_warmed = False

# Per-shape configs (coordinate descent optimized)
_CONFIGS = {configs_repr}

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
    cfg = _CONFIGS.get((m, n, k))
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, y=_y[key], config=cfg)
'''


def log(entry):
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(), **entry}) + "\n")


def save_state(state):
    LOG_DIR.mkdir(exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_state():
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except:
            pass
    return {
        "best_configs": {str(k): v for k, v in DEFAULT_CONFIGS.items()},
        "best_times": {},
        "best_geomean": float("inf"),
        "total_submissions": 0,
        "round": 0,
    }


def configs_to_repr(configs):
    """Convert configs dict to Python repr for embedding in submission."""
    lines = []
    for (m, n, k), cfg in sorted(configs.items()):
        if cfg is None:
            lines.append(f"    ({m}, {n}, {k}): None,")
        else:
            # Ensure cache_modifier is included
            cfg_with_cm = dict(cfg)
            if "cache_modifier" not in cfg_with_cm:
                cfg_with_cm["cache_modifier"] = None
            cfg_str = repr(cfg_with_cm)
            lines.append(f"    ({m}, {n}, {k}): {cfg_str},")
    return "{\n" + "\n".join(lines) + "\n}"


def generate_submission(configs, desc, idx):
    """Generate submission with all per-shape configs."""
    code = SUBMISSION_TEMPLATE.format(
        sweep_desc=desc,
        configs_repr=configs_to_repr(configs),
    )
    filepath = GEMM_DIR / f"sweep_{idx:04d}.py"
    filepath.write_text(code)
    return filepath


def submit(filepath, mode="benchmark"):
    """Submit and parse per-shape timing from output."""
    cmd = [
        POPCORN, "submit", "--gpu", "MI355X",
        "--leaderboard", "amd-mxfp4-mm",
        "--mode", mode, str(filepath), "--no-tui",
    ]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO))
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"

        if "Rate limit" in out:
            wait_match = re.search(r'Try again in (\d+)s', out)
            wait = int(wait_match.group(1)) + 15 if wait_match else 400
            print(f"  RATE LIMITED, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue

        # Parse per-shape timing
        times = {}
        lines = out.split('\n')
        for i, line in enumerate(lines):
            km = re.match(r'.*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)', line)
            if km and i + 1 < len(lines):
                tm = re.search(r'(\d+\.?\d*)\s*[±μu]', lines[i + 1])
                if tm:
                    shape = (int(km.group(2)), int(km.group(3)), int(km.group(1)))
                    times[shape] = float(tm.group(1))

        # Also try to parse geomean from output
        gm_match = re.search(r'[Gg]eomean[:\s]+(\d+\.?\d*)', out)
        geomean_reported = float(gm_match.group(1)) if gm_match else None

        if not times and ("Error" in out or "error" in out or "FAILED" in out):
            return None, out[:500]

        return times, out[:500]

    return None, "RATE_LIMITED_3x"


def geomean(times_dict):
    """Compute geometric mean of timing values."""
    vals = [v for v in times_dict.values() if v > 0]
    if not vals:
        return float("inf")
    product = 1.0
    for v in vals:
        product *= v
    return product ** (1.0 / len(vals))


def main():
    state = load_state()
    print(f"=== GEMM Coordinate Descent Sweeper ===")
    print(f"Started: {datetime.now()}")
    print(f"Previous submissions: {state['total_submissions']}")
    print(f"Previous best geomean: {state['best_geomean']}")

    # Convert string keys back to tuple keys
    best_configs = {}
    for k, v in state["best_configs"].items():
        try:
            shape = tuple(json.loads(k.replace("(", "[").replace(")", "]")))
            best_configs[shape] = v
        except:
            pass

    if not best_configs:
        best_configs = deepcopy(DEFAULT_CONFIGS)

    best_times = {tuple(json.loads(k.replace("(", "[").replace(")", "]"))): v
                  for k, v in state.get("best_times", {}).items()}
    best_geomean = state["best_geomean"]
    total_subs = state["total_submissions"]
    rnd = state["round"]

    # Baseline: submit with all current best configs
    print(f"\n--- Baseline submission with current best configs ---")
    filepath = generate_submission(best_configs, "baseline", 0)
    times, raw = submit(filepath)
    if times:
        gm = geomean(times)
        print(f"  Baseline geomean: {gm:.2f}μs")
        for shape, t in sorted(times.items()):
            print(f"    {shape}: {t:.2f}μs")
        if gm < best_geomean:
            best_geomean = gm
            best_times = times
    total_subs += 1

    # Main loop: coordinate descent per shape
    while True:
        rnd += 1
        improved_this_round = False
        print(f"\n{'='*60}")
        print(f"ROUND {rnd} | Best geomean: {best_geomean:.2f}μs | Subs: {total_subs}")
        print(f"{'='*60}")

        # Iterate over shapes
        for shape in SHAPES:
            m, n, k = shape
            current_cfg = best_configs.get(shape)
            print(f"\n  Shape ({m},{n},{k}): {'custom config' if current_cfg else 'library default'}")

            # If current config is None (library default), first test it as baseline
            # Then try explicit configs to see if we can beat the default
            if current_cfg is None:
                # Generate a reasonable starting config to begin sweeping from
                base_cfg = {
                    "BLOCK_SIZE_M": min(m, 32), "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1,
                    "num_warps": 4, "num_stages": 2,
                    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
                    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 512,
                    "cache_modifier": None,
                }
            else:
                base_cfg = current_cfg

            # Try each parameter
            for param in CONFIG_KEYS:
                if param not in PARAM_SPACE:
                    continue

                current_val = base_cfg.get(param)
                options = PARAM_SPACE[param]
                tested_vals = [current_val]

                for val in options:
                    if val == current_val or val in tested_vals:
                        continue
                    tested_vals.append(val)

                    # Create candidate config (always use full dict, not None)
                    candidate = deepcopy(best_configs)
                    candidate[shape] = deepcopy(base_cfg)
                    candidate[shape][param] = val

                    desc = f"R{rnd}_({m},{n},{k})_{param}={val}"
                    total_subs += 1
                    filepath = generate_submission(candidate, desc, total_subs)
                    print(f"    [{total_subs}] {param}={val}...", end=" ", flush=True)

                    times, raw = submit(filepath)
                    if times is None:
                        print(f"FAILED")
                        log({"type": "fail", "shape": str(shape), "param": param,
                             "val": val, "error": raw[:200]})
                        continue

                    gm = geomean(times)
                    shape_time = times.get(shape, float("inf"))
                    prev_shape_time = best_times.get(shape, float("inf"))

                    if gm < best_geomean:
                        print(f"BETTER! {gm:.2f}μs (was {best_geomean:.2f}μs)")
                        best_geomean = gm
                        best_times = times
                        best_configs[shape] = candidate[shape]
                        base_cfg = candidate[shape]
                        improved_this_round = True

                        log({"type": "improvement", "shape": str(shape),
                             "param": param, "val": val, "geomean": gm,
                             "times": {str(s): t for s, t in times.items()},
                             "config": candidate[shape]})
                    else:
                        print(f"{gm:.2f}μs (no improvement)")
                        log({"type": "tested", "shape": str(shape),
                             "param": param, "val": val, "geomean": gm})

                    # Save state after each submission
                    state = {
                        "best_configs": {str(s): c for s, c in best_configs.items()},
                        "best_times": {str(s): t for s, t in best_times.items()},
                        "best_geomean": best_geomean,
                        "total_submissions": total_subs,
                        "round": rnd,
                    }
                    save_state(state)

        # End of round: promote to leaderboard if improved
        if improved_this_round:
            print(f"\n  >> Round {rnd} improved! Submitting to leaderboard...")
            lb_filepath = generate_submission(best_configs, f"best_R{rnd}", total_subs + 1)
            _, raw = submit(lb_filepath, mode="leaderboard")
            print(f"  Leaderboard result: {raw[:200]}")
            total_subs += 1
        else:
            print(f"\n  Round {rnd}: no improvement.")

        # Random exploration phase (try 3 random configs)
        print(f"\n--- Random Exploration ---")
        for _ in range(3):
            # Pick random shape and randomize its config
            shape = random.choice(SHAPES)
            rand_cfg = {
                param: random.choice(vals) for param, vals in PARAM_SPACE.items()
            }
            candidate = deepcopy(best_configs)
            candidate[shape] = rand_cfg

            total_subs += 1
            desc = f"R{rnd}_random_{shape[0]}x{shape[1]}x{shape[2]}"
            filepath = generate_submission(candidate, desc, total_subs)
            print(f"    [{total_subs}] Random {shape}...", end=" ", flush=True)

            times, raw = submit(filepath)
            if times:
                gm = geomean(times)
                if gm < best_geomean:
                    print(f"BETTER! {gm:.2f}μs!")
                    best_geomean = gm
                    best_times = times
                    best_configs[shape] = rand_cfg
                    log({"type": "random_improvement", "shape": str(shape),
                         "geomean": gm, "config": rand_cfg})
                else:
                    print(f"{gm:.2f}μs")
            else:
                print("FAILED")

        # Summary
        print(f"\n--- Round {rnd} Summary ---")
        print(f"  Total submissions: {total_subs}")
        print(f"  Best geomean: {best_geomean:.2f}μs")
        for shape, t in sorted(best_times.items()):
            cfg = best_configs.get(shape, {})
            print(f"    {shape}: {t:.2f}μs  BM={cfg.get('BLOCK_SIZE_M')} BN={cfg.get('BLOCK_SIZE_N')} KS={cfg.get('NUM_KSPLIT')}")

        state = {
            "best_configs": {str(s): c for s, c in best_configs.items()},
            "best_times": {str(s): t for s, t in best_times.items()},
            "best_geomean": best_geomean,
            "total_submissions": total_subs,
            "round": rnd,
        }
        save_state(state)


if __name__ == "__main__":
    main()

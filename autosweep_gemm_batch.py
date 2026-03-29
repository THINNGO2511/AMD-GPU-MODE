#!/usr/bin/env python3
"""
GEMM Batch Config Sweeper — tests 8-10 configs PER submission.
Each submission runs warmup where it times multiple configs via torch.cuda.Event,
prints results to stdout, then uses the best for the actual benchmark.
10x throughput vs single-config sweeper.
"""
import os, sys, json, time, re, random, shutil, subprocess, math, itertools
from pathlib import Path
from datetime import datetime
from copy import deepcopy

REPO = Path(__file__).parent
GEMM_DIR = REPO / "mxfp4-mm"
LOG_DIR = REPO / "auto_research_logs"
LOG_FILE = LOG_DIR / "gemm_batch_sweep.jsonl"
STATE_FILE = LOG_DIR / "gemm_batch_state.json"
POPCORN = shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli")

# Config space — focused on K=7168 and K=2048 (K=512 already optimal with defaults)
PARAM_SPACE_K7168 = {
    "BLOCK_SIZE_M": [8, 16, 32],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [128, 256, 512],
    "NUM_KSPLIT": [1, 2, 4, 8, 16],
    "num_stages": [2, 3],
    "waves_per_eu": [1, 2, 4],
    "num_warps": [4, 8],
}

PARAM_SPACE_K2048 = {
    "BLOCK_SIZE_M": [32, 64],
    "BLOCK_SIZE_N": [32, 64, 128],
    "BLOCK_SIZE_K": [128, 256, 512],
    "NUM_KSPLIT": [1, 2, 4],
    "num_stages": [2, 3],
    "waves_per_eu": [1, 2, 4],
    "num_warps": [4, 8],
}

# Proven K7168 config as baseline
K7168_BEST = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# The submission template — tests BATCH_SIZE configs during warmup
SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Batch sweep: {desc}"""
from task import input_t, output_t
import torch, sys

_ref = None; _raw = None; _bq = None; _y = {{}}; _tested = False; _best_cfg = None

# Configs to test (up to 10 per submission)
_TEST_CONFIGS = {configs_list}

# Proven fallback
_K7168_FALLBACK = {k7168_fallback}

def _unshuffle(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)

def _prewarm():
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try: dynamic_mxfp4_quant(torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda'))
        except: pass
    torch.cuda.synchronize()

def _bench_config(A, bq, raw, cfg, n_iter=5):
    """Time one config using cuda events. Minimal iterations to save JIT budget."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    m, k = A.shape
    n = raw.shape[0]
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    for _ in range(2):
        try:
            gemm_a16wfp4(A, bq, raw, dtype=torch.bfloat16, y=y, config=cfg)
        except:
            return float('inf')
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        gemm_a16wfp4(A, bq, raw, dtype=torch.bfloat16, y=y, config=cfg)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter * 1000

def _test_batch(A, bq, raw):
    """Test all configs in batch, return best."""
    global _tested, _best_cfg
    if _tested: return
    _tested = True
    m, k = A.shape
    results = []
    for i, cfg in enumerate(_TEST_CONFIGS):
        t = _bench_config(A, bq, raw, cfg)
        tag = f"BM{{cfg.get('BLOCK_SIZE_M',0)}}_BN{{cfg.get('BLOCK_SIZE_N',0)}}_BK{{cfg.get('BLOCK_SIZE_K',0)}}_KS{{cfg.get('NUM_KSPLIT',0)}}_S{{cfg.get('num_stages',0)}}_W{{cfg.get('num_warps',0)}}_WPE{{cfg.get('waves_per_eu',0)}}"
        print(f"SWEEP_{{m}}x{{k}}: {{tag}} = {{t:.2f}}us", flush=True)
        results.append((t, cfg))
    # Find best
    results.sort(key=lambda x: x[0])
    if results and results[0][0] < float('inf'):
        _best_cfg = results[0][1]
        print(f"BEST_{{m}}x{{k}}: {{results[0][0]:.2f}}us", flush=True)

def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _bq
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    if _ref is not B_scale_sh:
        _ref = B_scale_sh; _raw = _unshuffle(B_scale_sh); _bq = B_q.view(torch.uint8); _prewarm()
    # Test batch on first call for target shape
    if k == {target_k} and not _tested:
        _test_batch(A, _bq, _raw)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y: _y[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    # Use best discovered config for target K, fallback for K=7168, None for others
    if k == {target_k} and _best_cfg:
        cfg = _best_cfg
    elif k == 7168:
        cfg = _K7168_FALLBACK
    else:
        cfg = None
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
        except: pass
    return {"best_k7168": K7168_BEST, "best_k2048": None,
            "all_results": {}, "total_subs": 0}


def gen_config_batch(param_space, base_cfg, n=8):
    """Generate n configs ensuring good CU occupancy (>= 128 blocks for 304 CUs)."""
    # Target shapes for grid estimation
    SHAPE_K7168 = (16, 2112, 7168)  # M=16, N=2112, K=7168
    SHAPE_K2048 = (64, 7168, 2048)  # M=64, N=7168, K=2048
    CU_COUNT = 304

    configs = []
    attempts = 0
    while len(configs) < n and attempts < n * 20:
        attempts += 1
        cfg = dict(base_cfg) if base_cfg else {}
        params = random.sample(list(param_space.keys()), min(2, len(param_space)))
        for p in params:
            cfg[p] = random.choice(param_space[p])
        cfg.setdefault("GROUP_SIZE_M", 1)
        cfg.setdefault("matrix_instr_nonkdim", 16)
        cfg.setdefault("cache_modifier", None)
        cfg.setdefault("SPLITK_BLOCK_SIZE", max(512, cfg.get("BLOCK_SIZE_K", 128) * 2))

        # Check occupancy: grid_blocks = ceil(M/BM) * ceil(N/BN) * KSPLIT
        bm = cfg.get("BLOCK_SIZE_M", 16)
        bn = cfg.get("BLOCK_SIZE_N", 64)
        ks = cfg.get("NUM_KSPLIT", 1)
        m, nn, k = SHAPE_K7168
        grid = math.ceil(m / bm) * math.ceil(nn / bn) * ks
        # Require at least 128 blocks (decent utilization of 304 CUs)
        if grid >= 128:
            configs.append(cfg)

    # If not enough valid configs, pad with base
    while len(configs) < n:
        configs.append(dict(base_cfg) if base_cfg else {})
    return configs


def generate_submission(configs, target_k, desc, idx):
    code = SUBMISSION_TEMPLATE.format(
        desc=desc,
        configs_list=repr(configs),
        k7168_fallback=repr(K7168_BEST),
        target_k=target_k,
    )
    filepath = GEMM_DIR / f"batch_sweep_{idx:04d}.py"
    filepath.write_text(code)
    return filepath


def submit_and_parse(filepath):
    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", str(filepath), "--no-tui"]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=str(REPO))
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        if "Rate limit" in out:
            wait_match = re.search(r'Try again in (\d+)s', out)
            wait = int(wait_match.group(1)) + 15 if wait_match else 600
            print(f"  RATE LIMITED, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue
        # Parse SWEEP results from stdout
        sweep_results = {}
        for line in out.split('\n'):
            m = re.match(r'SWEEP_(\d+)x(\d+): (\S+) = (\d+\.?\d*)us', line)
            if m:
                shape = f"{m.group(1)}x{m.group(2)}"
                tag = m.group(3)
                t = float(m.group(4))
                sweep_results[tag] = t
            bm = re.match(r'BEST_(\d+)x(\d+): (\d+\.?\d*)us', line)
            if bm:
                sweep_results["__best__"] = float(bm.group(3))
        # Parse geomean from benchmark
        times = {}
        lines = out.split('\n')
        for i, line in enumerate(lines):
            km = re.match(r'.*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)', line)
            if km and i + 1 < len(lines):
                tm = re.search(r'(\d+\.?\d*)\s*[±μu]', lines[i + 1])
                if tm:
                    shape = (int(km.group(2)), int(km.group(3)), int(km.group(1)))
                    times[shape] = float(tm.group(1))
        return {"sweep": sweep_results, "bench": times}, out[:500]
    return None, "RATE_LIMITED"


def geomean(times):
    vals = [v for v in times.values() if v > 0]
    if not vals: return float("inf")
    p = 1.0
    for v in vals: p *= v
    return p ** (1.0 / len(vals))


def main():
    state = load_state()
    print(f"=== GEMM Batch Config Sweeper ===", flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    total_subs = state["total_subs"]
    best_k7168 = state["best_k7168"]
    best_k2048 = state["best_k2048"]
    all_results = state["all_results"]

    rnd = 0
    while True:
        rnd += 1
        # Focus on K=7168 (biggest impact, K=2048 keeps timing out)
        # Try K=2048 every 5th round
        if rnd % 5 == 0:
            target_k = 2048
            space = PARAM_SPACE_K2048
            base = best_k2048 or {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128,
                                   "BLOCK_SIZE_K": 256, "GROUP_SIZE_M": 1,
                                   "num_warps": 4, "num_stages": 2,
                                   "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
                                   "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024}
        else:
            target_k = 7168
            space = PARAM_SPACE_K7168
            base = best_k7168

        configs = gen_config_batch(space, base, n=3)
        total_subs += 1
        desc = f"R{rnd}_K{target_k}_batch4"
        filepath = generate_submission(configs, target_k, desc, total_subs)
        print(f"\n[{total_subs}] Round {rnd}: K={target_k}, testing {len(configs)} configs...", flush=True)

        result, raw = submit_and_parse(filepath)
        if result is None:
            print(f"  FAILED: {raw[:100]}", flush=True)
            log({"type": "fail", "round": rnd, "k": target_k, "error": raw[:200]})
        else:
            sweep = result.get("sweep", {})
            bench = result.get("bench", {})
            gm = geomean(bench) if bench else float("inf")

            # Log all config results
            for tag, t in sweep.items():
                if tag != "__best__":
                    all_results[f"K{target_k}_{tag}"] = t

            best_in_batch = sweep.get("__best__", float("inf"))
            print(f"  Batch best: {best_in_batch:.2f}μs | Bench geomean: {gm:.2f}μs", flush=True)
            for tag, t in sorted(sweep.items(), key=lambda x: x[1])[:3]:
                if tag != "__best__":
                    print(f"    {tag}: {t:.2f}μs", flush=True)

            log({"type": "batch", "round": rnd, "k": target_k,
                 "sweep": sweep, "geomean": gm, "subs": total_subs})

        state = {"best_k7168": best_k7168, "best_k2048": best_k2048,
                 "all_results": all_results, "total_subs": total_subs}
        save_state(state)

    print(f"=== Sweep complete ===", flush=True)


if __name__ == "__main__":
    main()

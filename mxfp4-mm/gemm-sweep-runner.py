#!/usr/bin/env python3
# GEMM Config Sweep Runner for AMD GPU MODE Hackathon (MI355X / gfx950)
# Automated coordinate descent + grid search for gemm_a16wfp4 configs.
# Usage: see --help

import argparse, copy, json, os, re, subprocess, sys, time
from datetime import datetime
from itertools import product

POPCORN = os.path.expanduser("~/.local/bin/popcorn-cli")
DEFAULT_PROJECT_DIR = os.path.expanduser("~/Downloads/code/AMD-GPU-MODE/mxfp4-mm")
RESULTS_FILE = "sweep_runner_results.json"

BENCHMARK_SHAPES = [
    (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
    (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
]

BEST_CONFIGS = {
    (4, 2880, 512): None,
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
    },
    (32, 4096, 512): None,
    (32, 2880, 512): None,
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
        "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
    },
    (256, 3072, 1536): "afp4wfp4",
}

PARAM_RANGES = {
    "BLOCK_SIZE_M": [8, 16, 32, 64],
    "BLOCK_SIZE_N": [16, 32, 64, 128, 256],
    "BLOCK_SIZE_K": [128, 256, 512],
    "GROUP_SIZE_M": [1, 2, 4, 8],
    "NUM_KSPLIT": [1, 2, 4, 7, 8, 14],
    "num_warps": [1, 2, 4, 8],
    "num_stages": [1, 2, 3],
    "waves_per_eu": [1, 2, 4],
    "matrix_instr_nonkdim": [16, 32],
    "cache_modifier": [None, ".cg"],
    "SPLITK_BLOCK_SIZE": [512, 1024, 2048, 4096],
}

DEFAULT_A16WFP4_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
}

COORDESC_ORDER = [
    "BLOCK_SIZE_N", "NUM_KSPLIT", "BLOCK_SIZE_M", "num_warps",
    "BLOCK_SIZE_K", "num_stages", "waves_per_eu", "cache_modifier",
    "matrix_instr_nonkdim", "GROUP_SIZE_M", "SPLITK_BLOCK_SIZE",
]


def cfg2str(cfg):
    if cfg is None: return "None"
    parts = []
    for k, v in cfg.items():
        if v is None: parts.append('"%s": None' % k)
        elif isinstance(v, str): parts.append('"%s": "%s"' % (k, v))
        else: parts.append('"%s": %s' % (k, v))
    return "{" + ", ".join(parts) + "}"


def cfg_short(cfg):
    if cfg is None: return "DEFAULT"
    if cfg == "afp4wfp4": return "AFP4WFP4"
    S = {"BLOCK_SIZE_M":"BM","BLOCK_SIZE_N":"BN","BLOCK_SIZE_K":"BK",
         "GROUP_SIZE_M":"GSM","NUM_KSPLIT":"KS","num_warps":"W",
         "num_stages":"S","waves_per_eu":"WPE","matrix_instr_nonkdim":"MI",
         "SPLITK_BLOCK_SIZE":"SKB","cache_modifier":"CM"}
    keys = ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M",
            "NUM_KSPLIT","num_warps","num_stages","waves_per_eu",
            "matrix_instr_nonkdim","SPLITK_BLOCK_SIZE","cache_modifier"]
    return " ".join("%s=%s" % (S.get(k,k), cfg[k]) for k in keys if k in cfg)


def cfg_key(shape, cfg):
    m, n, k = shape
    if cfg is None: return "M%d_N%d_K%d_DEFAULT" % (m, n, k)
    if cfg == "afp4wfp4": return "M%d_N%d_K%d_AFP4WFP4" % (m, n, k)
    S = {"BLOCK_SIZE_M":"BM","BLOCK_SIZE_N":"BN","BLOCK_SIZE_K":"BK",
         "GROUP_SIZE_M":"GSM","NUM_KSPLIT":"KS","num_warps":"W",
         "num_stages":"S","waves_per_eu":"WPE","matrix_instr_nonkdim":"MI",
         "SPLITK_BLOCK_SIZE":"SKB"}
    parts = ["M%d_N%d_K%d" % (m, n, k)]
    for p in S:
        if p in cfg: parts.append("%s%s" % (S[p], cfg[p]))
    cm = cfg.get("cache_modifier")
    if cm: parts.append("CM%s" % cm.replace(".", ""))
    return "_".join(parts)


def geomean(td):
    vals = list(td.values())
    if not vals: return float('inf')
    p = 1.0
    for v in vals: p *= v
    return p ** (1.0 / len(vals))


def load_results(pd):
    p = os.path.join(pd, RESULTS_FILE)
    if os.path.exists(p):
        try:
            with open(p) as f: return json.load(f)
        except: return {}
    return {}


def save_results(r, pd):
    with open(os.path.join(pd, RESULTS_FILE), "w") as f:
        json.dump(r, f, indent=2, sort_keys=True)


def record(results, shape, cfg, times, pd):
    key = cfg_key(shape, cfg)
    entry = {"shape": list(shape), "config": cfg if cfg != "afp4wfp4" else "afp4wfp4",
             "timestamp": datetime.now().isoformat()}
    if times == "ERROR":
        entry["error"] = True
    elif times:
        entry["times"] = {str(k): v for k, v in times.items()}
        entry["geomean"] = geomean(times)
        if shape in times: entry["target_time"] = times[shape]
    results[key] = entry
    save_results(results, pd)
    return entry


def build_all_configs(target, exp_cfg, bc=None):
    cfgs = {}
    bc = bc or BEST_CONFIGS
    for s in BENCHMARK_SHAPES:
        cfgs[s] = exp_cfg if s == target else bc.get(s)
    return cfgs


def gen_submission(configs, project_dir):
    blocks, dispatch = [], []
    for (m, n, k), cfg in configs.items():
        if cfg == "afp4wfp4" or cfg is None: continue
        vn = "_CFG_%d_%d_%d" % (m, n, k)
        blocks.append("%s = %s" % (vn, cfg2str(cfg)))
        dispatch.append((m, n, k, vn))

    afp4 = [(m,n,k) for (m,n,k),c in configs.items() if c == "afp4wfp4"]
    afp4_cond = " or ".join("(k == %d and n == %d)" % (k,n) for m,n,k in afp4) or "False"

    dlines = []
    for m, n, k, vn in dispatch:
        kw = "if" if not dlines else "elif"
        dlines.append("    %s k == %d and n == %d:" % (kw, k, n))
        dlines.append("        cfg = %s" % vn)
    if dlines:
        dlines.append("    else:")
        dlines.append("        cfg = None")
    else:
        dlines.append("    cfg = None")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = []
    L.append('#!POPCORN leaderboard amd-mxfp4-mm')
    L.append('#!POPCORN gpu MI355X')
    L.append('# Auto-generated sweep submission -- %s' % ts)
    L.append('from task import input_t, output_t')
    L.append('import torch')
    L.append('')
    L.append('_bscale_ref = None')
    L.append('_bscale_raw = None')
    L.append('_bq_u8 = None')
    L.append('_y_cache = {}')
    L.append('')
    for b in blocks: L.append(b)
    L.append('')
    L.append('')
    L.append('def _unshuffle_e8m0(scale_sh):')
    L.append('    s = scale_sh.view(torch.uint8)')
    L.append('    sm, sn = s.shape')
    L.append('    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)')
    L.append('    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()')
    L.append('    return s.view(sm, sn)')
    L.append('')
    L.append('')
    L.append('def custom_kernel(data: input_t) -> output_t:')
    L.append('    global _bscale_ref, _bscale_raw, _bq_u8')
    L.append('')
    L.append('    A, B, B_q, B_shuffle, B_scale_sh = data')
    L.append('    m, k = A.shape')
    L.append('    n = B.shape[0]')
    L.append('')
    L.append('    if _bscale_ref is not B_scale_sh:')
    L.append('        _bscale_ref = B_scale_sh')
    L.append('        _bscale_raw = _unshuffle_e8m0(B_scale_sh)')
    L.append('        _bq_u8 = B_q.view(torch.uint8)')
    L.append('')
    L.append('    if %s:' % afp4_cond)
    L.append('        from aiter.ops.triton.quant import dynamic_mxfp4_quant')
    L.append('        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4')
    L.append('        A_fp4, A_scale = dynamic_mxfp4_quant(A)')
    L.append('        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,')
    L.append('                             dtype=torch.bfloat16)')
    L.append('')
    L.append('    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4')
    L.append('    key = (m, n)')
    L.append('    if key not in _y_cache:')
    L.append("        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')")
    L.append('')
    for dl in dlines: L.append(dl)
    L.append('')
    L.append('    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,')
    L.append('                       y=_y_cache[key], config=cfg)')
    L.append('')

    path = os.path.join(project_dir, "_sweep_submission.py")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def submit_and_parse(sub_path, project_dir, dry_run=False, max_retries=5):
    repo_root = os.path.dirname(project_dir)
    rel_path = os.path.relpath(sub_path, repo_root)

    if dry_run:
        print("  [DRY RUN] Would submit: %s" % rel_path)
        with open(sub_path) as f:
            for i, line in enumerate(f):
                print("  %s" % line.rstrip())
                if i > 60:
                    print("  ... (truncated)")
                    break
        return None

    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", rel_path, "--no-tui"]

    for attempt in range(max_retries):
        try:
            print("  Submitting (attempt %d)..." % (attempt + 1), end=" ", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=700, cwd=repo_root)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            return None

        with open(os.path.join(project_dir, "_last_sweep_output.txt"), "w") as df:
            df.write(out)

        if "Rate limit" in out or "rate limit" in out.lower():
            wm = re.search(r"[Tt]ry again in (\d+)", out)
            wait = int(wm.group(1)) + 15 if wm else 400
            print("RATE LIMITED, waiting %ds..." % wait, end=" ", flush=True)
            time.sleep(wait)
            continue

        errs = ["TypeError","RuntimeError","SyntaxError","AttributeError",
                "ValueError","NameError","KeyError","IndexError",
                "Benchmarking failed","AssertionError","ModuleNotFoundError"]
        if any(e in out for e in errs):
            el = [l for l in out.split("\n") if any(e in l for e in ["Error","error","Traceback","FAIL"])]
            print("ERROR: %s" % ("; ".join(el[:3]) if el else "unknown")[:150])
            return "ERROR"

        times = {}
        outlines = out.split("\n")
        for i, line in enumerate(outlines):
            km = re.match(r".*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)", line)
            if km and i + 1 < len(outlines):
                tm = re.search(r"([\d.]+)\s*[^\d\s]", outlines[i+1])
                if tm:
                    t_us = float(tm.group(1))
                    if t_us > 0.1:
                        times[(int(km.group(2)), int(km.group(3)), int(km.group(1)))] = t_us

        if times:
            gm = geomean(times)
            print("done gm=%.2fus (%d shapes)" % (gm, len(times)))
            return times
        else:
            if "success" in out.lower():
                print("SUCCESS but no timings parsed")
            else:
                print("NO TIMINGS (len=%d)" % len(out))
            return None

    print("MAX RETRIES")
    return None



def submit_and_parse(sub_path, project_dir, dry_run=False, max_retries=5):
    repo_root = os.path.dirname(project_dir)
    rel_path = os.path.relpath(sub_path, repo_root)

    if dry_run:
        print("  [DRY RUN] Would submit: %s" % rel_path)
        with open(sub_path) as f:
            for i, line in enumerate(f):
                print("  %s" % line.rstrip())
                if i > 60:
                    print("  ... (truncated)")
                    break
        return None

    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", rel_path, "--no-tui"]

    for attempt in range(max_retries):
        try:
            print("  Submitting (attempt %d)..." % (attempt + 1), end=" ", flush=True)
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=700, cwd=repo_root)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            return None

        with open(os.path.join(project_dir, "_last_sweep_output.txt"), "w") as df:
            df.write(out)

        if "Rate limit" in out or "rate limit" in out.lower():
            wm = re.search(r"[Tt]ry again in (\d+)", out)
            wait = int(wm.group(1)) + 15 if wm else 400
            print("RATE LIMITED, waiting %ds..." % wait, end=" ", flush=True)
            time.sleep(wait)
            continue

        errs = ["TypeError","RuntimeError","SyntaxError","AttributeError",
                "ValueError","NameError","KeyError","IndexError",
                "Benchmarking failed","AssertionError","ModuleNotFoundError"]
        if any(e in out for e in errs):
            el = [l for l in out.split("\n") if any(e in l for e in ["Error","error","Traceback","FAIL"])]
            print("ERROR: %s" % ("; ".join(el[:3]) if el else "unknown")[:150])
            return "ERROR"

        times = {}
        outlines = out.split("\n")
        for i, line in enumerate(outlines):
            km = re.match(r".*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)", line)
            if km and i + 1 < len(outlines):
                tm = re.search(r"([\d.]+)\s*[^\d\s]", outlines[i+1])
                if tm:
                    t_us = float(tm.group(1))
                    if t_us > 0.1:
                        times[(int(km.group(2)), int(km.group(3)), int(km.group(1)))] = t_us

        if times:
            gm = geomean(times)
            print("done gm=%.2fus (%d shapes)" % (gm, len(times)))
            return times
        else:
            if "success" in out.lower():
                print("SUCCESS but no timings parsed")
            else:
                print("NO TIMINGS (len=%d)" % len(out))
            return None

    print("MAX RETRIES")
    return None


def sweep_param(target, param, values, pd, dry_run=False):
    results = load_results(pd)
    m, n, k = target
    base = BEST_CONFIGS.get(target)
    if base is None or base == "afp4wfp4":
        base = copy.deepcopy(DEFAULT_A16WFP4_CONFIG)
    else:
        base = copy.deepcopy(base)

    print("\n" + "=" * 70)
    print("SWEEP: M=%d,N=%d,K=%d -- %s = %s" % (m, n, k, param, values))
    print("Base: %s" % cfg_short(base))
    print("=" * 70)

    best_gm, best_cfg, best_tt = float('inf'), None, float('inf')

    for i, val in enumerate(values):
        cfg = copy.deepcopy(base)
        if param == "cache_modifier":
            cfg["cache_modifier"] = None if val in (None, "None", "null") else str(val)
        else:
            cfg[param] = val

        key = cfg_key(target, cfg)
        if key in results and "error" not in results[key] and "times" in results[key]:
            gm = results[key]["geomean"]
            tt = results[key].get("target_time", "?")
            marker = " **BEST" if gm < best_gm else ""
            print("[%d/%d] %s=%s CACHED: target=%sus gm=%.2fus%s" %
                  (i+1, len(values), param, val, tt, gm, marker))
            if gm < best_gm:
                best_gm, best_cfg, best_tt = gm, cfg, results[key].get('target_time', float('inf'))
            continue

        print("[%d/%d] %s=%s" % (i+1, len(values), param, val))
        sp = gen_submission(build_all_configs(target, cfg), pd)
        times = submit_and_parse(sp, pd, dry_run=dry_run)
        record(results, target, cfg, times, pd)

        if times and times != "ERROR":
            gm = geomean(times)
            tt = times.get(target, float('inf'))
            marker = " **BEST" if gm < best_gm else ""
            print("  -> target=%.2fus gm=%.2fus%s" % (tt, gm, marker))
            if gm < best_gm:
                best_gm, best_cfg, best_tt = gm, cfg, tt

    print("\n" + "-" * 70)
    print("RESULT: Best %s=%s  target=%sus gm=%.2fus" %
          (param, best_cfg.get(param) if best_cfg else "?", best_tt, best_gm))
    if best_cfg: print("  Config: %s" % cfg_short(best_cfg))
    print("-" * 70)
    return best_cfg, best_tt, best_gm


def coordinate_descent(target, pd, max_rounds=3, dry_run=False):
    results = load_results(pd)
    m, n, k = target
    base = BEST_CONFIGS.get(target)
    if base is None or base == "afp4wfp4":
        base = copy.deepcopy(DEFAULT_A16WFP4_CONFIG)
    else:
        base = copy.deepcopy(base)

    cur = copy.deepcopy(base)
    print("\n" + "=" * 70)
    print("COORDESC: M=%d,N=%d,K=%d  rounds=%d" % (m, n, k, max_rounds))
    print("Start: %s" % cfg_short(cur))
    print("=" * 70)

    key = cfg_key(target, cur)
    if key in results and "error" not in results.get(key,{}) and "times" in results.get(key,{}):
        cur_gm = results[key]["geomean"]
        print("Starting (cached): gm=%.2fus" % cur_gm)
    else:
        sp = gen_submission(build_all_configs(target, cur), pd)
        times = submit_and_parse(sp, pd, dry_run=dry_run)
        if times and times != "ERROR":
            cur_gm = geomean(times)
            record(results, target, cur, times, pd)
            print("Starting: gm=%.2fus" % cur_gm)
        else:
            cur_gm = float('inf')
            print("Starting config failed")

    for rnd in range(1, max_rounds + 1):
        print("\n" + "=" * 70)
        print("ROUND %d/%d  current gm=%.2fus" % (rnd, max_rounds, cur_gm))
        print("Config: %s" % cfg_short(cur))
        print("=" * 70)
        improved = False

        for param in COORDESC_ORDER:
            if param not in PARAM_RANGES: continue
            vals = PARAM_RANGES[param]
            cur_val = cur.get(param)
            test = [v for v in vals if v != cur_val]
            if not test: continue

            print("\n  --- %s (cur=%s) test=%s ---" % (param, cur_val, test))
            best_val, best_gm_p = cur_val, cur_gm

            for val in test:
                cand = copy.deepcopy(cur)
                if param == "cache_modifier":
                    cand["cache_modifier"] = val
                else:
                    cand[param] = val

                key = cfg_key(target, cand)
                if key in results and "error" not in results[key] and "times" in results[key]:
                    gm = results[key]["geomean"]
                    tt = results[key].get("target_time", "?")
                    marker = " **BEST" if gm < best_gm_p else ""
                    print("    %s=%s CACHED gm=%.2fus%s" % (param, val, gm, marker))
                else:
                    sp = gen_submission(build_all_configs(target, cand), pd)
                    times = submit_and_parse(sp, pd, dry_run=dry_run)
                    record(results, target, cand, times, pd)
                    if times and times != "ERROR":
                        gm = results[cfg_key(target, cand)].get("geomean", float('inf'))
                        marker = " **BEST" if gm < best_gm_p else ""
                        print("    %s=%s gm=%.2fus%s" % (param, val, gm, marker))
                    else:
                        print("    %s=%s FAILED" % (param, val))
                        continue

                if gm < best_gm_p:
                    best_gm_p, best_val = gm, val

            if best_val != cur_val:
                if param == "cache_modifier":
                    cur["cache_modifier"] = best_val
                else:
                    cur[param] = best_val
                cur_gm = best_gm_p
                improved = True
                print("  >> %s: %s -> %s (gm=%.2fus)" % (param, cur_val, best_val, cur_gm))

        if not improved:
            print("\nConverged in round %d!" % rnd)
            break

    print("\n" + "=" * 70)
    print("COORDESC DONE: gm=%.2fus" % cur_gm)
    print("Config: %s" % cfg_short(cur))
    print("Dict: %s" % cfg2str(cur))
    print("=" * 70)
    return cur, cur_gm


def grid_search(target, pd, dry_run=False):
    m, n, k = target
    gp = {"BLOCK_SIZE_M": [8,16,32], "BLOCK_SIZE_N": [32,64,128],
          "BLOCK_SIZE_K": [256,512], "NUM_KSPLIT": [1,4,8],
          "num_warps": [2,4,8], "num_stages": [1,2]}
    combos = list(product(*gp.values()))
    names = list(gp.keys())

    print("\nGRID: M=%d,N=%d,K=%d -- %d combos (est %.1fh)" %
          (m, n, k, len(combos), len(combos)/10.0))

    results = load_results(pd)
    best_gm, best_cfg = float('inf'), None

    for i, combo in enumerate(combos):
        cfg = copy.deepcopy(DEFAULT_A16WFP4_CONFIG)
        for name, val in zip(names, combo): cfg[name] = val
        key = cfg_key(target, cfg)

        if key in results and "error" not in results[key] and "times" in results[key]:
            gm = results[key]["geomean"]
            if gm < best_gm: best_gm, best_cfg = gm, cfg
            continue

        print("[%d/%d] %s" % (i+1, len(combos), cfg_short(cfg)))
        sp = gen_submission(build_all_configs(target, cfg), pd)
        times = submit_and_parse(sp, pd, dry_run=dry_run)
        record(results, target, cfg, times, pd)
        if times and times != "ERROR":
            gm = geomean(times)
            if gm < best_gm:
                best_gm, best_cfg = gm, cfg
                print("  **BEST: %.2fus" % gm)

    if best_cfg:
        print("\nGRID BEST: gm=%.2fus  %s" % (best_gm, cfg_short(best_cfg)))
    return best_cfg, best_gm


def show_status(pd):
    results = load_results(pd)
    print("\n" + "=" * 70)
    print("SWEEP STATUS -- %d configs tested" % len(results))
    print("=" * 70)

    by_shape = {}
    for key, e in results.items():
        if "error" in e or "times" not in e: continue
        s = tuple(e.get("shape", [0,0,0]))
        by_shape.setdefault(s, []).append((key, e))

    for shape in BENCHMARK_SHAPES:
        m, n, k = shape
        print("\n--- M=%d, N=%d, K=%d ---" % (m, n, k))
        print("  Hardcoded best: %s" % cfg_short(BEST_CONFIGS.get(shape)))
        entries = by_shape.get(shape, [])
        if entries:
            entries.sort(key=lambda x: x[1].get("geomean", float('inf')))
            print("  Tested: %d | Top 5:" % len(entries))
            for key, e in entries[:5]:
                print("    %.2fus (target=%s) %s" %
                      (e.get("geomean",0), e.get("target_time","?"), key))
        else:
            print("  No results yet")

    all_e = [(k,v) for k,v in results.items() if "geomean" in v and "error" not in v]
    if all_e:
        all_e.sort(key=lambda x: x[1]["geomean"])
        print("\n--- TOP 10 OVERALL ---")
        for key, e in all_e[:10]:
            print("  %.2fus %s" % (e["geomean"], key))

    errs = sum(1 for v in results.values() if "error" in v)
    if errs: print("\nErrors: %d" % errs)


def parse_shape(s):
    parts = {}
    for p in s.split(","):
        k, v = p.strip().split("=")
        parts[k.strip().upper()] = int(v.strip())
    return (parts["M"], parts["N"], parts["K"])

def parse_values(s, param):
    raw = [v.strip() for v in s.split(",")]
    if param == "cache_modifier":
        return [None if v in ("None","null","none") else v for v in raw]
    out = []
    for v in raw:
        try: out.append(int(v))
        except:
            try: out.append(float(v))
            except: out.append(v)
    return out


def main():
    p = argparse.ArgumentParser(description="GEMM sweep runner for AMD GPU MODE hackathon")
    p.add_argument("--shape", help="Target shape: M=4,N=2880,K=512")
    p.add_argument("--param", help="Parameter to sweep")
    p.add_argument("--values", help="Comma-separated values")
    p.add_argument("--coordesc", action="store_true", help="Coordinate descent")
    p.add_argument("--grid", action="store_true", help="Grid search")
    p.add_argument("--status", action="store_true", help="Show status")
    p.add_argument("--rounds", type=int, default=3, help="Coordesc rounds (default 3)")
    p.add_argument("--dry-run", action="store_true", help="Generate but don't submit")
    p.add_argument("--dir", default=DEFAULT_PROJECT_DIR, help="Project dir")
    p.add_argument("--base", help="Override base config JSON for target shape")

    args = p.parse_args()
    pd = os.path.abspath(args.dir)

    if not os.path.isdir(pd):
        print("ERROR: Dir not found: %s" % pd); sys.exit(1)

    if args.status:
        show_status(pd); return

    if not args.shape:
        p.print_help(); sys.exit(1)

    target = parse_shape(args.shape)
    print("Target: M=%d, N=%d, K=%d" % target)

    if args.base:
        try:
            BEST_CONFIGS[target] = json.loads(args.base)
            print("Base override: %s" % cfg_short(BEST_CONFIGS[target]))
        except json.JSONDecodeError as e:
            print("ERROR: Bad JSON: %s" % e); sys.exit(1)

    if args.coordesc:
        coordinate_descent(target, pd, max_rounds=args.rounds, dry_run=args.dry_run)
    elif args.grid:
        grid_search(target, pd, dry_run=args.dry_run)
    elif args.param and args.values:
        sweep_param(target, args.param, parse_values(args.values, args.param), pd, dry_run=args.dry_run)
    else:
        print("ERROR: Use --coordesc, --grid, or --param+--values")
        p.print_help(); sys.exit(1)

if __name__ == "__main__":
    main()

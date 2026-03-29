#!/usr/bin/env python3
"""
MoE Coordinate Descent Sweeper.
Sweeps CK kernel combos, block_m, splitK for d=2048 bottleneck.
Also tests E=257 injection (currently skipped in best submission).
Never stops. Respects rate limits.
"""
import os, sys, json, time, re, random, shutil, subprocess, math
from pathlib import Path
from datetime import datetime
from copy import deepcopy

REPO = Path(__file__).parent
MOE_DIR = REPO / "moe-mxfp4"
LOG_DIR = REPO / "auto_research_logs"
LOG_FILE = LOG_DIR / "moe_sweep.jsonl"
STATE_FILE = LOG_DIR / "moe_sweep_state.json"
POPCORN = shutil.which("popcorn-cli") or os.path.expanduser("~/.local/bin/popcorn-cli")

# Available CK kernel names for FP4+Silu
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

# Sweep space
# Skip bm=16 (too small, causes slow paths) and bm=256 (too large for most shapes)
BLOCK_M_OPTIONS = [32, 64, 128]
SPLITK_OPTIONS = [0, 1, 2]

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE sweep: {config_name}"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Proven kernels for E<=64 d<2048
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# d=2048 sweep config
D2048_S1 = "{d2048_s1}"
D2048_S2 = "{d2048_s2}"
D2048_BM = {d2048_bm}
D2048_SK = {d2048_sk}

# E=257 config (currently untested!)
E257_INJECT = {e257_inject}
E257_S1 = S1_64
E257_S2 = S2_V1
E257_BM = {e257_bm}

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
                return D2048_BM
            elif est_m < 50: return 32
            else: return 64
        elif E257_INJECT and expert > 64:
            return E257_BM
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh)
        if qt == QuantType.per_1x32 and not r.run_1stage:
            try:
                est = token * topk // expert
                if expert <= 64 and inter_dim >= 2048:
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1, kernelName=D2048_S1, activation=act, quant_type=qt, dtype=dtype, splitk=D2048_SK, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=D2048_S2, activation=act, quant_type=qt, use_non_temporal_load=False),
                        D2048_BM, 0, False)
                elif expert <= 64 and inter_dim < 2048:
                    kn = S1_256 if est >= 100 else S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1, kernelName=kn, activation=act, quant_type=qt, dtype=dtype, splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=S2_V1, activation=act, quant_type=qt, use_non_temporal_load=False),
                        32 if est < 50 else 64, 0, False)
                elif E257_INJECT and expert > 64:
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1, kernelName=E257_S1, activation=act, quant_type=qt, dtype=dtype, splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=E257_S2, activation=act, quant_type=qt, use_non_temporal_load=False),
                        E257_BM, 0, False)
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
        "best_d2048": {"s1": "s1_64", "s2": "s2_v1", "bm": 64, "sk": 0},
        "best_e257": {"inject": False, "bm": 16},
        "best_geomean": float("inf"),
        "total_submissions": 0,
        "tested": [],
    }


def generate_submission(d2048_cfg, e257_cfg, name, idx):
    s1 = S1_KERNELS.get(d2048_cfg["s1"], d2048_cfg["s1"])
    s2 = S2_KERNELS.get(d2048_cfg["s2"], d2048_cfg["s2"])
    code = SUBMISSION_TEMPLATE.format(
        config_name=name,
        d2048_s1=s1, d2048_s2=s2,
        d2048_bm=d2048_cfg["bm"], d2048_sk=d2048_cfg["sk"],
        e257_inject=str(e257_cfg.get("inject", False)),
        e257_bm=e257_cfg.get("bm", 16),
    )
    filepath = MOE_DIR / f"sweep_{idx:04d}.py"
    filepath.write_text(code)
    return filepath


def submit(filepath, mode="benchmark"):
    cmd = [
        POPCORN, "submit", "--gpu", "MI355X",
        "--leaderboard", "amd-moe-mxfp4",
        "--mode", mode, str(filepath), "--no-tui",
    ]
    for attempt in range(3):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=str(REPO))
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT"
        if "Rate limit" in out:
            wait_match = re.search(r'Try again in (\d+)s', out)
            wait = int(wait_match.group(1)) + 15 if wait_match else 400
            print(f"  RATE LIMITED, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue
        # Parse timing from MoE benchmark output
        # Format: "bs: 16; seed: 9371; dexpert: 256; dhidden: 7168; nroutedexperts: 256; ..."
        #         " ⏱ 138 ± 0.1 µs"
        times = {}
        lines = out.split('\n')
        for i, line in enumerate(lines):
            bm = re.match(r'.*bs:\s*(\d+).*dexpert:\s*(\d+).*nroutedexperts:\s*(\d+)', line)
            if bm and i + 1 < len(lines):
                tm = re.search(r'(\d+\.?\d*)\s*±', lines[i + 1])
                if tm:
                    key = f"bs{bm.group(1)}_d{bm.group(2)}_E{bm.group(3)}"
                    times[key] = float(tm.group(1))
        gm_match = re.search(r'[Gg]eomean[:\s]+(\d+\.?\d*)', out)
        geomean_reported = float(gm_match.group(1)) if gm_match else None
        has_error = any(kw in out for kw in ["Error", "error", "FAILED", "Traceback", "Memory access fault"])
        if not times and has_error:
            return None, out[:500]
        if not times and "processing" not in out.split('\n')[-2:]:
            return None, out[:500]
        return (times, geomean_reported), out[:500]
    return None, "RATE_LIMITED_3x"


def geomean(times_dict):
    vals = [v for v in times_dict.values() if v > 0]
    if not vals: return float("inf")
    product = 1.0
    for v in vals: product *= v
    return product ** (1.0 / len(vals))


def main():
    state = load_state()
    print(f"=== MoE Coordinate Descent Sweeper ===")
    print(f"Started: {datetime.now()}")
    print(f"Previous subs: {state['total_submissions']}, best: {state['best_geomean']}")

    best_d2048 = state["best_d2048"]
    best_e257 = state["best_e257"]
    best_gm = state["best_geomean"]
    total_subs = state["total_submissions"]
    tested = set(tuple(x) if isinstance(x, list) else x for x in state.get("tested", []))

    # Phase 1: d=2048 kernel sweep
    print(f"\n=== Phase 1: d=2048 Kernel Sweep ===")
    for s1_name in S1_KERNELS:
        for s2_name in S2_KERNELS:
            for bm in BLOCK_M_OPTIONS:
                for sk in SPLITK_OPTIONS:
                    combo_key = f"{s1_name}_{s2_name}_bm{bm}_sk{sk}"
                    if combo_key in tested:
                        continue

                    cfg = {"s1": s1_name, "s2": s2_name, "bm": bm, "sk": sk}
                    total_subs += 1
                    filepath = generate_submission(cfg, best_e257, combo_key, total_subs)
                    print(f"  [{total_subs}] {combo_key}...", end=" ", flush=True)

                    result, raw = submit(filepath)
                    tested.add(combo_key)

                    if result is None:
                        print("FAILED")
                        log({"type": "fail", "config": combo_key, "error": raw[:200]})
                    else:
                        times, reported_gm = result
                        gm = reported_gm or geomean(times)
                        if gm < best_gm:
                            print(f"BETTER! {gm:.1f}μs (was {best_gm:.1f}μs)")
                            best_gm = gm
                            best_d2048 = cfg
                            log({"type": "improvement", "config": combo_key, "geomean": gm,
                                 "times": times})
                            # Auto-promote to leaderboard
                            print(f"    >> Submitting to leaderboard...")
                            lb_path = generate_submission(cfg, best_e257, f"best_{combo_key}", total_subs + 1)
                            submit(lb_path, mode="leaderboard")
                            total_subs += 1
                        else:
                            print(f"{gm:.1f}μs")
                            log({"type": "tested", "config": combo_key, "geomean": gm})

                    state = {
                        "best_d2048": best_d2048,
                        "best_e257": best_e257,
                        "best_geomean": best_gm,
                        "total_submissions": total_subs,
                        "tested": list(tested),
                    }
                    save_state(state)

    # Phase 2: E=257 injection (NEW - untested territory!)
    print(f"\n=== Phase 2: E=257 CK Kernel Injection ===")
    for e257_bm in [32, 64, 128]:
        e257_cfg = {"inject": True, "bm": e257_bm}
        combo_key = f"e257_bm{e257_bm}"
        if combo_key in tested:
            continue

        total_subs += 1
        filepath = generate_submission(best_d2048, e257_cfg, combo_key, total_subs)
        print(f"  [{total_subs}] E=257 inject bm={e257_bm}...", end=" ", flush=True)

        result, raw = submit(filepath)
        tested.add(combo_key)

        if result is None:
            print("FAILED")
            log({"type": "fail", "config": combo_key, "error": raw[:200]})
        else:
            times, reported_gm = result
            gm = reported_gm or geomean(times)
            if gm < best_gm:
                print(f"BETTER! {gm:.1f}μs!")
                best_gm = gm
                best_e257 = e257_cfg
                log({"type": "e257_improvement", "config": combo_key, "geomean": gm})
                lb_path = generate_submission(best_d2048, e257_cfg, f"best_{combo_key}", total_subs + 1)
                submit(lb_path, mode="leaderboard")
                total_subs += 1
            else:
                print(f"{gm:.1f}μs")

        save_state({
            "best_d2048": best_d2048, "best_e257": best_e257,
            "best_geomean": best_gm, "total_submissions": total_subs,
            "tested": list(tested),
        })

    # Phase 3: Continuous random exploration
    print(f"\n=== Phase 3: Continuous Random Exploration ===")
    while True:
        s1 = random.choice(list(S1_KERNELS.keys()))
        s2 = random.choice(list(S2_KERNELS.keys()))
        bm = random.choice(BLOCK_M_OPTIONS)
        sk = random.choice(SPLITK_OPTIONS)
        e257 = random.choice([True, False])
        e257_bm = random.choice([16, 32, 64])

        cfg = {"s1": s1, "s2": s2, "bm": bm, "sk": sk}
        e_cfg = {"inject": e257, "bm": e257_bm}
        combo_key = f"rand_{s1}_{s2}_bm{bm}_sk{sk}_e{e257}_ebm{e257_bm}"

        if combo_key in tested:
            continue

        total_subs += 1
        filepath = generate_submission(cfg, e_cfg, combo_key, total_subs)
        print(f"  [{total_subs}] Random: {combo_key}...", end=" ", flush=True)

        result, raw = submit(filepath)
        tested.add(combo_key)

        if result is None:
            print("FAILED")
        else:
            times, reported_gm = result
            gm = reported_gm or geomean(times)
            if gm < best_gm:
                print(f"BETTER! {gm:.1f}μs!")
                best_gm = gm
                best_d2048 = cfg
                if e257: best_e257 = e_cfg
                lb_path = generate_submission(cfg, e_cfg, f"best_rand_{total_subs}", total_subs + 1)
                submit(lb_path, mode="leaderboard")
                total_subs += 1
            else:
                print(f"{gm:.1f}μs")

        save_state({
            "best_d2048": best_d2048, "best_e257": best_e257,
            "best_geomean": best_gm, "total_submissions": total_subs,
            "tested": list(tested),
        })


if __name__ == "__main__":
    main()

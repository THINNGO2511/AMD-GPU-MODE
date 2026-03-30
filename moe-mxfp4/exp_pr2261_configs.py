#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE d=2048 Config Sweep: Try ALL available CK kernel combinations for d=2048.

STRATEGY:
Instead of guessing which kernel works, enumerate every stage1+stage2 .co file
that could handle d=2048 and try them in a controlled A/B test.

For E=33 bs=512 d=2048:
  - token_num = 512 * 9 = 4608 (after routing)
  - est_m = 4608 / 33 ~ 140 tokens per expert
  - Stage1: gate_up GEMM [est_m, 7168] x [7168, 2*2048] (with SiLU)
  - Stage2: down GEMM [est_m, 2048] x [2048, 7168] (with weighted sum)

EXPERIMENTS:
1. Enumerate all .co kernel binaries — find ones compatible with d=2048 K/N dims
2. Try splitk=1 vs splitk=0 (auto) for d=2048 stage1
3. Try Nswizzle variants if available
4. Try v1 vs v3 stage2 kernels for d=2048
5. Try block_m=64 vs 128 vs 256 for d=2048
6. Try use_nt=True specifically for d=2048 (opposite of global False)
7. Time each variant and report

The key insight: d=2048 uses different .co kernels than d=256/512.
The auto-selection might not be optimal. PR #2261 added E=32 d=2048 configs
but they may not match E=33 exactly.
"""
import torch
import functools
import sys
import os
import time
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_timings = {}
_variant_name = "default"
_probed = False

# === ALL KNOWN CK KERNEL NAMES ===
# Stage1 variants (gate+up GEMM with SiLU activation)
STAGE1_KERNELS = {
    "s1_64x32": "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256x32": "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256x64": "moe_ck2stages_gemm1_256x64x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
    "s1_256x128": "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16",
}

# Stage2 variants (down GEMM with weighted accumulation)
STAGE2_KERNELS = {
    "s2_64x32_v1": "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "s2_256x32_v3": "moe_ck2stages_gemm2_256x32x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "s2_256x64_v3": "moe_ck2stages_gemm2_256x64x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
    "s2_256x128_v3": "moe_ck2stages_gemm2_256x128x128x128_1x4_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16",
}

# Configuration matrix: (stage1_key, stage2_key, block_m, splitk, use_nt)
# Only configure for d>=2048 shapes. d<2048 uses proven injection.
D2048_CONFIGS = [
    # Config 0: NO injection (auto, current baseline)
    None,
    # Config 1: PR #2261 style - large tiles, block_m=128 (for est_m>=100)
    ("s1_256x128", "s2_256x128_v3", 128, 0, False),
    # Config 2: Medium tiles - 256x64, block_m=64
    ("s1_256x64", "s2_256x64_v3", 64, 0, False),
    # Config 3: Small tiles - 64x32 stage1, 64x32 stage2 v1 (minimal tile)
    ("s1_64x32", "s2_64x32_v1", 32, 0, False),
    # Config 4: 256x32 stage1 + 256x32 stage2 (medium stage1, wider stage2)
    ("s1_256x32", "s2_256x32_v3", 64, 0, False),
    # Config 5: PR #2261 small-token config (s1_64x32 + s2_256x32)
    ("s1_64x32", "s2_256x32_v3", 32, 0, False),
    # Config 6: Large stage1, small stage2 (asymmetric)
    ("s1_256x128", "s2_64x32_v1", 128, 0, False),
    # Config 7: 256x64 stage1 + 64x32 stage2 v1 (proven d<2048 stage2)
    ("s1_256x64", "s2_64x32_v1", 64, 0, False),
    # Config 8: 256x32 + v1 small stage2, block_m=32
    ("s1_256x32", "s2_64x32_v1", 32, 0, False),
    # Config 9: Auto with use_nt=True for d=2048
    None,  # handled specially
    # Config 10: Auto with block_m=64 (override default)
    None,  # handled specially
    # Config 11: 256x128 with splitk=1
    ("s1_256x128", "s2_256x128_v3", 128, 1, False),
    # Config 12: 256x64 with block_m=128
    ("s1_256x64", "s2_256x64_v3", 128, 0, False),
]

# Which config to run THIS submission (cycle through on each submission)
# Change this number to test different configs: 0-12
ACTIVE_CONFIG = 0  # Set to 0 for baseline, then increment


def _probe_kernels():
    """Discover all available .co files on the runner."""
    global _probed
    if _probed:
        return
    _probed = True

    print("=" * 80, file=sys.stderr)
    print(f"=== PR2261 CONFIG SWEEP: ACTIVE_CONFIG={ACTIVE_CONFIG} ===", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # Enumerate .co files to discover NEW kernels not in our lists
    co_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    try:
        if os.path.exists(co_dir):
            all_cos = sorted(os.listdir(co_dir))
            known_s1 = set(STAGE1_KERNELS.values())
            known_s2 = set(STAGE2_KERNELS.values())

            # Find NEW kernels we don't know about
            new_s1 = []
            new_s2 = []
            for co in all_cos:
                name = co.replace('.co', '')
                if 'gemm1' in name and name not in known_s1:
                    new_s1.append(name)
                elif 'gemm2' in name and name not in known_s2:
                    new_s2.append(name)

            if new_s1:
                print(f"\n[DISCOVERY] NEW stage1 kernels ({len(new_s1)}):", file=sys.stderr)
                for k in new_s1:
                    print(f"  {k}", file=sys.stderr)
            if new_s2:
                print(f"\n[DISCOVERY] NEW stage2 kernels ({len(new_s2)}):", file=sys.stderr)
                for k in new_s2:
                    print(f"  {k}", file=sys.stderr)

            # Check file sizes — larger .co might be more optimized kernels
            print(f"\n[CO] Kernel file sizes:", file=sys.stderr)
            for co in all_cos:
                fpath = os.path.join(co_dir, co)
                sz = os.path.getsize(fpath)
                print(f"  {co}: {sz} bytes", file=sys.stderr)
    except Exception as e:
        print(f"[CO] Error: {e}", file=sys.stderr)

    # Check for FlyDSL kernels
    try:
        flydsl_dir = "/home/runner/aiter/hsa/gfx950/flydsl/"
        if os.path.exists(flydsl_dir):
            entries = sorted(os.listdir(flydsl_dir))
            print(f"\n[FLYDSL] {flydsl_dir}: {len(entries)} files", file=sys.stderr)
            for e in entries[:20]:
                print(f"  {e}", file=sys.stderr)
    except:
        pass

    # Check for CKTile MoE kernels
    try:
        cktile_dir = "/home/runner/aiter/hsa/gfx950/cktile_fmoe/"
        if os.path.exists(cktile_dir):
            entries = sorted(os.listdir(cktile_dir))
            print(f"\n[CKTILE] {cktile_dir}: {len(entries)} files", file=sys.stderr)
            for e in entries[:20]:
                print(f"  {e}", file=sys.stderr)
    except:
        pass

    # Check what the CURRENT auto-selection picks for d=2048
    # This helps us know the baseline
    print(f"\n[CFG] Active config: {ACTIVE_CONFIG}", file=sys.stderr)
    if ACTIVE_CONFIG < len(D2048_CONFIGS) and D2048_CONFIGS[ACTIVE_CONFIG] is not None:
        s1k, s2k, bm, sk, nt = D2048_CONFIGS[ACTIVE_CONFIG]
        print(f"  stage1: {s1k} -> {STAGE1_KERNELS.get(s1k, '?')[:80]}...", file=sys.stderr)
        print(f"  stage2: {s2k} -> {STAGE2_KERNELS.get(s2k, '?')[:80]}...", file=sys.stderr)
        print(f"  block_m={bm} splitk={sk} use_nt={nt}", file=sys.stderr)
    else:
        print(f"  Using AUTO selection (no injection for d>=2048)", file=sys.stderr)

    # Dump the config resolution path for d=2048 shapes
    import csv
    csv_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
    try:
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames
                print(f"\n[CSV] Headers: {headers}", file=sys.stderr)
                for row in reader:
                    e = row.get('expert', row.get('num_expert', ''))
                    d = row.get('inter_dim', row.get('intermediate_size', ''))
                    cu = row.get('cu_num', '')
                    # Show E=32/33 entries
                    if e in ('32', '33'):
                        print(f"  [E={e}] {dict(row)}", file=sys.stderr)
                    # Also show d=2048+ entries
                    elif d and int(d) >= 2048:
                        print(f"  [d={d}] {dict(row)}", file=sys.stderr)
    except Exception as e:
        print(f"[CSV] Error: {e}", file=sys.stderr)

    # Check for runner aiter version and recent changes
    try:
        import aiter
        print(f"\n[VER] aiter version: {getattr(aiter, '__version__', '?')}", file=sys.stderr)
        print(f"[VER] aiter path: {aiter.__file__}", file=sys.stderr)
    except:
        pass

    # Check if there are new APIs on the fused_moe module
    import inspect
    try:
        funcs = [(name, getattr(fm, name)) for name in dir(fm)
                 if callable(getattr(fm, name, None)) and not name.startswith('__')]
        print(f"\n[API] fused_moe callable attributes:", file=sys.stderr)
        for name, func in sorted(funcs):
            try:
                sig = inspect.signature(func)
                print(f"  {name}{sig}", file=sys.stderr)
            except:
                print(f"  {name}(...)", file=sys.stderr)
    except Exception as e:
        print(f"[API] Error: {e}", file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe_kernels()

    # Standard: use_nt=False for E<=64
    orig_use_nt = fm.use_nt
    if ACTIVE_CONFIG == 9:
        # Config 9: try use_nt=True for d=2048, False for others
        fm.use_nt = lambda t, k, e: True if e <= 64 else orig_use_nt(t, k, e)
        print("[PATCH] use_nt=True for E<=64 (config 9 experiment)", file=sys.stderr)
    else:
        fm.use_nt = lambda token, topk, expert: False

    # Opus sorting
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # block_m selection
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if inter_dim >= 2048:
                # Check if active config overrides block_m
                cfg = D2048_CONFIGS[ACTIVE_CONFIG] if ACTIVE_CONFIG < len(D2048_CONFIGS) else None
                if cfg is not None:
                    _, _, bm, _, _ = cfg
                    return bm
                elif ACTIVE_CONFIG == 10:
                    return 64  # Config 10: force block_m=64
                # Default d=2048 block_m
                if est_m >= 100:
                    return 128
                else:
                    return 64
            else:
                if est_m < 50:
                    return 32
                else:
                    return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # CK kernel injection
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)

        if expert <= 64 and q_type == QuantType.per_1x32 and not result.run_1stage:
            est_m = token * topk // expert if expert > 0 else 0

            if inter_dim >= 2048:
                # d=2048: apply ACTIVE_CONFIG
                cfg = D2048_CONFIGS[ACTIVE_CONFIG] if ACTIVE_CONFIG < len(D2048_CONFIGS) else None

                if cfg is not None:
                    s1_key, s2_key, bm, splitk, use_nt = cfg
                    s1_name = STAGE1_KERNELS.get(s1_key, '')
                    s2_name = STAGE2_KERNELS.get(s2_key, '')

                    if s1_name and s2_name:
                        print(f"[INJECT] d={inter_dim} est_m={est_m} config={ACTIVE_CONFIG}: "
                              f"s1={s1_key} s2={s2_key} bm={bm} sk={splitk}", file=sys.stderr)
                        try:
                            return fm.MOEMetadata(
                                functools.partial(fm.ck_moe_stage1,
                                    kernelName=s1_name, activation=activation,
                                    quant_type=q_type, dtype=dtype,
                                    splitk=splitk, use_non_temporal_load=use_nt),
                                functools.partial(aiter.ck_moe_stage2_fwd,
                                    kernelName=s2_name, activation=activation,
                                    quant_type=q_type, use_non_temporal_load=use_nt),
                                bm, 0, False)
                        except Exception as e:
                            print(f"[INJECT] d=2048 error: {e}", file=sys.stderr)
                else:
                    # Config is None = auto (log what auto picks)
                    s1kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    s2kw = result.stage2.keywords if hasattr(result.stage2, 'keywords') else {}
                    print(f"[AUTO] d={inter_dim} est_m={est_m} bm={result.block_m} "
                          f"s1={s1kw.get('kernelName', 'DEFAULT')[:60]} "
                          f"s2={s2kw.get('kernelName', 'DEFAULT')[:60]}", file=sys.stderr)

            elif inter_dim < 2048:
                # d<2048: proven CK injection
                try:
                    kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    if not kw.get('kernelName', ''):
                        S1_64 = STAGE1_KERNELS["s1_64x32"]
                        S1_256 = STAGE1_KERNELS["s1_256x32"]
                        S2_V1 = STAGE2_KERNELS["s2_64x32_v1"]
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
                except Exception:
                    pass

        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()
    _call_count += 1
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    bs = hidden_states.shape[0]
    E = config.get("n_expert", 0)
    d = config.get("d_expert", 0)
    shape_key = f"bs{bs}_E{E}_d{d}"

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Time every call for accurate per-shape profiling
    if _call_count <= 150:
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    result = fused_moe(
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

    if _call_count <= 150:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1e6
        if shape_key not in _timings:
            _timings[shape_key] = []
        _timings[shape_key].append(elapsed_us)

        if _call_count <= 20 or d >= 2048:
            print(f"[TIME] call={_call_count} {shape_key}: {elapsed_us:.1f}us "
                  f"(cfg={ACTIVE_CONFIG})", file=sys.stderr)

    # Print per-shape timing summary periodically
    if _call_count in (50, 100, 150):
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"[SUMMARY] Config={ACTIVE_CONFIG}, after {_call_count} calls:", file=sys.stderr)
        for key, times in sorted(_timings.items()):
            # Skip first 2 warmup calls per shape
            warmup = 2
            times_steady = [t for i, t in enumerate(times) if i >= warmup]
            if times_steady:
                avg = sum(times_steady) / len(times_steady)
                mn = min(times_steady)
                mx = max(times_steady)
                p50 = sorted(times_steady)[len(times_steady)//2]
                is_d2048 = "d2048" in key
                marker = " <<<< TARGET" if is_d2048 else ""
                print(f"  {key}: avg={avg:.1f}us p50={p50:.1f}us min={mn:.1f}us "
                      f"max={mx:.1f}us n={len(times_steady)}{marker}", file=sys.stderr)
        # Compute approximate geomean
        shape_avgs = []
        for key, times in sorted(_timings.items()):
            times_steady = [t for i, t in enumerate(times) if i >= 2]
            if times_steady:
                shape_avgs.append(sum(times_steady) / len(times_steady))
        if shape_avgs:
            import math
            geomean = math.exp(sum(math.log(x) for x in shape_avgs) / len(shape_avgs))
            print(f"  GEOMEAN (approx): {geomean:.1f}us", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)

    return result

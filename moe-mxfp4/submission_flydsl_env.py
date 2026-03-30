#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE -- FlyDSL environment variable activation + proven CK injection.

Strategy: Set FlyDSL env vars BEFORE importing aiter so the fused_moe
C++ pipeline selects FlyDSL kernels internally (instead of us calling
flydsl_moe_stage1/stage2 directly which crashes). This is the approach
used by Kimi K2.5 -- the fused_moe wrapper detects FlyDSL availability
via env vars and routes to FlyDSL kernels automatically.

Env vars tested:
  AITER_USE_FLYDSL_MOE=1        -- global FlyDSL MoE enable
  AITER_USE_FLYDSL_MOE_STAGE1=1 -- FlyDSL for stage1 (gate+up+SiLU)
  AITER_USE_FLYDSL_MOE_STAGE2=1 -- FlyDSL for stage2 (down proj)
  AITER_ENFORCE_DSL=1           -- force DSL path even without CSV match
  CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT=3 -- CK tile bf16 conversion mode
  AITER_USE_NT=0                -- disable non-temporal loads
  CU_NUM=256                    -- match runner CU count

Fallback: if FlyDSL env vars don't change behavior, the proven CK
injection for E<=64 d<2048 still applies, with use_nt=False globally,
opus sorting, and block_m tuning.

Diagnostics: prints shape/time/config to stderr for each call.
"""
import os
import sys
import time

# ============================================================
# SET ENV VARS *BEFORE* ANY AITER IMPORTS
# This is critical -- aiter reads these at module load time
# ============================================================

# FlyDSL MoE activation
os.environ["AITER_USE_FLYDSL_MOE"] = "1"
os.environ["AITER_USE_FLYDSL_MOE_STAGE1"] = "1"
os.environ["AITER_USE_FLYDSL_MOE_STAGE2"] = "1"

# Force DSL path even if CSV doesn't have a matching entry
os.environ["AITER_ENFORCE_DSL"] = "1"

# CK tile bf16 conversion mode (may affect kernel selection)
os.environ["CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT"] = "3"

# Standard env vars
os.environ["AITER_USE_NT"] = "0"          # non-temporal loads OFF
os.environ["CU_NUM"] = "256"              # match MI355X runner
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"  # opus sort kernel

# ============================================================
# NOW import aiter
# ============================================================
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

# ============================================================
# Globals
# ============================================================
_patched = False
_call_count = 0
_timings = {}  # shape_key -> [times]

# CK kernel names (proven 15% faster for E<=64 d<2048)
STAGE1_64 = (
    "moe_ck2stages_gemm1_64x32x32x128_1x1_"
    "MulABScaleShuffled_v3_Nswizzle0_Quant3_"
    "MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
)
STAGE1_256 = (
    "moe_ck2stages_gemm1_256x32x128x128_1x4_"
    "MulABScaleShuffled_v3_Nswizzle0_Quant3_"
    "MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
)
STAGE2_V1 = (
    "moe_ck2stages_gemm2_64x32x32x128_1x1_"
    "MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_"
    "MulRoutedWeight1_FP4X2_FP4X2_B16"
)


def _probe_flydsl_state():
    """One-time probe: report what FlyDSL state aiter picked up from env vars."""
    # Check if FlyDSL is available
    try:
        avail = fm.is_flydsl_available() if hasattr(fm, 'is_flydsl_available') else "N/A"
        print(f"[FLYDSL_ENV] is_flydsl_available={avail}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL_ENV] avail check: {e}", file=sys.stderr)

    # Check for FlyDSL-related attributes on fm
    fly_attrs = [a for a in dir(fm) if 'fly' in a.lower() or 'dsl' in a.lower()]
    print(f"[FLYDSL_ENV] fm fly/dsl attrs: {fly_attrs}", file=sys.stderr)

    # Check _USE_OPUS_MOE_SORTING
    if hasattr(fm, '_USE_OPUS_MOE_SORTING'):
        print(f"[FLYDSL_ENV] _USE_OPUS_MOE_SORTING={fm._USE_OPUS_MOE_SORTING}", file=sys.stderr)

    # Check env vars as seen by aiter
    for key in ["AITER_USE_FLYDSL_MOE", "AITER_USE_FLYDSL_MOE_STAGE1",
                "AITER_USE_FLYDSL_MOE_STAGE2", "AITER_ENFORCE_DSL",
                "CK_TILE_FLOAT_TO_BFLOAT16_DEFAULT", "AITER_USE_NT",
                "CU_NUM", "AITER_USE_OPUS_MOE_SORTING"]:
        val = os.environ.get(key, "<not set>")
        print(f"[FLYDSL_ENV] {key}={val}", file=sys.stderr)

    # Probe get_2stage_cfgs to see if FlyDSL is being selected
    try:
        import inspect
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        # Find lines mentioning flydsl
        lines = src.split('\n')
        fly_lines = [(i+1, l) for i, l in enumerate(lines) if 'flydsl' in l.lower()]
        if fly_lines:
            print(f"[FLYDSL_ENV] get_2stage_cfgs has {len(fly_lines)} flydsl refs:", file=sys.stderr)
            for ln, l in fly_lines[:10]:
                print(f"  L{ln}: {l.rstrip()}", file=sys.stderr)
                # Also show 2 lines of context after
                for ctx in range(ln, min(ln+3, len(lines))):
                    print(f"  L{ctx+1}: {lines[ctx].rstrip()}", file=sys.stderr)
                print(f"  ---", file=sys.stderr)
        else:
            print(f"[FLYDSL_ENV] get_2stage_cfgs has NO flydsl refs!", file=sys.stderr)

        # Show total line count
        print(f"[FLYDSL_ENV] get_2stage_cfgs: {len(lines)} lines total", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL_ENV] source probe: {e}", file=sys.stderr)

    # Check if any FlyDSL kernel binaries exist
    try:
        import glob as globmod
        hsa_base = '/home/runner/aiter/hsa/gfx950/'
        fly_bins = []
        for root, dirs, files in os.walk(hsa_base):
            for f in files:
                if 'flydsl' in f.lower():
                    fly_bins.append(os.path.join(root, f))
        print(f"[FLYDSL_ENV] FlyDSL binaries: {len(fly_bins)}", file=sys.stderr)
        for p in fly_bins[:5]:
            print(f"  {p}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL_ENV] binary scan: {e}", file=sys.stderr)

    # Check for FlyDSL-related env var handling in aiter source
    try:
        import importlib
        # Check if flydsl module has config reading
        if hasattr(fm, '_flydsl_stage2_wrapper'):
            print(f"[FLYDSL_ENV] _flydsl_stage2_wrapper exists", file=sys.stderr)
            sig = inspect.signature(fm._flydsl_stage2_wrapper)
            print(f"[FLYDSL_ENV] _flydsl_stage2_wrapper sig: {sig}", file=sys.stderr)
        if hasattr(fm, '_flydsl_stage1_wrapper'):
            print(f"[FLYDSL_ENV] _flydsl_stage1_wrapper exists", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL_ENV] wrapper probe: {e}", file=sys.stderr)

    # Check cfg_2stages cache state
    try:
        if hasattr(fm, 'cfg_2stages'):
            print(f"[FLYDSL_ENV] cfg_2stages={fm.cfg_2stages}", file=sys.stderr)
        if hasattr(fm, '_use_flydsl_moe'):
            print(f"[FLYDSL_ENV] _use_flydsl_moe={fm._use_flydsl_moe}", file=sys.stderr)
        # Check module-level flags set from env vars
        for attr in ['_USE_FLYDSL', '_FLYDSL_ENABLED', '_use_flydsl',
                     'USE_FLYDSL_MOE', 'use_flydsl_moe', '_FLYDSL_MOE']:
            if hasattr(fm, attr):
                print(f"[FLYDSL_ENV] fm.{attr}={getattr(fm, attr)}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL_ENV] state probe: {e}", file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # ---- Probe FlyDSL state first ----
    _probe_flydsl_state()

    # ---- 1. use_nt=False for ALL shapes ----
    fm.use_nt = lambda token, topk, expert: False

    # ---- 2. Force opus sorting ----
    if hasattr(fm, '_USE_OPUS_MOE_SORTING'):
        fm._USE_OPUS_MOE_SORTING = True

    # ---- 3. block_m tuning ----
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128  # d=2048 large batch: default=128 is better
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # ---- 4. get_2stage_cfgs: CK injection for E<=64 d<2048 ----
    # FlyDSL env vars may cause the original get_2stage_cfgs to select
    # FlyDSL kernels automatically. We only override for the proven
    # CK kernel injection cases (E<=64 d<2048).
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

        # Log what the original function selected (FlyDSL or CK?)
        s1_name = ""
        s2_name = ""
        try:
            if hasattr(result, 'stage1') and hasattr(result.stage1, 'keywords'):
                s1_name = result.stage1.keywords.get('kernelName', '')
            if hasattr(result, 'stage2') and hasattr(result.stage2, 'keywords'):
                s2_name = result.stage2.keywords.get('kernelName', '')
            if hasattr(result, 'stage1') and hasattr(result.stage1, 'func'):
                s1_func = result.stage1.func.__name__ if hasattr(result.stage1.func, '__name__') else str(result.stage1.func)
            else:
                s1_func = "?"
            if hasattr(result, 'stage2') and hasattr(result.stage2, 'func'):
                s2_func = result.stage2.func.__name__ if hasattr(result.stage2.func, '__name__') else str(result.stage2.func)
            else:
                s2_func = "?"
            is_1stage = result.run_1stage if hasattr(result, 'run_1stage') else "?"
            bm = result.block_m if hasattr(result, 'block_m') else "?"
            ks = result.ksplit if hasattr(result, 'ksplit') else "?"
            print(f"[FLYDSL_ENV] get_2stage_cfgs(E={expert},d={inter_dim},tok={token}): "
                  f"1stage={is_1stage} bm={bm} ks={ks} "
                  f"s1_func={s1_func} s1_kn={s1_name[:60]} "
                  f"s2_func={s2_func} s2_kn={s2_name[:60]}",
                  file=sys.stderr)
            # Check if FlyDSL was selected
            if 'flydsl' in s1_name.lower() or 'flydsl' in s2_name.lower():
                print(f"[FLYDSL_ENV] *** FLYDSL KERNEL SELECTED! ***", file=sys.stderr)
            if 'flydsl' in s1_func.lower() or 'flydsl' in s2_func.lower():
                print(f"[FLYDSL_ENV] *** FLYDSL WRAPPER SELECTED! ***", file=sys.stderr)
        except Exception as e:
            print(f"[FLYDSL_ENV] log err: {e}", file=sys.stderr)

        # For E<=64 d<2048: inject proven CK kernels (don't override FlyDSL
        # if the env vars caused it to be selected -- let it run!)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                # Check if FlyDSL was already selected for this config
                s1_is_flydsl = ('flydsl' in s1_name.lower() or 'flydsl' in s1_func.lower())
                s2_is_flydsl = ('flydsl' in s2_name.lower() or 'flydsl' in s2_func.lower())
                if s1_is_flydsl or s2_is_flydsl:
                    # FlyDSL was selected -- let it run, don't override
                    print(f"[FLYDSL_ENV] Keeping FlyDSL selection for E={expert} d={inter_dim}",
                          file=sys.stderr)
                    return result

                # No FlyDSL -- inject proven CK kernels
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass

        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None
    print(f"[FLYDSL_ENV] All patches applied", file=sys.stderr)


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

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = topk_ids.max().item() + 1 if topk_ids.numel() > 0 else 0
    topk = topk_ids.shape[1]
    d_hidden = config.get("d_hidden", hidden_states.shape[1])
    d_expert = config.get("d_expert", 0)
    shape_key = f"M{M}_E{E}_d{d_expert}_k{topk}"

    t0 = time.time()

    out = fused_moe(
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

    torch.cuda.synchronize()
    dt = (time.time() - t0) * 1e6  # microseconds

    if shape_key not in _timings:
        _timings[shape_key] = []
    _timings[shape_key].append(dt)
    n = len(_timings[shape_key])

    # Print timing diagnostics (first 3 calls per shape, then every 10th)
    if n <= 3 or n % 10 == 0:
        avg = sum(_timings[shape_key][-10:]) / min(n, 10)
        print(f"[FLYDSL_ENV] call={_call_count} {shape_key} "
              f"dt={dt:.0f}us avg={avg:.0f}us (n={n})",
              file=sys.stderr)

    return out

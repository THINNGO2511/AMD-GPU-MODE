#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Profiling v3 — Steady-state timing (benchmark mode).
Only profiles after warmup (call #4+), skips JIT noise.
Also reads fused_dynamic_mxfp4_quant_moe_sort source to understand the fused kernel.
"""
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_phase_times = {}
_warmup_done = {}  # per case_key

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _timed(orig_fn, name):
    @functools.wraps(orig_fn)
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        result = orig_fn(*args, **kwargs)
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 1000
        _phase_times.setdefault(name, []).append(us)
        return result
    return wrapper


def _dump_fused_quant_sort_source():
    """Read and dump the fused_dynamic_mxfp4_quant_moe_sort source."""
    import inspect, os
    try:
        src = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
        lines = src.split('\n')
        print(f"[PROF] fused_dynamic_mxfp4_quant_moe_sort ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:100]):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[PROF] Cannot get source: {e}", file=sys.stderr)

    # Also dump moe_sorting source
    try:
        src = inspect.getsource(fm.moe_sorting)
        lines = src.split('\n')
        print(f"\n[PROF] moe_sorting ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:80]):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[PROF] Cannot get moe_sorting source: {e}", file=sys.stderr)

    # Read fused_moe_2stages source
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        lines = src.split('\n')
        print(f"\n[PROF] fused_moe_2stages ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:120]):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[PROF] Cannot get fused_moe_2stages source: {e}", file=sys.stderr)

    # Read fused_moe source (the top-level Python function)
    try:
        src = inspect.getsource(fm.fused_moe)
        lines = src.split('\n')
        print(f"\n[PROF] fused_moe ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:150]):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[PROF] Cannot get fused_moe source: {e}", file=sys.stderr)

    # Check Triton sort kernel
    sort_paths = [
        '/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/moe_sort.py',
        '/home/runner/aiter/aiter/ops/triton/moe/moe_sort.py',
    ]
    for p in sort_paths:
        if os.path.exists(p):
            content = open(p).read()
            lines = content.split('\n')
            print(f"\n[PROF] {p} ({len(lines)} lines):", file=sys.stderr)
            for i, line in enumerate(lines[:120]):
                print(f"  {i+1}: {line}", file=sys.stderr)
            break

    # Also find the Triton quant+sort kernel
    try:
        triton_base = '/home/runner/aiter/aiter/ops/triton/'
        for root, dirs, files in os.walk(triton_base):
            for f in files:
                if ('quant' in f.lower() and 'sort' in f.lower()) or f == 'moe_sort.py':
                    fp = os.path.join(root, f)
                    print(f"\n[PROF] Found: {fp}", file=sys.stderr)
                    content = open(fp).read()
                    for i, line in enumerate(content.split('\n')[:80]):
                        print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[PROF] Walk error: {e}", file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _dump_fused_quant_sort_source()

    # Wrap timing on key functions
    for mod, mn in [(fm, 'fm'), (aiter, 'aiter')]:
        for attr in ['fused_dynamic_mxfp4_quant_moe_sort']:
            if hasattr(mod, attr) and callable(getattr(mod, attr)):
                setattr(mod, attr, _timed(getattr(mod, attr), 'quant_sort'))
                print(f"[PROF] Wrapped {mn}.{attr}", file=sys.stderr)

    for mod, mn in [(fm, 'fm')]:
        for attr in ['moe_sorting']:
            if hasattr(mod, attr) and callable(getattr(mod, attr)):
                setattr(mod, attr, _timed(getattr(mod, attr), 'sorting'))
                print(f"[PROF] Wrapped {mn}.{attr}", file=sys.stderr)

    # Only wrap top-level stage functions (avoid double-counting nested calls)
    fm.ck_moe_stage1 = _timed(fm.ck_moe_stage1, 'stage1')
    if hasattr(aiter, 'ck_moe_stage2_fwd'):
        aiter.ck_moe_stage2_fwd = _timed(aiter.ck_moe_stage2_fwd, 'stage2')
    print(f"[PROF] Wrapped stage1/stage2", file=sys.stderr)

    # Apply best_kernels optimizations
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

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
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
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
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception as e:
                print(f"[PROF] inject err: {e}", file=sys.stderr)
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None
    print(f"[PROF] All patches applied.\n", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    _call_count += 1

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = topk_ids.max().item() + 1
    topk = topk_ids.shape[1]
    d_expert = config.get("d_expert", 0)
    case_key = (M, E, d_expert)

    # Track warmup per case
    _warmup_done.setdefault(case_key, 0)
    _warmup_done[case_key] += 1
    is_warmed = _warmup_done[case_key] >= 3

    _phase_times.clear()

    if is_warmed:
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()

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

    if is_warmed and _warmup_done[case_key] <= 6:
        t1.record()
        torch.cuda.synchronize()
        total_us = t0.elapsed_time(t1) * 1000

        print(f"\n{'='*70}", file=sys.stderr)
        print(f"[PROF] #{_call_count} (run {_warmup_done[case_key]}) | "
              f"bs={M} E={E} topk={topk} d={d_expert} | TOTAL={total_us:.0f}us",
              file=sys.stderr)

        accounted = 0
        for name in ['quant_sort', 'sorting', 'stage1', 'stage2']:
            if name in _phase_times:
                t = sum(_phase_times[name])
                pct = 100 * t / total_us if total_us > 0 else 0
                ncalls = len(_phase_times[name])
                print(f"  {name}: {t:.0f}us ({pct:.0f}%) [{ncalls}x]", file=sys.stderr)
                accounted += t

        overhead = total_us - accounted
        pct_oh = 100 * overhead / total_us if total_us > 0 else 0
        print(f"  [python/other]: {overhead:.0f}us ({pct_oh:.0f}%)", file=sys.stderr)
        print(f"{'='*70}", file=sys.stderr)

    return result

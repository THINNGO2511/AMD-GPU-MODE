#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE d=2048 Deep Probe: Source dump + quant path analysis + new parameter discovery.

GOALS:
1. Read and dump key MoE source files on the runner to find new optimization paths
2. Check fused_moe_2stages.py for new parameters (blockPerCu, splitK variants)
3. Dump the full CSV for E=33 d=2048 entries with cu_num=256
4. Check if token_num_quant_moe_sort_switch is tunable
5. Look for any new env vars or parameters added since session 6
6. Try forcing the fused quant+sort path for ALL token counts (threshold=8192)
7. Enumerate ALL available .co kernel binaries that might work for d=2048

The d=2048 shape (E=33, bs=512, est_m~140) is 333us and dominates our geomean.
Finding even a 10% improvement here drops geomean from 169us to ~155us.
"""
import torch
import functools
import sys
import os
import time
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_timings = {}  # shape_key -> [times]
_source_dumped = False

# Proven CK kernels for E<=64 d<2048
S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _dump_sources():
    """Read and dump key MoE source files to find optimization paths."""
    global _source_dumped
    if _source_dumped:
        return
    _source_dumped = True

    print("=" * 80, file=sys.stderr)
    print("=== MoE d=2048 DEEP SOURCE PROBE ===", file=sys.stderr)
    print("=" * 80, file=sys.stderr)

    # 1. Dump fused_moe.py — look for new parameters, env vars, thresholds
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        print(f"\n[SRC] fused_moe_2stages ({len(src)} chars):", file=sys.stderr)
        # Print first 200 lines (the important signature and logic)
        lines = src.split('\n')
        for i, line in enumerate(lines[:200]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
        if len(lines) > 200:
            print(f"  ... ({len(lines)-200} more lines)", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] fused_moe_2stages error: {e}", file=sys.stderr)

    # 2. Dump fused_moe top-level function
    try:
        src = inspect.getsource(fm.fused_moe)
        print(f"\n[SRC] fused_moe ({len(src)} chars):", file=sys.stderr)
        for i, line in enumerate(src.split('\n')[:80]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] fused_moe error: {e}", file=sys.stderr)

    # 3. Dump get_2stage_cfgs
    try:
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        print(f"\n[SRC] get_2stage_cfgs ({len(src)} chars):", file=sys.stderr)
        for i, line in enumerate(src.split('\n')[:150]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] get_2stage_cfgs error: {e}", file=sys.stderr)

    # 4. Check for new module-level variables and functions
    print(f"\n[MOD] fused_moe module attributes:", file=sys.stderr)
    for name in sorted(dir(fm)):
        if name.startswith('_') and not name.startswith('__'):
            val = getattr(fm, name, '?')
            print(f"  {name} = {val} ({type(val).__name__})", file=sys.stderr)

    # 5. Check public attributes
    print(f"\n[MOD] fused_moe public attrs:", file=sys.stderr)
    for name in sorted(dir(fm)):
        if not name.startswith('_'):
            val = getattr(fm, name, '?')
            if not callable(val) or name in ('token_num_quant_moe_sort_switch',):
                print(f"  {name} = {val} ({type(val).__name__})", file=sys.stderr)

    # 6. Dump the fused quant+sort kernel source
    try:
        from aiter.ops.triton import fused_dynamic_mxfp4_quant_moe_sort as fq
        src = inspect.getsource(fq)
        print(f"\n[SRC] fused_dynamic_mxfp4_quant_moe_sort ({len(src)} chars):", file=sys.stderr)
        lines = src.split('\n')
        for i, line in enumerate(lines[:100]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
        if len(lines) > 100:
            print(f"  ... ({len(lines)-100} more lines)", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] fused quant+sort: {e}", file=sys.stderr)

    # 7. Check MOEMetadata class
    try:
        src = inspect.getsource(fm.MOEMetadata)
        print(f"\n[SRC] MOEMetadata:", file=sys.stderr)
        for line in src.split('\n'):
            print(f"  {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] MOEMetadata: {e}", file=sys.stderr)

    # 8. Check for blockPerCu or new parameters in CK stage functions
    try:
        src = inspect.getsource(fm.ck_moe_stage1)
        print(f"\n[SRC] ck_moe_stage1 ({len(src)} chars):", file=sys.stderr)
        for i, line in enumerate(src.split('\n')[:60]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] ck_moe_stage1: {e}", file=sys.stderr)

    try:
        s2_func = getattr(aiter, 'ck_moe_stage2_fwd', None)
        if s2_func:
            src = inspect.getsource(s2_func)
            print(f"\n[SRC] ck_moe_stage2_fwd ({len(src)} chars):", file=sys.stderr)
            for i, line in enumerate(src.split('\n')[:60]):
                print(f"  {i+1:4d}| {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] ck_moe_stage2_fwd: {e}", file=sys.stderr)

    # 9. Read and analyze the tuned CSV for d=2048 / E=33 entries
    import csv
    csv_paths = [
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    ]
    for csv_path in csv_paths:
        try:
            if not os.path.exists(csv_path):
                print(f"\n[CSV] {csv_path}: NOT FOUND", file=sys.stderr)
                continue
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                print(f"\n[CSV] {os.path.basename(csv_path)} headers: {reader.fieldnames}", file=sys.stderr)
                relevant = []
                all_rows = list(reader)
                for row in all_rows:
                    # Collect: E=33, or inter_dim>=2048, or cu_num=256
                    e = row.get('expert', row.get('num_expert', ''))
                    d = row.get('inter_dim', row.get('intermediate_size', ''))
                    cu = row.get('cu_num', '')
                    if e in ('32', '33') or (d and int(d) >= 2048) or cu == '256':
                        relevant.append(row)
                print(f"[CSV] Relevant rows (E=32/33 or d>=2048 or cu=256): {len(relevant)} / {len(all_rows)}", file=sys.stderr)
                for r in relevant[:40]:
                    print(f"  {dict(r)}", file=sys.stderr)
                if len(relevant) > 40:
                    print(f"  ... ({len(relevant)-40} more)", file=sys.stderr)
        except Exception as e:
            print(f"[CSV] {csv_path}: {e}", file=sys.stderr)

    # 10. List ALL .co files in fmoe_2stages directory
    co_dirs = [
        "/home/runner/aiter/hsa/gfx950/fmoe_2stages/",
        "/home/runner/aiter/hsa/gfx950/",
    ]
    for co_dir in co_dirs:
        try:
            if not os.path.exists(co_dir):
                continue
            entries = sorted(os.listdir(co_dir))
            # Filter for .co files or directories
            co_files = [e for e in entries if e.endswith('.co')]
            subdirs = [e for e in entries if os.path.isdir(os.path.join(co_dir, e))]
            print(f"\n[CO] {co_dir}: {len(co_files)} .co files, {len(subdirs)} subdirs", file=sys.stderr)
            if subdirs:
                print(f"  Subdirs: {subdirs}", file=sys.stderr)
            # Group .co files by type (stage1 vs stage2)
            s1_cos = [c for c in co_files if 'gemm1' in c or 'stage1' in c]
            s2_cos = [c for c in co_files if 'gemm2' in c or 'stage2' in c]
            other_cos = [c for c in co_files if c not in s1_cos and c not in s2_cos]
            print(f"  Stage1: {len(s1_cos)}, Stage2: {len(s2_cos)}, Other: {len(other_cos)}", file=sys.stderr)
            # Print ALL stage1 and stage2 .co files — we need to find d=2048-compatible ones
            print(f"\n  === STAGE1 .co files ===", file=sys.stderr)
            for c in s1_cos:
                print(f"    {c}", file=sys.stderr)
            print(f"\n  === STAGE2 .co files ===", file=sys.stderr)
            for c in s2_cos:
                print(f"    {c}", file=sys.stderr)
            if other_cos:
                print(f"\n  === OTHER .co files ===", file=sys.stderr)
                for c in other_cos[:20]:
                    print(f"    {c}", file=sys.stderr)
        except Exception as e:
            print(f"[CO] {co_dir}: {e}", file=sys.stderr)

    # 11. Check for new env variables that might affect MoE
    env_keys = [
        'AITER_USE_NT', 'AITER_KSPLIT', 'AITER_CONFIG_FMOE',
        'AITER_USE_OPUS_MOE_SORTING', 'AITER_BYPASS_TUNE_CONFIG',
        'AITER_MOE_BLOCK_M', 'AITER_MOE_SPLITK', 'AITER_BLOCK_PER_CU',
        'AITER_MOE_USE_FUSED_QUANT', 'AITER_MOE_QUANT_THRESHOLD',
        'AITER_DEBUG', 'AITER_VERBOSE', 'AITER_TUNE',
    ]
    print(f"\n[ENV] Relevant environment variables:", file=sys.stderr)
    for k in env_keys:
        v = os.environ.get(k, '<not set>')
        print(f"  {k} = {v}", file=sys.stderr)

    # 12. Check what happens with get_block_size_M for d=2048
    try:
        src = inspect.getsource(fm.get_block_size_M)
        print(f"\n[SRC] get_block_size_M:", file=sys.stderr)
        for line in src.split('\n'):
            print(f"  {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] get_block_size_M: {e}", file=sys.stderr)

    # 13. Check use_nt source
    try:
        src = inspect.getsource(fm.use_nt)
        print(f"\n[SRC] use_nt:", file=sys.stderr)
        for line in src.split('\n'):
            print(f"  {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] use_nt: {e}", file=sys.stderr)

    # 14. Check _moe_sorting_impl for any new parameters
    try:
        src = inspect.getsource(fm._moe_sorting_impl)
        print(f"\n[SRC] _moe_sorting_impl ({len(src)} chars):", file=sys.stderr)
        for i, line in enumerate(src.split('\n')[:60]):
            print(f"  {i+1:4d}| {line}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] _moe_sorting_impl: {e}", file=sys.stderr)

    # 15. Check for Triton autotuned kernels that might have d=2048 configs
    try:
        config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
        if os.path.exists(config_dir):
            configs = sorted(os.listdir(config_dir))
            fp4_configs = [c for c in configs if 'fp4' in c.lower() or 'FP4' in c]
            print(f"\n[CFG] Triton GEMM configs: {len(configs)} total, {len(fp4_configs)} FP4", file=sys.stderr)
            # Look for configs that might match d=2048 shapes
            for c in fp4_configs:
                if '2048' in c or '7168' in c or '4096' in c:
                    print(f"  {c}", file=sys.stderr)
    except Exception as e:
        print(f"[CFG] configs: {e}", file=sys.stderr)

    # 16. Read fused_moe_2stages source for the FULL function (not just signature)
    try:
        import aiter.fused_moe as fm_src
        src_file = inspect.getfile(fm_src)
        print(f"\n[SRC] fused_moe module file: {src_file}", file=sys.stderr)
        with open(src_file) as f:
            content = f.read()
        # Find token_num_quant_moe_sort_switch
        idx = content.find('token_num_quant_moe_sort_switch')
        if idx >= 0:
            ctx = content[max(0, idx-100):idx+200]
            print(f"\n[SRC] token_num_quant_moe_sort_switch context:", file=sys.stderr)
            print(f"  {ctx}", file=sys.stderr)
        # Find any blockPerCu references
        for needle in ['blockPerCu', 'block_per_cu', 'blocks_per_cu', 'occupancy']:
            idx = content.find(needle)
            if idx >= 0:
                ctx = content[max(0, idx-80):idx+120]
                print(f"\n[SRC] '{needle}' found:", file=sys.stderr)
                print(f"  {ctx}", file=sys.stderr)
        # Find splitK / ksplit references
        for needle in ['splitk', 'splitK', 'ksplit', 'KSPLIT', 'split_k']:
            idx = content.find(needle)
            if idx >= 0:
                ctx = content[max(0, idx-60):idx+100]
                print(f"\n[SRC] '{needle}' found:", file=sys.stderr)
                print(f"  {ctx}", file=sys.stderr)
    except Exception as e:
        print(f"[SRC] Full source read: {e}", file=sys.stderr)

    print("\n" + "=" * 80, file=sys.stderr)
    print("=== END SOURCE PROBE ===", file=sys.stderr)
    print("=" * 80, file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _dump_sources()

    # Standard optimizations (proven best)
    fm.use_nt = lambda token, topk, expert: False

    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # EXPERIMENT: Try modifying the quant threshold
    # For d=2048 bs=512: token_num = 512*9 = 4608 > 1024 (separate path)
    # For d=2048 bs=128: token_num = 128*9 = 1152 > 1024 (separate path)
    # For d=2048 bs=16:  token_num = 16*9 = 144 < 1024 (fused path)
    # Hypothesis: fused path might be faster for d=2048 even at higher token counts
    try:
        if hasattr(fm, 'token_num_quant_moe_sort_switch'):
            old_val = fm.token_num_quant_moe_sort_switch
            fm.token_num_quant_moe_sort_switch = 8192  # Force fused for ALL
            print(f"[THRESH] token_num_quant_moe_sort_switch: {old_val} -> 8192", file=sys.stderr)
        else:
            print(f"[THRESH] No direct attr, checking in fused_moe_2stages...", file=sys.stderr)
            # Try to find and patch it in the function's globals
            if hasattr(fm, 'fused_moe_2stages'):
                g = fm.fused_moe_2stages.__globals__
                if 'token_num_quant_moe_sort_switch' in g:
                    old = g['token_num_quant_moe_sort_switch']
                    g['token_num_quant_moe_sort_switch'] = 8192
                    print(f"[THRESH] Patched via globals: {old} -> 8192", file=sys.stderr)
                else:
                    # List all numeric globals that might be thresholds
                    thresholds = {k: v for k, v in g.items()
                                  if isinstance(v, (int, float)) and not k.startswith('__')}
                    print(f"[THRESH] Numeric globals: {thresholds}", file=sys.stderr)
    except Exception as e:
        print(f"[THRESH] Error: {e}", file=sys.stderr)

    # CK injection for E<=64 d<2048 only (d>=2048 uses auto for now)
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

        # Log ALL d=2048 config details
        if inter_dim >= 2048:
            est_m = token * topk // expert if expert > 0 else 0
            print(f"\n[D2048] token={token} dim={model_dim} inter={inter_dim} E={expert} "
                  f"topk={topk} est_m={est_m}", file=sys.stderr)
            print(f"  run_1stage={result.run_1stage} block_m={result.block_m}", file=sys.stderr)
            if not result.run_1stage:
                s1kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                s2kw = result.stage2.keywords if hasattr(result.stage2, 'keywords') else {}
                print(f"  stage1: {s1kw.get('kernelName', 'AUTO')}", file=sys.stderr)
                print(f"  stage2: {s2kw.get('kernelName', 'AUTO')}", file=sys.stderr)
                # Dump ALL stage1 kwargs
                print(f"  s1 full kwargs: {dict(s1kw)}", file=sys.stderr)
                print(f"  s2 full kwargs: {dict(s2kw)}", file=sys.stderr)
                # Check the partial function itself
                print(f"  s1 func: {result.stage1.func.__name__ if hasattr(result.stage1, 'func') else '?'}", file=sys.stderr)
                print(f"  s2 func: {result.stage2.func.__name__ if hasattr(result.stage2, 'func') else '?'}", file=sys.stderr)
                # Check for any extra fields in MOEMetadata
                for attr in dir(result):
                    if not attr.startswith('_'):
                        print(f"  metadata.{attr} = {getattr(result, attr)}", file=sys.stderr)

        # Inject CK kernels for E<=64 d<2048 (proven path)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
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
    dh = config.get("d_hidden", 0)
    shape_key = f"bs{bs}_E{E}_d{d}"

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Time each call for shape-level profiling
    if _call_count <= 100:
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

    if _call_count <= 100:
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1e6
        if shape_key not in _timings:
            _timings[shape_key] = []
        _timings[shape_key].append(elapsed_us)
        print(f"[TIME] call={_call_count} {shape_key}: {elapsed_us:.1f}us", file=sys.stderr)

    # Print summary at call 100
    if _call_count == 100:
        print(f"\n[SUMMARY] Per-shape timing (first 100 calls):", file=sys.stderr)
        for key, times in sorted(_timings.items()):
            times_skip = times[2:] if len(times) > 3 else times  # skip warmup
            if times_skip:
                avg = sum(times_skip) / len(times_skip)
                mn = min(times_skip)
                mx = max(times_skip)
                print(f"  {key}: avg={avg:.1f}us min={mn:.1f}us max={mx:.1f}us n={len(times_skip)}", file=sys.stderr)

    return result

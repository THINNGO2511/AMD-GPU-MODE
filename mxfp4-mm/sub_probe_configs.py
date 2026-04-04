from task import input_t, output_t
import torch
import time
import os
import sys
import json
import glob
import inspect
import functools

# ============================================================================
# GEMM Config Probe — Systematic analysis of config selection per benchmark shape
#
# Goals:
# 1. List ALL config files at /home/runner/aiter/aiter/ops/triton/configs/gemm/
# 2. Check which config files match our exact N,K values
# 3. Dump the ACTUAL config selected by gemm_a16wfp4 for each benchmark shape
# 4. Dump the ACTUAL config selected by gemm_afp4wfp4 for each shape
# 5. Read the gemm_a16wfp4 wrapper source (config= parameter handling)
# 6. Time each shape with default config, then with each available tuned config
# 7. Probe the _get_config function internals
# ============================================================================

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False
_call_count = 0

# Benchmark shapes from task.yml
BENCHMARK_SHAPES = [
    (4, 2880, 512),
    (16, 2112, 7168),
    (32, 4096, 512),
    (32, 2880, 512),
    (64, 7168, 2048),
    (256, 3072, 1536),
]

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe_configs():
    """One-time probe of all config files and selection logic."""
    global _probed
    if _probed:
        return
    _probed = True

    cfg_base = "/home/runner/aiter/aiter/ops/triton/configs/gemm"

    # ---- SECTION 1: List ALL config files ----
    print("=" * 70)
    print("SECTION 1: ALL CONFIG FILES IN GEMM CONFIG DIRECTORY")
    print("=" * 70)
    all_files = []
    if os.path.isdir(cfg_base):
        all_files = sorted(os.listdir(cfg_base))
        print(f"Total files: {len(all_files)}")
        for f in all_files:
            full = os.path.join(cfg_base, f)
            sz = os.path.getsize(full) if os.path.isfile(full) else -1
            print(f"  {f}  ({sz} bytes)")
    else:
        print(f"  DIR NOT FOUND: {cfg_base}")

    # ---- SECTION 2: Filter for FP4 / A16WFP4 / gfx950 configs ----
    print("\n" + "=" * 70)
    print("SECTION 2: FP4 CONFIG FILES (matching our kernel types)")
    print("=" * 70)
    fp4_files = [f for f in all_files if 'fp4' in f.lower() or 'FP4' in f]
    a16w_files = [f for f in all_files if 'A16WFP4' in f]
    afp4_files = [f for f in all_files if 'AFP4WFP4' in f and 'A16' not in f]
    gfx950_files = [f for f in all_files if 'gfx950' in f]
    print(f"FP4-related: {len(fp4_files)}")
    print(f"A16WFP4: {len(a16w_files)}")
    print(f"AFP4WFP4 (not A16): {len(afp4_files)}")
    print(f"gfx950: {len(gfx950_files)}")

    # ---- SECTION 3: Check for configs matching our EXACT N,K values ----
    print("\n" + "=" * 70)
    print("SECTION 3: CONFIG FILES MATCHING OUR BENCHMARK N,K VALUES")
    print("=" * 70)
    our_nk_pairs = [(2880, 512), (2112, 7168), (4096, 512), (7168, 2048), (3072, 1536)]
    for n_val, k_val in our_nk_pairs:
        # Try multiple naming patterns
        patterns = [
            f"gfx950-GEMM-A16WFP4-N={n_val}-K={k_val}.json",
            f"gfx950-GEMM-AFP4WFP4-N={n_val}-K={k_val}.json",
            f"gfx950-GEMM-A16WFP4_PRESHUFFLED-N={n_val}-K={k_val}.json",
            f"gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N={n_val}-K={k_val}.json",
        ]
        found_any = False
        for pat in patterns:
            fpath = os.path.join(cfg_base, pat)
            if os.path.exists(fpath):
                found_any = True
                print(f"  FOUND: {pat}")
                try:
                    with open(fpath) as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        for mk, cfg in sorted(data.items()):
                            print(f"    M_key={mk}: {json.dumps(cfg)}")
                    else:
                        print(f"    {json.dumps(data)[:500]}")
                except Exception as e:
                    print(f"    READ ERROR: {e}")
        if not found_any:
            print(f"  MISSING: N={n_val} K={k_val} (no config file)")

    # ---- SECTION 4: Read ALL gfx950 A16WFP4 config files (full content) ----
    print("\n" + "=" * 70)
    print("SECTION 4: FULL CONTENT OF ALL A16WFP4 CONFIG FILES")
    print("=" * 70)
    for f in sorted(all_files):
        if 'A16WFP4' in f and f.endswith('.json'):
            fpath = os.path.join(cfg_base, f)
            try:
                with open(fpath) as fh:
                    data = json.load(fh)
                print(f"\n  === {f} ===")
                if isinstance(data, dict):
                    for mk, cfg in sorted(data.items(), key=lambda x: str(x[0])):
                        print(f"    {mk}: {json.dumps(cfg)}")
                elif isinstance(data, list):
                    for i, entry in enumerate(data[:10]):
                        print(f"    [{i}]: {json.dumps(entry)}")
                else:
                    print(f"    {json.dumps(data)[:800]}")
            except Exception as e:
                print(f"  === {f} === ERROR: {e}")

    # ---- SECTION 5: Read ALL gfx950 AFP4WFP4 config files (just N,K from name) ----
    print("\n" + "=" * 70)
    print("SECTION 5: ALL AFP4WFP4 CONFIG FILE NAMES + N,K PAIRS")
    print("=" * 70)
    afp4_nk_available = set()
    for f in sorted(all_files):
        if 'AFP4WFP4' in f and f.endswith('.json') and 'A16' not in f:
            # Parse N and K from filename
            parts = f.replace('.json', '').split('-')
            n_part = [p for p in parts if p.startswith('N=')]
            k_part = [p for p in parts if p.startswith('K=')]
            n_val = n_part[0].split('=')[1] if n_part else '?'
            k_val = k_part[0].split('=')[1] if k_part else '?'
            afp4_nk_available.add((n_val, k_val))
            print(f"  {f}  (N={n_val}, K={k_val})")
    print(f"\n  Available N,K pairs: {sorted(afp4_nk_available)}")

    # ---- SECTION 6: Probe gemm_a16wfp4 source — config loading logic ----
    print("\n" + "=" * 70)
    print("SECTION 6: GEMM_A16WFP4 WRAPPER SOURCE (config handling)")
    print("=" * 70)
    try:
        from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as mod
        src = inspect.getsource(mod)
        lines = src.split('\n')
        print(f"Total source lines: {len(lines)}")

        # Print the wrapper function that handles config= parameter
        in_wrapper = False
        indent_level = None
        for i, line in enumerate(lines):
            # Find the public gemm_a16wfp4 function (not the Triton kernel)
            if 'def gemm_a16wfp4' in line and '@triton' not in lines[max(0,i-1)]:
                in_wrapper = True
                indent_level = len(line) - len(line.lstrip())
            if in_wrapper:
                print(f"  {i+1}: {line}")
                # End when we hit next top-level def/class
                if i > 0 and in_wrapper and line.strip() and not line[0].isspace() and 'def ' in line:
                    break
                if len(line) - len(line.lstrip()) <= indent_level and line.strip().startswith('def ') and i > 0:
                    if 'def gemm_a16wfp4' not in line:
                        break

        # Also find _get_config or config loading
        print("\n  --- Config-related lines ---")
        for i, line in enumerate(lines):
            ll = line.lower()
            if any(kw in ll for kw in ['_get_config', 'config_dir', 'json', 'config_path',
                                        'load_config', 'read_config', 'configs_dir']):
                print(f"  {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR reading source: {e}")

    # ---- SECTION 7: Probe _get_config function directly ----
    print("\n" + "=" * 70)
    print("SECTION 7: _get_config FUNCTION (if accessible)")
    print("=" * 70)
    # Try multiple module paths
    config_modules = [
        "aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4",
        "aiter.ops.triton.gemm.basic.gemm_a16wfp4",
        "aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4",
        "aiter.ops.triton.gemm.basic.gemm_afp4wfp4",
    ]
    for modname in config_modules:
        try:
            mod = __import__(modname, fromlist=['_get_config'])
            if hasattr(mod, '_get_config'):
                gc = mod._get_config
                print(f"  Found _get_config in {modname}")
                print(f"  Type: {type(gc)}")
                print(f"  Has __wrapped__: {hasattr(gc, '__wrapped__')}")
                # Try to get source
                try:
                    src = inspect.getsource(gc if not hasattr(gc, '__wrapped__') else gc.__wrapped__)
                    for line in src.split('\n')[:40]:
                        print(f"    {line}")
                except Exception as e2:
                    print(f"    Source error: {e2}")
                # Call it for each benchmark shape
                print(f"\n  Config results for benchmark shapes:")
                for m_val, n_val, k_val in BENCHMARK_SHAPES:
                    try:
                        result = gc(m_val, n_val, k_val)
                        print(f"    M={m_val:3d} N={n_val:4d} K={k_val:4d}: {result}")
                    except Exception as e3:
                        print(f"    M={m_val:3d} N={n_val:4d} K={k_val:4d}: ERROR {e3}")
        except Exception as e:
            pass  # silently skip non-existent modules

    # ---- SECTION 8: Check for _get_a16wfp4_config or similar ----
    print("\n" + "=" * 70)
    print("SECTION 8: ALL FUNCTIONS IN GEMM MODULES")
    print("=" * 70)
    for modname in ["aiter.ops.triton.gemm.basic.gemm_a16wfp4",
                    "aiter.ops.triton.gemm.basic.gemm_afp4wfp4"]:
        try:
            mod = __import__(modname, fromlist=['__all__'])
            funcs = [x for x in dir(mod) if not x.startswith('__')]
            print(f"  {modname.split('.')[-1]}: {funcs}")
        except Exception as e:
            print(f"  {modname.split('.')[-1]}: import error: {e}")

    # ---- SECTION 9: Check how config is consumed (kernel decorator) ----
    print("\n" + "=" * 70)
    print("SECTION 9: TRITON KERNEL DECORATOR (autotune configs)")
    print("=" * 70)
    try:
        from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as mod
        src = inspect.getsource(mod)
        lines = src.split('\n')
        # Find @triton.autotune or @triton.jit sections
        for i, line in enumerate(lines):
            if '@triton' in line or 'autotune' in line.lower():
                context = lines[max(0, i-1):min(len(lines), i+10)]
                for j, cl in enumerate(context):
                    print(f"  {i+j}: {cl}")
                print("  ...")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ---- SECTION 10: Check config file naming pattern ----
    print("\n" + "=" * 70)
    print("SECTION 10: CONFIG FILE NAMING PATTERN ANALYSIS")
    print("=" * 70)
    # Extract unique N and K values from config filenames
    n_values = set()
    k_values = set()
    for f in all_files:
        if f.endswith('.json'):
            parts = f.replace('.json', '').split('-')
            for p in parts:
                if p.startswith('N='):
                    try:
                        n_values.add(int(p[2:]))
                    except ValueError:
                        pass
                if p.startswith('K='):
                    try:
                        k_values.add(int(p[2:]))
                    except ValueError:
                        pass
    print(f"  N values in configs: {sorted(n_values)}")
    print(f"  K values in configs: {sorted(k_values)}")
    print(f"  Our N values: [2112, 2880, 3072, 4096, 7168]")
    print(f"  Our K values: [512, 1536, 2048, 7168]")
    our_n = {2112, 2880, 3072, 4096, 7168}
    our_k = {512, 1536, 2048, 7168}
    print(f"  N values MISSING from configs: {sorted(our_n - n_values)}")
    print(f"  K values MISSING from configs: {sorted(our_k - k_values)}")
    print(f"  N values in configs but NOT ours: {sorted(n_values - our_n)}")

    # ---- SECTION 11: Triton version and cache dir ----
    print("\n" + "=" * 70)
    print("SECTION 11: TRITON VERSION + CACHE")
    print("=" * 70)
    try:
        import triton
        print(f"  Triton version: {triton.__version__}")
        cache_dir = os.environ.get('TRITON_CACHE_DIR', '~/.triton/cache')
        print(f"  TRITON_CACHE_DIR: {cache_dir}")
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 70)
    print("CONFIG PROBE COMPLETE")
    print("=" * 70)


def _time_kernel(fn, warmup=3, iters=20):
    """Time a function using CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us
    times.sort()
    # Trim outliers: use median of middle 60%
    trim = max(1, len(times) // 5)
    trimmed = times[trim:-trim] if trim < len(times) // 2 else times
    avg = sum(trimmed) / len(trimmed)
    return avg, min(times), max(times)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8, _call_count

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _call_count += 1

    # Run config probe once on first call
    if _call_count == 1:
        _probe_configs()

    # Per-shape timing and config report on first few calls
    if _call_count <= 12:
        shape_key = (m, n, k)
        print(f"\n--- CALL {_call_count}: M={m} N={n} K={k} ---")

        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

        # Time with default config (config=None)
        def run_default():
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)

        avg_d, min_d, max_d = _time_kernel(run_default)
        print(f"  DEFAULT: avg={avg_d:.1f}us min={min_d:.1f}us max={max_d:.1f}us")

        # Time with our K=7168 custom config
        if k == 7168:
            cfg_k7 = {
                "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
                "cache_modifier": None, "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
            }
            def run_k7():
                return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg_k7)
            avg_7, min_7, max_7 = _time_kernel(run_k7)
            print(f"  K7168_CFG (BM=8 BN=64 KS=8): avg={avg_7:.1f}us min={min_7:.1f}us")

            # Also try BM=16 (session 15 found slightly better)
            cfg_k7_bm16 = dict(cfg_k7)
            cfg_k7_bm16["BLOCK_SIZE_M"] = 16
            def run_k7_bm16():
                return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg_k7_bm16)
            avg_bm16, _, _ = _time_kernel(run_k7_bm16)
            print(f"  K7168_BM16: avg={avg_bm16:.1f}us")

        # Time with .cg cache modifier
        cg_cfg = {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
        }
        def run_cg():
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cg_cfg)
        try:
            avg_cg, _, _ = _time_kernel(run_cg)
            print(f"  CG_CACHE (BM=16 BN=128 KS=1 .cg): avg={avg_cg:.1f}us")
        except Exception as e:
            print(f"  CG_CACHE: ERROR {type(e).__name__}: {str(e)[:100]}")

        # Try stages=3
        stg3_cfg = {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
            "cache_modifier": None, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
        }
        def run_stg3():
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=stg3_cfg)
        try:
            avg_s3, _, _ = _time_kernel(run_stg3)
            print(f"  STAGES3 (BM=16 BN=128 KS=1): avg={avg_s3:.1f}us")
        except Exception as e:
            print(f"  STAGES3: ERROR {type(e).__name__}: {str(e)[:100]}")

        # For K=1536, also test afp4wfp4 path
        if k == 1536:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            def run_afp4():
                A_fp4, A_scale = dynamic_mxfp4_quant(A)
                return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                                     dtype=torch.bfloat16)
            avg_af, _, _ = _time_kernel(run_afp4)
            print(f"  AFP4WFP4 (quant+gemm): avg={avg_af:.1f}us")

        # Try wpe=4 (waves_per_eu=4)
        wpe4_cfg = {
            "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
            "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
            "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
        }
        def run_wpe4():
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=wpe4_cfg)
        try:
            avg_w4, _, _ = _time_kernel(run_wpe4)
            print(f"  WPE4_CG (BM=16 BN=128 W=8 .cg wpe=4): avg={avg_w4:.1f}us")
        except Exception as e:
            print(f"  WPE4_CG: ERROR {type(e).__name__}: {str(e)[:100]}")

        # Try split-K for K>=1536
        if k >= 1536:
            ksplit_val = max(1, k // 512)  # target 512 per split
            sk_cfg = {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
                "cache_modifier": None, "NUM_KSPLIT": ksplit_val,
                "SPLITK_BLOCK_SIZE": max(1024, k // ksplit_val * 2),
            }
            def run_sk():
                return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=sk_cfg)
            try:
                avg_sk, _, _ = _time_kernel(run_sk)
                print(f"  SPLITK={ksplit_val} (BM=16 BN=128): avg={avg_sk:.1f}us")
            except Exception as e:
                print(f"  SPLITK={ksplit_val}: ERROR {type(e).__name__}: {str(e)[:100]}")

        sys.stdout.flush()

    # Production path: return correct result
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    cfg = None
    if k == 7168:
        cfg = {
            "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
            "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
            "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
        }
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)

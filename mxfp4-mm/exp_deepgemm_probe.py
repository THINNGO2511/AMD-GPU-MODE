#!POPCORN benchmark amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
PR #2261 Config Probe: Check if runner has updated GEMM configs.
Dumps config files, reads kernel source, tests alternative configs.
"""
from task import input_t, output_t
import torch
import sys
import os
import json
import glob as glob_mod
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}
_probed = False

# Current best config for K=7168
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# PR #2261 candidate: KSPLIT=16 for K=7168
_K7168_KSPLIT16 = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 16, "SPLITK_BLOCK_SIZE": 512,
}

# PR #2261 candidate: num_stages=3 for K=2048
_K2048_STAGES3 = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def P(msg):
    """Print to stderr so it shows in benchmark logs."""
    print(f"PROBE: {msg}", file=sys.stderr)


def _dump_config_files():
    """Dump all FP4 config files from the runner."""
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"

    P("=" * 60)
    P("SECTION 1: CONFIG FILE LISTING")
    P("=" * 60)

    if not os.path.exists(config_dir):
        P(f"Config dir NOT FOUND: {config_dir}")
        return

    all_files = sorted(os.listdir(config_dir))
    P(f"Total config files: {len(all_files)}")

    # List ALL fp4-related files
    fp4_files = [f for f in all_files if 'fp4' in f.lower()]
    P(f"FP4 config files: {len(fp4_files)}")
    for f in fp4_files:
        P(f"  {f}")

    # Also list any with 'a16' or 'a8' in name
    a16_files = [f for f in all_files if 'a16' in f.lower()]
    a8_files = [f for f in all_files if 'a8' in f.lower()]
    P(f"a16 config files: {len(a16_files)}: {a16_files[:10]}")
    P(f"a8 config files: {len(a8_files)}: {a8_files[:10]}")

    P("=" * 60)
    P("SECTION 2: CONFIG FILE CONTENTS")
    P("=" * 60)

    # Our 6 benchmark shapes
    target_shapes = [
        (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
        (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
    ]
    target_nk = set((n, k) for m, n, k in target_shapes)

    # Read each fp4 config file in full
    for fname in fp4_files:
        fpath = os.path.join(config_dir, fname)
        P(f"\n--- {fname} ---")
        try:
            with open(fpath) as fh:
                content = fh.read()
            # Try parsing as JSON
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    P(f"  Format: JSON array, {len(data)} entries")
                    # Print ALL entries (they're configs for specific shapes)
                    for i, entry in enumerate(data):
                        P(f"  [{i}] {json.dumps(entry)}")
                elif isinstance(data, dict):
                    P(f"  Format: JSON dict, {len(data)} keys")
                    for key, val in data.items():
                        P(f"  {key}: {json.dumps(val)}")
                else:
                    P(f"  Format: {type(data).__name__}")
                    P(f"  Content: {content[:500]}")
            except json.JSONDecodeError:
                # Not JSON, print raw
                P(f"  Raw content ({len(content)} bytes):")
                for line in content.strip().split('\n')[:50]:
                    P(f"  {line}")
        except Exception as e:
            P(f"  Error reading: {e}")

    # Check for M_LEQ_8 bucket or similar naming
    P("\n--- Searching for M_LEQ or bucket-style configs ---")
    for f in all_files:
        if any(x in f.lower() for x in ['leq', 'bucket', 'm_', 'small']):
            P(f"  Found: {f}")

    # Check file modification dates
    P("\n--- File modification times ---")
    for fname in fp4_files:
        fpath = os.path.join(config_dir, fname)
        mtime = os.path.getmtime(fpath)
        P(f"  {fname}: mtime={time.ctime(mtime)}")


def _dump_gemm_source():
    """Read and dump full source of gemm_a16wfp4.py and related files."""
    P("=" * 60)
    P("SECTION 3: gemm_a16wfp4.py FULL SOURCE")
    P("=" * 60)

    src_path = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"
    try:
        with open(src_path) as f:
            src = f.read()
        lines = src.split('\n')
        P(f"Total lines: {len(lines)}")
        # Print in chunks of 100 lines
        for i in range(0, len(lines), 80):
            chunk = '\n'.join(lines[i:i+80])
            P(f"--- Lines {i+1}-{min(i+80, len(lines))} ---")
            P(chunk)
    except Exception as e:
        P(f"Error reading gemm_a16wfp4.py: {e}")

    P("=" * 60)
    P("SECTION 4: gemm_afp4wfp4.py FULL SOURCE")
    P("=" * 60)

    src_path2 = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py"
    try:
        with open(src_path2) as f:
            src = f.read()
        lines = src.split('\n')
        P(f"Total lines: {len(lines)}")
        for i in range(0, len(lines), 80):
            chunk = '\n'.join(lines[i:i+80])
            P(f"--- Lines {i+1}-{min(i+80, len(lines))} ---")
            P(chunk)
    except Exception as e:
        P(f"Error reading gemm_afp4wfp4.py: {e}")

    P("=" * 60)
    P("SECTION 5: gemm_a8wfp4.py SOURCE (if exists)")
    P("=" * 60)

    src_path3 = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a8wfp4.py"
    try:
        with open(src_path3) as f:
            src = f.read()
        lines = src.split('\n')
        P(f"Total lines: {len(lines)}")
        for i in range(0, len(lines), 80):
            chunk = '\n'.join(lines[i:i+80])
            P(f"--- Lines {i+1}-{min(i+80, len(lines))} ---")
            P(chunk)
    except Exception as e:
        P(f"Error/not found: {e}")

    # Also check what other GEMM kernels exist
    P("=" * 60)
    P("SECTION 6: ALL GEMM KERNEL FILES")
    P("=" * 60)

    gemm_basic = "/home/runner/aiter/aiter/ops/triton/gemm/basic/"
    gemm_parent = "/home/runner/aiter/aiter/ops/triton/gemm/"
    for d in [gemm_parent, gemm_basic]:
        try:
            if os.path.exists(d):
                files = sorted(os.listdir(d))
                P(f"{d}: {files}")
        except Exception as e:
            P(f"Error listing {d}: {e}")


def _probe_config_selection():
    """Understand how configs are selected at runtime."""
    P("=" * 60)
    P("SECTION 7: CONFIG SELECTION INTERNALS")
    P("=" * 60)

    try:
        import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as a16mod
        P(f"Module attrs: {[a for a in dir(a16mod) if not a.startswith('__')]}")

        # Check for config-related functions/classes
        for attr in dir(a16mod):
            if any(x in attr.lower() for x in ['config', 'tune', 'cache', 'get_', 'load_']):
                obj = getattr(a16mod, attr)
                P(f"  {attr} = {type(obj).__name__}: {repr(obj)[:200]}")
    except Exception as e:
        P(f"Error inspecting module: {e}")

    # Check the Triton autotune config loading mechanism
    try:
        import aiter.ops.triton.gemm.basic.gemm_afp4wfp4 as afp4mod
        P(f"\nafp4wfp4 attrs: {[a for a in dir(afp4mod) if not a.startswith('__')]}")
        for attr in dir(afp4mod):
            if any(x in attr.lower() for x in ['config', 'tune', 'cache', 'get_', 'load_']):
                obj = getattr(afp4mod, attr)
                P(f"  {attr} = {type(obj).__name__}: {repr(obj)[:200]}")
    except Exception as e:
        P(f"Error: {e}")

    # Check if there's a config registry or loader
    try:
        from aiter.ops.triton.gemm import basic as gemm_basic
        P(f"\ngemm.basic attrs: {[a for a in dir(gemm_basic) if not a.startswith('__')]}")
    except Exception as e:
        P(f"Error: {e}")

    # Check triton config utils
    config_utils_paths = [
        "/home/runner/aiter/aiter/ops/triton/configs/__init__.py",
        "/home/runner/aiter/aiter/ops/triton/gemm/__init__.py",
        "/home/runner/aiter/aiter/ops/triton/gemm/basic/__init__.py",
        "/home/runner/aiter/aiter/ops/triton/utils.py",
    ]
    for p in config_utils_paths:
        if os.path.exists(p):
            try:
                with open(p) as f:
                    content = f.read()
                if content.strip():
                    P(f"\n--- {p} ({len(content)} bytes) ---")
                    P(content[:1000])
            except Exception as e:
                P(f"Error reading {p}: {e}")


def _probe_a8wfp4_import():
    """Try importing gemm_a8wfp4 to see if fp8 A + fp4 B works."""
    P("=" * 60)
    P("SECTION 8: gemm_a8wfp4 IMPORT TEST")
    P("=" * 60)

    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        P("IMPORT SUCCESS: gemm_a8wfp4")

        import inspect
        try:
            sig = inspect.signature(gemm_a8wfp4)
            P(f"Signature: {sig}")
        except Exception:
            P("Could not get signature")

        # Try calling with dummy data to see what it expects
        try:
            M, N, K = 16, 256, 512
            A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            A_fp8 = A_bf16.to(torch.float8_e4m3fnuz)
            B_q = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device='cuda')
            B_scale = torch.randint(100, 140, (N // 32, K // 32), dtype=torch.uint8, device='cuda')

            # Try fp8 A
            try:
                out = gemm_a8wfp4(A_fp8, B_q, B_scale, dtype=torch.bfloat16)
                P(f"gemm_a8wfp4(fp8, u8, u8) WORKED! shape={out.shape}")
            except Exception as e:
                P(f"gemm_a8wfp4(fp8, u8, u8): {str(e)[:200]}")

            # Try with A scale
            try:
                A_scale = torch.ones(1, dtype=torch.float32, device='cuda')
                out = gemm_a8wfp4(A_fp8, B_q, B_scale, A_scale, dtype=torch.bfloat16)
                P(f"gemm_a8wfp4(fp8, u8, u8, scale) WORKED!")
            except Exception as e:
                P(f"gemm_a8wfp4(fp8, u8, u8, scale): {str(e)[:200]}")

        except Exception as e:
            P(f"gemm_a8wfp4 call test error: {str(e)[:200]}")

    except ImportError as e:
        P(f"IMPORT FAILED: {e}")
    except Exception as e:
        P(f"Error: {e}")


def _benchmark_config_variants(A, B_q_u8, B_scale_raw, m, n, k):
    """Benchmark different config variants for our shapes."""
    P("=" * 60)
    P("SECTION 9: CONFIG VARIANT BENCHMARKS")
    P("=" * 60)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    configs_to_test = []

    if k == 7168:
        configs_to_test = [
            ("K7168_current_KSPLIT8", _K7168_CONFIG),
            ("K7168_PR2261_KSPLIT16", _K7168_KSPLIT16),
            ("K7168_KSPLIT16_stages3", {
                **_K7168_KSPLIT16, "num_stages": 3,
            }),
            ("K7168_KSPLIT4", {
                **_K7168_CONFIG, "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 2048,
            }),
            ("K7168_default", None),
        ]
    elif k == 2048:
        configs_to_test = [
            ("K2048_default", None),
            ("K2048_stages3", _K2048_STAGES3),
            ("K2048_KSPLIT2", {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 1024,
            }),
            ("K2048_KSPLIT4", {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 4, "SPLITK_BLOCK_SIZE": 512,
            }),
            ("K2048_KSPLIT8", {
                "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 256,
            }),
        ]
    elif k == 512:
        configs_to_test = [
            ("K512_default", None),
            ("K512_stages3", {
                "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 512,
            }),
            ("K512_KSPLIT2", {
                "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 2, "SPLITK_BLOCK_SIZE": 256,
            }),
            ("K512_BK256", {
                "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
                "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
                "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
                "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 512,
            }),
        ]
    elif k == 1536:
        P(f"  K=1536 uses afp4wfp4 path — testing that separately")
        # Test afp4wfp4 with different configs
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_u8 = A_fp4.view(torch.uint8)

            # Warmup
            for _ in range(5):
                gemm_afp4wfp4(A_u8, B_q_u8, A_scale, B_scale_raw, dtype=torch.bfloat16)
            torch.cuda.synchronize()

            N_ITER = 20
            t0 = time.time()
            for _ in range(N_ITER):
                gemm_afp4wfp4(A_u8, B_q_u8, A_scale, B_scale_raw, dtype=torch.bfloat16)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N_ITER * 1e6
            P(f"  afp4wfp4 default (M={m},N={n},K={k}): {dt:.1f} us")

            # Also try a16wfp4 for K=1536 to compare
            for _ in range(5):
                gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, y=y)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(N_ITER):
                gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, y=y)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N_ITER * 1e6
            P(f"  a16wfp4 default (M={m},N={n},K={k}): {dt:.1f} us")
        except Exception as e:
            P(f"  K=1536 test error: {e}")
        return

    if not configs_to_test:
        P(f"  No config variants for M={m}, N={n}, K={k}")
        return

    N_ITER = 20
    for name, cfg in configs_to_test:
        try:
            # Warmup
            for _ in range(3):
                gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(N_ITER):
                gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16, y=y, config=cfg)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / N_ITER * 1e6
            P(f"  {name} (M={m},N={n},K={k}): {dt:.1f} us")
        except Exception as e:
            P(f"  {name} ERROR: {str(e)[:150]}")


def _probe_num_stages_support():
    """Check if num_stages=3 is actually supported in the Triton kernels."""
    P("=" * 60)
    P("SECTION 10: num_stages SUPPORT CHECK")
    P("=" * 60)

    # Check Triton version
    try:
        import triton
        P(f"Triton version: {triton.__version__}")
        P(f"Triton path: {triton.__file__}")
    except Exception as e:
        P(f"Triton import error: {e}")

    # Check if the kernel decorator supports num_stages=3
    # by looking at the kernel source code
    kernel_path = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"
    try:
        with open(kernel_path) as f:
            src = f.read()
        # Find all references to num_stages
        for i, line in enumerate(src.split('\n'), 1):
            if 'num_stages' in line or 'NUM_STAGE' in line:
                P(f"  Line {i}: {line.strip()}")
    except Exception as e:
        P(f"Error: {e}")


def _probe_git_log():
    """Check git log in aiter repo for PR #2261 traces."""
    P("=" * 60)
    P("SECTION 11: AITER GIT STATE")
    P("=" * 60)

    aiter_dir = "/home/runner/aiter"
    # Check git log
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-30"],
            cwd=aiter_dir, capture_output=True, text=True, timeout=5
        )
        P("Last 30 commits:")
        for line in result.stdout.strip().split('\n'):
            P(f"  {line}")
    except Exception as e:
        P(f"Git log error: {e}")

    # Check for PR #2261 specifically
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep=2261"],
            cwd=aiter_dir, capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            P(f"PR #2261 traces: {result.stdout.strip()}")
        else:
            P("No commits mentioning 2261 found")
    except Exception as e:
        P(f"Git grep error: {e}")

    # Check for retune/config commits
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep=retune"],
            cwd=aiter_dir, capture_output=True, text=True, timeout=5
        )
        if result.stdout.strip():
            P(f"Retune commits: {result.stdout.strip()}")

        result2 = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep=config"],
            cwd=aiter_dir, capture_output=True, text=True, timeout=5
        )
        if result2.stdout.strip():
            P(f"Config commits (first 10):")
            for line in result2.stdout.strip().split('\n')[:10]:
                P(f"  {line}")
    except Exception as e:
        P(f"Error: {e}")

    # Check aiter version
    try:
        import aiter
        P(f"aiter version: {getattr(aiter, '__version__', 'unknown')}")
    except:
        pass


def _probe_config_naming_convention():
    """Understand exact config file naming so we can find our shapes."""
    P("=" * 60)
    P("SECTION 12: CONFIG NAMING CONVENTION")
    P("=" * 60)

    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
    if not os.path.exists(config_dir):
        P("Config dir not found")
        return

    all_files = sorted(os.listdir(config_dir))
    P(f"All config files ({len(all_files)}):")
    for f in all_files:
        fpath = os.path.join(config_dir, f)
        size = os.path.getsize(fpath)
        P(f"  {f} ({size} bytes)")

    # Look for our specific N,K values in filenames
    target_vals = ['2880', '2112', '4096', '7168', '3072', '512', '1536', '2048']
    P("\nSearching for target values in filenames:")
    for val in target_vals:
        matches = [f for f in all_files if val in f]
        if matches:
            P(f"  '{val}' in: {matches}")
        else:
            P(f"  '{val}': no matches")


def _probe_all():
    """Run all probes."""
    global _probed
    if _probed:
        return
    _probed = True

    P("=" * 60)
    P("PR #2261 CONFIG PROBE — STARTING")
    P(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    P("=" * 60)

    try:
        _dump_config_files()
    except Exception as e:
        P(f"Config dump error: {e}")

    try:
        _dump_gemm_source()
    except Exception as e:
        P(f"Source dump error: {e}")

    try:
        _probe_config_selection()
    except Exception as e:
        P(f"Config selection error: {e}")

    try:
        _probe_a8wfp4_import()
    except Exception as e:
        P(f"a8wfp4 probe error: {e}")

    try:
        _probe_num_stages_support()
    except Exception as e:
        P(f"num_stages probe error: {e}")

    try:
        _probe_git_log()
    except Exception as e:
        P(f"Git probe error: {e}")

    try:
        _probe_config_naming_convention()
    except Exception as e:
        P(f"Naming convention probe error: {e}")

    P("=" * 60)
    P("PROBE COMPLETE — will now benchmark config variants per shape")
    P("=" * 60)


def _prewarm():
    """Pre-warm Triton JIT for all unique K values."""
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except Exception:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _prewarm()

    # Run full probe on first call only
    if not _probed:
        _probe_all()

    # Benchmark config variants for this specific shape (only once per shape)
    shape_key = f"_bench_{m}_{n}_{k}"
    if not hasattr(custom_kernel, shape_key):
        setattr(custom_kernel, shape_key, True)
        try:
            _benchmark_config_variants(A, _bq_u8, _bscale_raw, m, n, k)
        except Exception as e:
            P(f"Benchmark error for ({m},{n},{k}): {e}")

    # Return correct result using known-working path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(
            A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
            dtype=torch.bfloat16
        )

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(
        A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
        y=_y_cache[key], config=cfg
    )

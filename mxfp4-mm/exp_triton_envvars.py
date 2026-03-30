#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Source Dump: Read and dump full source of all GEMM-related aiter files.
Goal: find hidden optimization paths, config params, env vars, autotune logic.
"""
import os
import sys
import torch
from task import input_t, output_t

# ─── Source dump helpers ───
def _dump_file(path, label):
    """Read and print a file's full contents."""
    try:
        if os.path.isfile(path):
            with open(path) as f:
                src = f.read()
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DUMP] {label}: {path} ({len(src)} bytes, {src.count(chr(10))} lines)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            # Split into chunks to avoid truncation
            lines = src.split('\n')
            chunk_size = 100
            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                print(chunk, file=sys.stderr)
            return src
        else:
            print(f"[DUMP] {label}: FILE NOT FOUND: {path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[DUMP] {label}: ERROR: {e}", file=sys.stderr)
        return None


def _list_dir(path, label):
    """List all files in a directory recursively."""
    try:
        if os.path.isdir(path):
            entries = []
            for root, dirs, files in os.walk(path):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    sz = os.path.getsize(fp)
                    rel = os.path.relpath(fp, path)
                    entries.append((rel, sz))
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DIR] {label}: {path} ({len(entries)} files)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            for rel, sz in sorted(entries):
                print(f"  {sz:>8} {rel}", file=sys.stderr)
        else:
            print(f"[DIR] {label}: NOT FOUND: {path}", file=sys.stderr)
    except Exception as e:
        print(f"[DIR] {label}: ERROR: {e}", file=sys.stderr)


_dumped = False
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _do_dump():
    """Dump all GEMM-related source files."""
    global _dumped
    if _dumped:
        return
    _dumped = True

    # 1. Primary GEMM kernel files
    _dump_file("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py",
               "gemm_a16wfp4 (main GEMM kernel)")
    _dump_file("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py",
               "gemm_afp4wfp4 (K=1536 path)")
    _dump_file("/home/runner/aiter/aiter/ops/triton/quant.py",
               "quant.py (quantization)")

    # 2. List ALL files in GEMM directories
    _list_dir("/home/runner/aiter/aiter/ops/triton/gemm/basic/",
              "gemm/basic/ directory")
    _list_dir("/home/runner/aiter/aiter/ops/triton/gemm/",
              "gemm/ full directory tree")

    # 3. Check for other GEMM-related files we might have missed
    _list_dir("/home/runner/aiter/aiter/ops/triton/",
              "ops/triton/ (all Triton ops)")

    # 4. Dump any other gemm files found
    for name in ["gemm_a16w4.py", "gemm_a16w8.py", "gemm_a16wfp4_preshuffle.py",
                 "gemm_a8w4.py", "gemm_a8w8.py", "gemm_splitk.py",
                 "matmul_kernel.py", "matmul.py", "common.py", "utils.py",
                 "__init__.py"]:
        path = f"/home/runner/aiter/aiter/ops/triton/gemm/basic/{name}"
        if os.path.isfile(path):
            _dump_file(path, f"gemm/basic/{name}")

    # 5. Check parent gemm/ __init__.py for imports
    for name in ["__init__.py", "common.py", "utils.py"]:
        path = f"/home/runner/aiter/aiter/ops/triton/gemm/{name}"
        if os.path.isfile(path):
            _dump_file(path, f"gemm/{name}")

    # 6. Config files for GEMM tuning
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
    if os.path.isdir(config_dir):
        configs = sorted(os.listdir(config_dir))
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"[CONFIG] GEMM configs: {len(configs)} files", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        for c in configs[:50]:
            print(f"  {c}", file=sys.stderr)
        if len(configs) > 50:
            print(f"  ... {len(configs)-50} more", file=sys.stderr)
        # Dump a few FP4-related configs
        fp4_configs = [c for c in configs if 'fp4' in c.lower() or 'gfx950' in c.lower()]
        print(f"\n[CONFIG] FP4/gfx950 configs: {len(fp4_configs)}", file=sys.stderr)
        for c in fp4_configs[:10]:
            cpath = os.path.join(config_dir, c)
            _dump_file(cpath, f"config/{c}")

    # 7. Check for environment variable usage in Triton kernels
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[ENV] Searching for env var usage in Triton GEMM code...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/ops/triton/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'os.environ' in content or 'getenv' in content:
                        print(f"\n[ENV] {fpath}: contains env var usage", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            if 'os.environ' in line or 'getenv' in line:
                                print(f"  L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 8. Look for split-K implementation details
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[SPLITK] Searching for split-K logic...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/ops/triton/gemm/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'KSPLIT' in content or 'split_k' in content or 'splitk' in content.lower():
                        print(f"\n[SPLITK] {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            ll = line.lower()
                            if 'ksplit' in ll or 'split_k' in ll or 'splitk' in ll:
                                print(f"  L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 9. Check aiter top-level for any new GEMM APIs
    try:
        import aiter
        gemm_attrs = [a for a in dir(aiter) if 'gemm' in a.lower() or 'fp4' in a.lower()
                      or 'quant' in a.lower() or 'mm' in a.lower()]
        print(f"\n[AITER] GEMM-related top-level attrs: {gemm_attrs}", file=sys.stderr)
        import inspect
        for attr in gemm_attrs:
            obj = getattr(aiter, attr)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  {attr}{sig}", file=sys.stderr)
                except:
                    print(f"  {attr} (no sig)", file=sys.stderr)
    except Exception as e:
        print(f"[AITER] Error: {e}", file=sys.stderr)

    # 10. Dump the fp4_utils source
    _dump_file("/home/runner/aiter/aiter/utility/fp4_utils.py", "fp4_utils")

    # 11. Check for autotune configs embedded in kernel
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[AUTOTUNE] Checking autotune decorators...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/ops/triton/gemm/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'autotune' in content or '@triton.autotune' in content:
                        print(f"\n[AUTOTUNE] {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            if 'autotune' in line.lower() or 'config' in line.lower():
                                print(f"  L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 12. Check for any CK GEMM ASM kernel CSV
    _dump_file("/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
               "dsv3_fp4_tuned_fmoe.csv (first 50 lines)")

    # 13. List CK ASM binary files
    _list_dir("/home/runner/aiter/hsa/gfx950/gemm/", "gemm .co files")
    _list_dir("/home/runner/aiter/hsa/gfx950/", "hsa/gfx950 top-level")


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except:
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

    _do_dump()

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                        y=_y_cache[key], config=cfg)

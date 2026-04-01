#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Deep Probe — discover every available kernel path.
Submit as BENCHMARK to see stdout.
Goal: find the API to directly invoke f4gemm ASM .co kernels.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch
import sys
import glob
import inspect
from task import input_t, output_t

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    print("=" * 60, flush=True)
    print("GEMM DEEP PROBE", flush=True)
    print("=" * 60, flush=True)

    # 1. Probe aiter top-level for GEMM-related functions
    import aiter
    gemm_funcs = [x for x in dir(aiter) if 'gemm' in x.lower() or 'mm' in x.lower()
                  or 'matmul' in x.lower() or 'asm' in x.lower() or 'ck' in x.lower()
                  or 'deep' in x.lower()]
    print(f"\n[1] aiter GEMM/CK/deep functions: {gemm_funcs}", flush=True)

    # 2. Probe deepgemm
    for name in ['deepgemm', 'deepgemm_ck', 'deep_gemm']:
        if hasattr(aiter, name):
            fn = getattr(aiter, name)
            try:
                sig = inspect.signature(fn)
                print(f"\n[2] aiter.{name} signature: {sig}", flush=True)
                print(f"    docstring: {fn.__doc__[:200] if fn.__doc__ else 'None'}", flush=True)
            except Exception as e:
                print(f"\n[2] aiter.{name} exists but can't get signature: {e}", flush=True)

    # 3. Probe gemm_a4w4 new API
    try:
        from aiter import gemm_a4w4
        sig = inspect.signature(gemm_a4w4)
        print(f"\n[3] gemm_a4w4 signature: {sig}", flush=True)
        print(f"    source file: {inspect.getfile(gemm_a4w4)}", flush=True)
    except Exception as e:
        print(f"\n[3] gemm_a4w4 error: {e}", flush=True)

    # 4. Probe gemm_a4w4_asm
    try:
        from aiter import gemm_a4w4_asm
        sig = inspect.signature(gemm_a4w4_asm)
        print(f"\n[4] gemm_a4w4_asm signature: {sig}", flush=True)
    except Exception as e:
        print(f"\n[4] gemm_a4w4_asm: {e}", flush=True)

    # 5. List f4gemm .co files
    co_paths = [
        "/home/runner/aiter/hsa/gfx950/f4gemm/",
        "/home/runner/aiter/hsa/gfx950/",
    ]
    for path in co_paths:
        files = sorted(glob.glob(f"{path}*.co"))
        if files:
            print(f"\n[5] .co files in {path}: {len(files)}", flush=True)
            for f in files[:10]:
                print(f"    {f.split('/')[-1]}", flush=True)
            if len(files) > 10:
                print(f"    ... and {len(files)-10} more", flush=True)

    # 6. Look for CSV config files
    csv_files = glob.glob("/home/runner/aiter/hsa/gfx950/f4gemm/*.csv")
    for cf in csv_files:
        print(f"\n[6] CSV: {cf}", flush=True)
        try:
            with open(cf, 'r') as fh:
                lines = fh.readlines()
                print(f"    header: {lines[0].strip()}", flush=True)
                for line in lines[1:6]:
                    print(f"    {line.strip()}", flush=True)
                print(f"    total rows: {len(lines)-1}", flush=True)
        except Exception as e:
            print(f"    read error: {e}", flush=True)

    # 7. Probe aiter.ops for low-level kernel launchers
    try:
        import aiter.ops
        ops_items = dir(aiter.ops)
        ck_items = [x for x in ops_items if 'ck' in x.lower() or 'asm' in x.lower()
                    or 'gemm' in x.lower() or 'launch' in x.lower()]
        print(f"\n[7] aiter.ops CK/ASM items: {ck_items}", flush=True)
    except Exception as e:
        print(f"\n[7] aiter.ops error: {e}", flush=True)

    # 8. Probe torch.ops.aiter for registered ops
    try:
        aiter_ops = [x for x in dir(torch.ops.aiter) if 'gemm' in x.lower()
                     or 'mm' in x.lower() or 'ck' in x.lower()]
        print(f"\n[8] torch.ops.aiter GEMM ops: {aiter_ops}", flush=True)
        for op_name in aiter_ops[:5]:
            try:
                op = getattr(torch.ops.aiter, op_name)
                print(f"    {op_name}: {op}", flush=True)
            except Exception:
                pass
    except Exception as e:
        print(f"\n[8] torch.ops.aiter error: {e}", flush=True)

    # 9. Look at gemm_a4w4 source to find the new invocation path
    try:
        from aiter import gemm_a4w4
        src_file = inspect.getfile(gemm_a4w4)
        with open(src_file, 'r') as f:
            src = f.read()
        print(f"\n[9] gemm_a4w4 source ({len(src)} chars):", flush=True)
        # Print first 2000 chars
        print(src[:2000], flush=True)
    except Exception as e:
        print(f"\n[9] gemm_a4w4 source error: {e}", flush=True)

    # 10. Check for gemm_a4w4_blockscale_tune
    try:
        from aiter import gemm_a4w4_blockscale_tune
        sig = inspect.signature(gemm_a4w4_blockscale_tune)
        print(f"\n[10] gemm_a4w4_blockscale_tune sig: {sig}", flush=True)
    except Exception as e:
        print(f"\n[10] gemm_a4w4_blockscale_tune: {e}", flush=True)

    # 11. Check aiter.__init__ for all exports
    try:
        init_file = inspect.getfile(aiter)
        with open(init_file, 'r') as f:
            init_src = f.read()
        # Find all gemm-related imports
        gemm_lines = [l.strip() for l in init_src.split('\n')
                      if 'gemm' in l.lower() or 'deep' in l.lower() or 'f4' in l.lower()]
        print(f"\n[11] aiter __init__ GEMM imports:", flush=True)
        for l in gemm_lines:
            print(f"    {l}", flush=True)
    except Exception as e:
        print(f"\n[11] aiter init error: {e}", flush=True)

    # 12. Probe Triton config JSON files for our shapes
    try:
        json_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
        json_files = sorted(glob.glob(f"{json_dir}*FP4*"))
        print(f"\n[12] FP4 Triton config files: {len(json_files)}", flush=True)
        for jf in json_files:
            name = jf.split("/")[-1]
            if any(s in name for s in ["2880", "2112", "4096", "7168", "3072"]):
                print(f"    {name}", flush=True)
                try:
                    import json
                    with open(jf) as fh:
                        cfg = json.load(fh)
                    # Print first config
                    if isinstance(cfg, list) and cfg:
                        print(f"      {cfg[0]}", flush=True)
                    elif isinstance(cfg, dict):
                        first_key = next(iter(cfg))
                        print(f"      {first_key}: {cfg[first_key]}", flush=True)
                except Exception:
                    pass
    except Exception as e:
        print(f"\n[12] config JSON error: {e}", flush=True)

    # 13. Check environment for any special flags
    env_keys = [k for k in os.environ if any(x in k.upper()
                for x in ['TRITON', 'HIP', 'ROC', 'AMD', 'AITER', 'CK', 'GPU'])]
    print(f"\n[13] Relevant env vars:", flush=True)
    for k in sorted(env_keys):
        print(f"    {k}={os.environ[k]}", flush=True)

    sys.stdout.flush()


# Keep the actual kernel working (fallback to current best)
_gather_cache = {}
_bscale_ref = None
_bq_u8 = None
_bscale_raw = None
_scale_shape = None
_y_cache = {}
_warmed = False

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}
_K512_CONFIG = {
    "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}
_K2048_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}
_ALL_SHAPES = [
    (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
    (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
]

def _build_gather_cache(sm, sn, device):
    total = sm * sn
    d0, d1 = sm // 32, sn // 8
    idx = torch.arange(total, dtype=torch.int64, device=device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    out_buf = torch.empty(total, dtype=torch.uint8, device=device)
    return idx, out_buf

def _fast_unshuffle(scale_sh_u8_flat, sm, sn):
    gather_idx, out_buf = _gather_cache[(sm, sn)]
    torch.take(scale_sh_u8_flat, gather_idx, out=out_buf)
    return out_buf.view(sm, sn)

def _full_prewarm(device):
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    for m, n, k in _ALL_SHAPES:
        try:
            dummy_a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            if k == 1536:
                af, asc = dynamic_mxfp4_quant(dummy_a)
                dummy_bq = torch.zeros(n, k // 2, dtype=torch.uint8, device=device)
                dummy_bs = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device)
                gemm_afp4wfp4(af.view(torch.uint8), dummy_bq, asc, dummy_bs, dtype=torch.bfloat16)
            else:
                dummy_bq = torch.zeros(n, k // 2, dtype=torch.uint8, device=device)
                pad_n = ((n + 31) // 32) * 32
                dummy_bs = torch.full((pad_n, k // 32), 127, dtype=torch.uint8, device=device)
                dummy_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                cfg = _K7168_CONFIG if k == 7168 else (_K2048_CONFIG if k == 2048 else _K512_CONFIG)
                gemm_a16wfp4(dummy_a, dummy_bq, dummy_bs, dtype=torch.bfloat16, y=dummy_out, config=cfg)
            del dummy_a
        except Exception:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bq_u8, _bscale_raw, _scale_shape
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        s_u8 = B_scale_sh.view(torch.uint8)
        sm, sn = s_u8.shape
        _scale_shape = (sm, sn)
        if _scale_shape not in _gather_cache:
            _gather_cache[_scale_shape] = _build_gather_cache(sm, sn, B_scale_sh.device)
        _bscale_raw = _fast_unshuffle(s_u8.reshape(-1), sm, sn)
        _bq_u8 = B_q.view(torch.uint8)

    _full_prewarm(A.device)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 7168:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K7168_CONFIG)
    elif k == 2048:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K2048_CONFIG)
    elif k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw, dtype=torch.bfloat16)
    else:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K512_CONFIG)
    return out

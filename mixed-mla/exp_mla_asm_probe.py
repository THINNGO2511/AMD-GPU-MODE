#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Source Dump: Read and dump full MLA source code, ASM kernels, metadata logic.
Goal: find hidden optimization paths — page_size native support, reduce kernel,
      batched decode, Q quant overlap, new parameters, metadata shortcuts.
"""
import os
import sys
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_meta_cache = {}
_alloc_cache = {}
_dumped = False


# ─── Source dump helpers ───
def _dump_file(path, label):
    try:
        if os.path.isfile(path):
            with open(path) as f:
                src = f.read()
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DUMP] {label}: {path} ({len(src)} bytes, {src.count(chr(10))} lines)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
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


def _do_dump():
    global _dumped
    if _dumped:
        return
    _dumped = True

    # 1. Core MLA source
    _dump_file("/home/runner/aiter/aiter/mla.py", "mla.py (MLA decode entry)")

    # 2. MLA C++ backend — check for Python bindings
    try:
        import inspect
        from aiter import mla
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"[API] aiter.mla module inspection", file=sys.stderr)
        print(f"{'='*80}", file=sys.stderr)
        all_attrs = sorted(dir(mla))
        print(f"All attrs: {all_attrs}", file=sys.stderr)
        for attr in all_attrs:
            if attr.startswith('_'):
                continue
            obj = getattr(mla, attr)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    print(f"  {attr}{sig}", file=sys.stderr)
                except:
                    print(f"  {attr} (builtin/no sig)", file=sys.stderr)
            else:
                print(f"  {attr} = {repr(obj)[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"[API] Error: {e}", file=sys.stderr)

    # 3. List ALL MLA .co files
    _list_dir("/home/runner/aiter/hsa/gfx950/mla/", "MLA .co kernels")

    # 4. List PA (page attention) .co files
    _list_dir("/home/runner/aiter/hsa/gfx950/pa/", "PA .co kernels")

    # 5. List FMHA V3 .co files
    _list_dir("/home/runner/aiter/hsa/gfx950/fmha_v3_fwd/", "FMHA V3 FWD .co kernels")

    # 6. Check for reduce kernel source
    _dump_file("/home/runner/aiter/aiter/ops/triton/mla_reduce.py", "mla_reduce.py")
    _dump_file("/home/runner/aiter/aiter/ops/triton/mla.py", "ops/triton/mla.py")

    # 7. Search for ALL MLA-related files
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[SEARCH] All MLA-related files in aiter:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py') and ('mla' in fname.lower() or 'decode' in fname.lower()
                                          or 'page' in fname.lower()):
                fpath = os.path.join(root, fname)
                sz = os.path.getsize(fpath)
                print(f"  {sz:>8} {fpath}", file=sys.stderr)

    # 8. Check metadata module source
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[META] Metadata functions inspection", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        import inspect
        import aiter
        for fn_name in ['get_mla_metadata_info_v1', 'get_mla_metadata_v1',
                        'get_mla_metadata_info_v2', 'get_mla_metadata_v2',
                        'get_mla_metadata_info', 'get_mla_metadata']:
            fn = getattr(aiter, fn_name, None)
            if fn is not None:
                try:
                    sig = inspect.signature(fn)
                    print(f"  {fn_name}{sig}", file=sys.stderr)
                except:
                    print(f"  {fn_name} EXISTS (no sig)", file=sys.stderr)
            else:
                print(f"  {fn_name} NOT FOUND", file=sys.stderr)
    except Exception as e:
        print(f"[META] Error: {e}", file=sys.stderr)

    # 9. Dump all MLA-related Python files found
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in sorted(files):
            if fname.endswith('.py') and 'mla' in fname.lower():
                fpath = os.path.join(root, fname)
                _dump_file(fpath, f"MLA file: {fname}")

    # 10. Check for page_size handling in C++ bindings
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[PAGED] Searching for page_size handling...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'page_size' in content or 'kPageSize' in content or 'page_table' in content:
                        print(f"\n  {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            if 'page_size' in line or 'kPageSize' in line or 'page_table' in line:
                                print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 11. Check for num_kv_splits auto-tuning logic
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[SPLITS] Searching for num_kv_splits logic...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'num_kv_splits' in content or 'kv_split' in content:
                        print(f"\n  {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            if 'num_kv_splits' in line or 'kv_split' in line:
                                print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 12. Check torch.ops.aiter for MLA ops
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[OPS] torch.ops.aiter MLA-related ops:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        ops = [a for a in dir(torch.ops.aiter) if 'mla' in a.lower() or 'decode' in a.lower()
               or 'reduce' in a.lower() or 'stage' in a.lower() or 'meta' in a.lower()]
        print(f"  MLA-related ops: {ops}", file=sys.stderr)
        all_ops = sorted(dir(torch.ops.aiter))
        print(f"  All aiter ops ({len(all_ops)}): {all_ops}", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 13. Check for batched decode path
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[BATCH] Searching for batched decode...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'batch' in content.lower() and ('decode' in content.lower() or 'mla' in content.lower()):
                        has_batch_decode = any('batch_decode' in line.lower() or 'batched_decode' in line.lower()
                                              for line in content.split('\n'))
                        if has_batch_decode:
                            print(f"\n  BATCHED DECODE found: {fpath}", file=sys.stderr)
                            for i, line in enumerate(content.split('\n')):
                                if 'batch_decode' in line.lower() or 'batched_decode' in line.lower():
                                    print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 14. Check aiter.dtypes for available data types
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[DTYPES] aiter.dtypes:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        import aiter
        dt = aiter.dtypes if hasattr(aiter, 'dtypes') else None
        if dt:
            for a in sorted(dir(dt)):
                if not a.startswith('_'):
                    print(f"  {a} = {getattr(dt, a)}", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 15. Check hsa/gfx950 top-level directory structure
    _list_dir("/home/runner/aiter/hsa/gfx950/", "hsa/gfx950 top-level (subdirs)")

    # 16. Check for any new 3-buffer or mxfp4 MLA paths
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[MXFP4-MLA] Searching for MXFP4 MLA kernel paths...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/"):
        for fname in files:
            if ('mxfp4' in fname.lower() or 'fp4' in fname.lower()) and \
               ('mla' in fname.lower() or 'decode' in fname.lower()):
                fpath = os.path.join(root, fname)
                print(f"  {fpath}", file=sys.stderr)

    # 17. Dump any .co file names containing fp4 in MLA dirs
    for subdir in ["mla", "pa", "fmha_v3_fwd"]:
        dpath = f"/home/runner/aiter/hsa/gfx950/{subdir}/"
        if os.path.isdir(dpath):
            for fname in sorted(os.listdir(dpath)):
                if 'fp4' in fname.lower() or 'mxfp4' in fname.lower() or 'f4' in fname.lower():
                    print(f"  FP4 kernel: {subdir}/{fname}", file=sys.stderr)


# ─── Triton Q quantization kernels (proven) ───
@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))


@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, amax_ptr,
                     FP8_MAX: tl.constexpr, N, BLOCK: tl.constexpr):
    amax = tl.load(amax_ptr)
    amax = tl.where(amax < 1e-12, 1e-12, amax)
    scale = amax / FP8_MAX
    if tl.program_id(0) == 0:
        tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


# ─── Metadata builder (proven pg8 approach) ───
def _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr):
    total_kv = batch_size * kv_seq_len
    if page_size == 1:
        num_pages = total_kv
        kv_indptr_pages = kv_indptr
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = seq_lens.to(torch.int32)
    else:
        num_pages = total_kv // page_size
        kv_indptr_pages = kv_indptr // page_size
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % page_size).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)

    kv_gran = max(1, 16 // page_size)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, nq, dtype_q, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size,
        kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=dtype_q,
        dtype_kv=FP8_DTYPE,
    )

    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, page_size)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    _do_dump()

    # Route: pg1+bf16Q for kv<=1024, pg8+fp8Q for kv>=8192
    if kv_seq_len <= 1024:
        page_size = 1
        dtype_q = BF16
        use_fp8_q = False
    else:
        page_size = 8
        dtype_q = FP8_DTYPE
        use_fp8_q = True

    num_kv_splits = 8 if total_kv <= 8192 else 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits, page_size, use_fp8_q)
    if cache_key not in _meta_cache:
        _meta_cache[cache_key] = _build_meta(
            batch_size, kv_seq_len, q_seq_len, nq, nkv,
            num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr)

    (wm, wi, wis, ri, rfm, rpm,
     kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

    if use_fp8_q:
        alloc_key = ("fp8", q.shape[0], nq, dv, dq)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = (
                torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
                torch.zeros(1, dtype=torch.float32, device="cuda"),
                torch.empty(1, dtype=torch.float32, device="cuda"),
                torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
            )
        o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

        mla_decode_fwd(
            q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale_buf, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o
    else:
        alloc_key = ("bf16", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]

        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
        return o

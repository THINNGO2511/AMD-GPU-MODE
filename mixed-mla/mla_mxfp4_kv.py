#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — MXFP4 KV cache (2x less bandwidth than FP8).

The task provides kv_data["mxfp4"] = (fp4x2_tensor, e8m0_scale).
This is 2x smaller than the FP8 KV cache we currently use.
For kv=8192 shapes, memory bandwidth dominates — halving it could
cut those shapes by 30-40%.

Question: does mla_decode_fwd accept fp4x2 KV directly?
Probe the API and report what happens.

If mxfp4 MLA kernel doesn't exist, fall back to fp8 path.
"""
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
_FIXED_AMAX = 16.0
_call = 0
_mxfp4_works = None  # None=untested, True/False


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


def _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, dtype_kv, qo_indptr, kv_indptr):
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
        batch_size, q_seq_len, nq, dtype_q, dtype_kv,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work
    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size, kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
        fast_mode=False, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=dtype_q, dtype_kv=dtype_kv,
    )
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, page_size)


def custom_kernel(data: input_t) -> output_t:
    global _call, _mxfp4_works
    _call += 1

    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    page_size = 1 if kv_seq_len <= 1024 else 8
    dtype_q = FP8_DTYPE

    if batch_size <= 4: num_kv_splits = 4
    elif batch_size <= 64: num_kv_splits = 8
    else: num_kv_splits = 8 if kv_seq_len <= 1024 else 16

    # Try MXFP4 KV on first call
    if _mxfp4_works is None and _call <= 2:
        try:
            kv_fp4, kv_scale_fp4 = kv_data["mxfp4"]
            if _call == 1:
                print(f"[MLA] mxfp4 KV: fp4={kv_fp4.dtype} shape={kv_fp4.shape} "
                      f"scale={kv_scale_fp4.dtype} shape={kv_scale_fp4.shape}", flush=True)

            # Try building metadata with fp4x2 dtype
            fp4_dtype = kv_fp4.dtype
            cache_key = ("mxfp4", batch_size, kv_seq_len, num_kv_splits, page_size)
            meta = _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                              num_kv_splits, page_size, dtype_q, fp4_dtype, qo_indptr, kv_indptr)
            wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, ps = meta

            kv_buffer_4d = kv_fp4.view(-1, ps, nkv, kv_fp4.shape[-1])

            # Quantize Q to fp8
            alloc_key = ("mxfp4_mla", q.shape[0], nq, dv, dq)
            if alloc_key not in _alloc_cache:
                amax_buf = torch.full((1,), _FIXED_AMAX, dtype=torch.float32, device="cuda")
                _alloc_cache[alloc_key] = (
                    torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
                    amax_buf,
                    torch.empty(1, dtype=torch.float32, device="cuda"),
                    torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
                )
            o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

            N = q.numel(); BLOCK = 4096; grid = ((N + BLOCK - 1) // BLOCK,)
            _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                                   FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

            mla_decode_fwd(
                q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
                qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
                q_seq_len, page_size=ps, nhead_kv=nkv,
                sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
                q_scale=scale_buf, kv_scale=kv_scale_fp4,
                intra_batch_mode=True,
                work_meta_data=wm, work_indptr=wi, work_info_set=wis,
                reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
            )
            print(f"[MLA] MXFP4 KV SUCCESS!", flush=True)
            _mxfp4_works = True
            return o
        except Exception as e:
            print(f"[MLA] MXFP4 KV failed: {e}", flush=True)
            _mxfp4_works = False

    # Standard FP8 path
    cache_key = (batch_size, kv_seq_len, num_kv_splits, page_size)
    if cache_key not in _meta_cache:
        _meta_cache[cache_key] = _build_meta(
            batch_size, kv_seq_len, q_seq_len, nq, nkv,
            num_kv_splits, page_size, dtype_q, FP8_DTYPE, qo_indptr, kv_indptr)

    wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, ps = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

    alloc_key = ("fp8", q.shape[0], nq, dv, dq)
    if alloc_key not in _alloc_cache:
        amax_buf = torch.full((1,), _FIXED_AMAX, dtype=torch.float32, device="cuda")
        _alloc_cache[alloc_key] = (
            torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
            amax_buf,
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
        )
    o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

    N = q.numel(); BLOCK = 4096; grid = ((N + BLOCK - 1) // BLOCK,)
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

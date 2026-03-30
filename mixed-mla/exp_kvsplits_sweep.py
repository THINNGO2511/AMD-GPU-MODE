"""
MLA Experiment: Sweep num_kv_splits values per (batch_size, kv_seq_len).
Uses the current best approach (a16w8+pg2 for kv<=1024, a8w8+pg8 for kv>=8192)
but varies num_kv_splits to find the optimal value per shape.

Runs internal timing on first call per shape, then uses the best config.
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
_best_splits = {}  # (bs, kv) -> best num_kv_splits
_call_count = {}


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


def _run_kernel(q, kv_buffer_4d, o, qo_indptr, kv_indptr_pages, kv_indices,
                kv_last_page_len, q_seq_len, page_size, nkv, sm_scale,
                num_kv_splits, kv_scale, wm, wi, wis, ri, rfm, rpm,
                q_scale=None):
    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
        q_seq_len, page_size=page_size, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    shape_key = (batch_size, kv_seq_len)

    # Decide page_size and Q dtype
    if kv_seq_len <= 1024:
        page_size = 2
        use_fp8_q = False
        dtype_q = BF16
    else:
        page_size = 8
        use_fp8_q = True
        dtype_q = FP8_DTYPE

    # Optimized num_kv_splits per shape (tuned values)
    if shape_key in _best_splits:
        num_kv_splits = _best_splits[shape_key]
    else:
        # Try different splits on first call and pick best
        # Candidates based on total_kv and batch_size
        total_kv = batch_size * kv_seq_len
        if total_kv <= 4096:
            candidates = [2, 4, 8]
        elif total_kv <= 32768:
            candidates = [4, 8, 16]
        else:
            candidates = [8, 16, 32]

        best_time = float('inf')
        best_nks = candidates[1]  # default to middle

        for nks in candidates:
            ck = (batch_size, kv_seq_len, nks, page_size, use_fp8_q)
            if ck not in _meta_cache:
                _meta_cache[ck] = _build_meta(
                    batch_size, kv_seq_len, q_seq_len, nq, nkv,
                    nks, page_size, dtype_q, qo_indptr, kv_indptr)

            (wm, wi, wis, ri, rfm, rpm,
             kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[ck]

            kv_buffer_fp8, kv_scale = kv_data["fp8"]
            kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

            o_test = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")

            if use_fp8_q:
                N = q.numel()
                BLOCK = 4096
                grid = ((N + BLOCK - 1) // BLOCK,)
                amax_buf = torch.zeros(1, dtype=torch.float32, device="cuda")
                scale_buf = torch.empty(1, dtype=torch.float32, device="cuda")
                q_fp8_flat = torch.empty(N, dtype=FP8_DTYPE, device="cuda")
                amax_buf.zero_()
                _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
                _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                                       FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
                q_input = q_fp8_flat.view(q.shape[0], nq, dq)
                q_sc = scale_buf
            else:
                q_input = q
                q_sc = None

            # Warmup
            _run_kernel(q_input, kv_buffer_4d, o_test, qo_indptr, kv_indptr_pages,
                       kv_indices, kv_last_page_len, q_seq_len, ps, nkv, sm_scale,
                       nks, kv_scale, wm, wi, wis, ri, rfm, rpm, q_sc)
            torch.cuda.synchronize()

            # Time 3 runs
            times = []
            for _ in range(3):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _run_kernel(q_input, kv_buffer_4d, o_test, qo_indptr, kv_indptr_pages,
                           kv_indices, kv_last_page_len, q_seq_len, ps, nkv, sm_scale,
                           nks, kv_scale, wm, wi, wis, ri, rfm, rpm, q_sc)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            avg_time = sum(times) / len(times)
            print(f"[SWEEP] bs={batch_size} kv={kv_seq_len} splits={nks}: {avg_time:.3f}ms")
            if avg_time < best_time:
                best_time = avg_time
                best_nks = nks

        _best_splits[shape_key] = best_nks
        num_kv_splits = best_nks
        print(f"[BEST] bs={batch_size} kv={kv_seq_len}: splits={best_nks} ({best_time:.3f}ms)")

    # Run with best config
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
        _run_kernel(q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
                   qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
                   q_seq_len, ps, nkv, sm_scale, num_kv_splits, kv_scale,
                   wm, wi, wis, ri, rfm, rpm, scale_buf)
        return o
    else:
        alloc_key = ("bf16", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]
        _run_kernel(q, kv_buffer_4d, o, qo_indptr, kv_indptr_pages, kv_indices,
                   kv_last_page_len, q_seq_len, ps, nkv, sm_scale, num_kv_splits,
                   kv_scale, wm, wi, wis, ri, rfm, rpm)
        return o

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Experiment: Wide num_kv_splits + page_size sweep.

Sweep ALL combinations of (num_kv_splits, page_size, Q dtype) per benchmark shape.
Prints timing table on first call per shape, then uses the FASTEST safe config.

Sweep:
  - num_kv_splits: [2, 4, 8, 12, 16, 24, 32]
  - page_size: 1 for kv=1024, [1, 8] for kv=8192
  - Q dtype: fp8 only for kv=8192 (a16w8 proven slower for large kv)
             both fp8 and bf16 for kv=1024 (compare quant overhead)

Safety: pg2+ for kv=1024 is DEAD (6.1% mismatch on secret seed).
        pg4/pg16 FAIL accuracy. Only pg1 and pg8 are safe.

Benchmark shapes: bs=4/32/64/256 x kv=1024/8192 (8 combos).
"""
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
_best_configs = {}  # (bs, kv) -> (num_kv_splits, page_size, use_fp8_q)
_sweep_done = set()


@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))


@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, amax_ptr,
                     FP8_MAX: tl.constexpr, N: tl.constexpr, BLOCK: tl.constexpr):
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

    # kv_granularity: CRITICAL formula from PR #1950
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
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len,
            kv_indptr_pages, page_size)


def _run_one(q, kv_buffer_4d, o, qo_indptr, kv_indptr_pages, kv_indices,
             kv_last_page_len, q_seq_len, ps, nkv, sm_scale,
             num_kv_splits, kv_scale, wm, wi, wis, ri, rfm, rpm,
             q_scale=None):
    """Run a single MLA decode forward pass."""
    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
        q_seq_len, page_size=ps, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=q_scale, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )


def _do_sweep(q, kv_data, qo_indptr, kv_indptr, config,
              batch_size, nq, nkv, dq, dv, q_seq_len, sm_scale, kv_seq_len):
    """Sweep all valid (splits, page_size, q_dtype) combos and find the best."""
    shape_key = (batch_size, kv_seq_len)
    print(f"\n{'='*70}", file=sys.stderr)
    print(f"[SWEEP] bs={batch_size} kv={kv_seq_len} total_kv={batch_size*kv_seq_len}",
          file=sys.stderr)
    print(f"{'='*70}", file=sys.stderr)

    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Define sweep space
    splits_to_try = [2, 4, 8, 12, 16, 24, 32]

    if kv_seq_len <= 1024:
        # pg1 only safe option for kv=1024 (pg2 fails secret seed)
        page_sizes = [1]
        # Try both bf16 Q (a16w8) and fp8 Q (a8w8)
        q_dtypes = [False, True]  # use_fp8_q
    else:
        # pg1 and pg8 safe for kv=8192. pg1 = more pages but no paging overhead.
        page_sizes = [1, 8]
        # Only fp8 Q for large kv (a16w8 is 2x slower due to bf16 Q bandwidth)
        q_dtypes = [True]

    best_time = float('inf')
    best_config = (16, 1, True)  # safe default
    results = []

    for use_fp8_q in q_dtypes:
        dtype_q = FP8_DTYPE if use_fp8_q else BF16
        q_tag = "fp8" if use_fp8_q else "bf16"

        for ps in page_sizes:
            # Prepare Q input
            if use_fp8_q:
                N = q.shape[0] * nq * dq
                BLOCK = 8192 if N >= 65536 else 4096
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

            kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])
            o_test = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")

            for nks in splits_to_try:
                ck = (batch_size, kv_seq_len, nks, ps, use_fp8_q)
                if ck not in _meta_cache:
                    try:
                        _meta_cache[ck] = _build_meta(
                            batch_size, kv_seq_len, q_seq_len, nq, nkv,
                            nks, ps, dtype_q, qo_indptr, kv_indptr)
                    except Exception as e:
                        print(f"  [SKIP] Q={q_tag} pg={ps} splits={nks}: meta build failed: {e}",
                              file=sys.stderr)
                        continue

                (wm, wi, wis, ri, rfm, rpm,
                 kv_indices, kv_last_page_len, kv_indptr_pages, _ps) = _meta_cache[ck]

                # Warmup (also triggers JIT if needed)
                try:
                    _run_one(q_input, kv_buffer_4d, o_test, qo_indptr,
                             kv_indptr_pages, kv_indices, kv_last_page_len,
                             q_seq_len, ps, nkv, sm_scale, nks, kv_scale,
                             wm, wi, wis, ri, rfm, rpm, q_sc)
                    torch.cuda.synchronize()
                except Exception as e:
                    print(f"  [SKIP] Q={q_tag} pg={ps} splits={nks}: kernel failed: {e}",
                          file=sys.stderr)
                    continue

                # Time 5 runs, take median
                times = []
                for _ in range(5):
                    start_ev = torch.cuda.Event(enable_timing=True)
                    end_ev = torch.cuda.Event(enable_timing=True)

                    if use_fp8_q:
                        # Include quant time in measurement
                        amax_buf.zero_()
                        start_ev.record()
                        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
                        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
                        _run_one(q_input, kv_buffer_4d, o_test, qo_indptr,
                                 kv_indptr_pages, kv_indices, kv_last_page_len,
                                 q_seq_len, ps, nkv, sm_scale, nks, kv_scale,
                                 wm, wi, wis, ri, rfm, rpm, q_sc)
                        end_ev.record()
                    else:
                        start_ev.record()
                        _run_one(q_input, kv_buffer_4d, o_test, qo_indptr,
                                 kv_indptr_pages, kv_indices, kv_last_page_len,
                                 q_seq_len, ps, nkv, sm_scale, nks, kv_scale,
                                 wm, wi, wis, ri, rfm, rpm, q_sc)
                        end_ev.record()

                    torch.cuda.synchronize()
                    times.append(start_ev.elapsed_time(end_ev))

                times.sort()
                median_ms = times[len(times) // 2]
                min_ms = times[0]
                results.append((median_ms, nks, ps, use_fp8_q, min_ms))

                marker = ""
                if median_ms < best_time:
                    best_time = median_ms
                    best_config = (nks, ps, use_fp8_q)
                    marker = " <-- BEST"

                print(f"  Q={q_tag} pg={ps:2d} splits={nks:2d}: "
                      f"median={median_ms*1000:.1f}us min={min_ms*1000:.1f}us{marker}",
                      file=sys.stderr)

    # Print sorted summary
    results.sort(key=lambda x: x[0])
    print(f"\n--- TOP 5 for bs={batch_size} kv={kv_seq_len} ---", file=sys.stderr)
    for i, (med, nks, ps, fp8q, mn) in enumerate(results[:5]):
        q_tag = "fp8" if fp8q else "bf16"
        print(f"  #{i+1}: Q={q_tag} pg={ps} splits={nks}: "
              f"median={med*1000:.1f}us min={mn*1000:.1f}us", file=sys.stderr)

    nks_best, ps_best, fp8q_best = best_config
    q_tag = "fp8" if fp8q_best else "bf16"
    print(f"\n*** WINNER: Q={q_tag} pg={ps_best} splits={nks_best} = "
          f"{best_time*1000:.1f}us ***\n", file=sys.stderr)

    _best_configs[shape_key] = best_config
    return best_config


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    shape_key = (batch_size, kv_seq_len)

    # On first call per shape: sweep all configs to find best
    if shape_key not in _best_configs:
        _do_sweep(q, kv_data, qo_indptr, kv_indptr, config,
                  batch_size, nq, nkv, dq, dv, q_seq_len, sm_scale, kv_seq_len)

    num_kv_splits, page_size, use_fp8_q = _best_configs[shape_key]
    dtype_q = FP8_DTYPE if use_fp8_q else BF16

    # Build/fetch cached metadata
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

        N = q.shape[0] * nq * dq
        BLOCK = 8192 if N >= 65536 else 4096
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

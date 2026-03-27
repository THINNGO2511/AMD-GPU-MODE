#!POPCORN leaderboard amd-mixed-mla  
#!POPCORN gpu MI355X
"""MLA: Use EXACT same API as reference.py but with pg2 + optimized splits.
Match the reference mla_decode_fwd call signature exactly."""
import torch
import aiter
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
import triton, triton.language as tl

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 2
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_cache = {}

@triton.jit
def _q_amax_k(q_ptr, out_ptr, N, BS: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BS + tl.arange(0, BS)
    m = off < N
    x = tl.load(q_ptr + off, mask=m, other=0.0).to(tl.float32)
    tl.atomic_max(out_ptr, tl.max(tl.abs(x)))

@triton.jit  
def _q_cast_k(x_ptr, o_ptr, s_ptr, a_ptr, FP8M: tl.constexpr, N, BS: tl.constexpr):
    a = tl.load(a_ptr)
    a = tl.where(a < 1e-12, 1e-12, a)
    s = a / FP8M
    if tl.program_id(0) == 0: tl.store(s_ptr, s)
    pid = tl.program_id(0)
    off = pid * BS + tl.arange(0, BS)
    m = off < N
    x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32) / s
    tl.store(o_ptr + off, tl.clamp(x, -FP8M, FP8M).to(o_ptr.dtype.element_ty), mask=m)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    q_seq_len = config.get("q_seq_len", 1)
    total_kv = bs * kv_seq_len

    kv_fp8, kv_scale = kv_data["fp8"]
    num_kv_splits = 16 if total_kv > 8192 else 8

    ck = (bs, kv_seq_len, num_kv_splits)
    if ck not in _cache:
        # pg2 setup
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        num_pages = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr_pg = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
        kv_indptr_pg[1:] = torch.cumsum(num_pages, 0)
        total_pages = kv_indptr_pg[-1].item()
        kv_last = seq_lens % PAGE_SIZE
        kv_last[kv_last == 0] = PAGE_SIZE
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=q.device)

        # Metadata — match reference exactly
        info = get_mla_metadata_info_v1(
            bs, q_seq_len, nq, FP8_DTYPE, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True)
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr_pg, kv_last,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=FP8_DTYPE)

        o = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device="cuda")
        amax = torch.zeros(1, dtype=torch.float32, device="cuda")
        scale = torch.empty(1, dtype=torch.float32, device="cuda")
        qfp8 = torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda")

        _cache[ck] = (kv_indptr_pg, kv_last, kv_indices,
                      {"work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
                       "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm},
                      o, amax, scale, qfp8)

    kv_indptr_pg, kv_last, kv_indices, meta, o, amax, scale, qfp8 = _cache[ck]

    # Quantize Q to fp8
    N = q.numel()
    BS = 4096
    grid = ((N + BS - 1) // BS,)
    amax.zero_()
    _q_amax_k[grid](q, amax, N, BS=BS)
    _q_cast_k[grid](q, qfp8, scale, amax, FP8M=_FP8_MAX, N=N, BS=BS)
    q_fp8 = qfp8.view(q.shape[0], nq, dq)

    kv_4d = kv_fp8.view(-1, PAGE_SIZE, nkv, kv_fp8.shape[-1])

    # Call mla_decode_fwd with EXACT reference API signature
    mla_decode_fwd(
        q_fp8, kv_4d, o,
        qo_indptr, kv_indptr_pg, kv_indices, kv_last,
        q_seq_len,
        page_size=PAGE_SIZE, nhead_kv=nkv, sm_scale=sm_scale, logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=scale, kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o

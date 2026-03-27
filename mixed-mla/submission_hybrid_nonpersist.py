#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Hybrid: Non-persistent for small cases, persistent for large.
- bs<=32, kv=1024: non-persistent (faster, no metadata overhead)
- everything else: persistent pg2 (from hybrid_v5, proven best)
"""
import torch
import triton
import triton.language as tl
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
PAGE_SIZE = 2
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_cache = {}

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
    if tl.program_id(0) == 0: tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


def _get_splits_nonpersist(bs, total_kv):
    """Aiter's auto-split formula for non-persistent."""
    cu_num = 304
    avg_kv = total_kv / bs
    overhead = 84.1
    tmp = [(bs*i/((bs*i+cu_num-1)//cu_num*cu_num)*avg_kv/(avg_kv+overhead*i), i)
           for i in range(1, 17)]
    ns = sorted(tmp, key=lambda x: x[0], reverse=True)[0][1]
    min_bn = 32  # fp8 nhead=16 → min_block_n=32
    ns = min(ns, int(total_kv/bs + min_bn - 1) // min_bn)
    if ns > 1: ns = min(ns, int(abs(total_kv/bs - 1) // min_bn + 1))
    return max(1, ns)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = bs * kv_seq_len

    kv_fp8, kv_scale = kv_data["fp8"]
    
    # Decision: non-persistent for small kv, persistent for large
    use_nonpersist = (kv_seq_len <= 1024 and bs <= 64)
    use_a16w8 = (kv_seq_len <= 1024 and not use_nonpersist)
    
    ck = (bs, kv_seq_len, use_nonpersist)
    if ck not in _cache:
        # pg2 setup
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        num_pages = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr_pg = torch.zeros(bs+1, dtype=torch.int32, device=q.device)
        kv_indptr_pg[1:] = torch.cumsum(num_pages, 0)
        total_pages = kv_indptr_pg[-1].item()
        kv_last = seq_lens % PAGE_SIZE
        kv_last[kv_last == 0] = PAGE_SIZE
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=q.device)

        o = torch.empty(q.shape[0], nq, dv, dtype=BF16, device="cuda")
        amax_buf = torch.zeros(1, dtype=torch.float32, device="cuda")
        scale_buf = torch.empty(1, dtype=torch.float32, device="cuda")
        q_fp8_flat = torch.empty(q.shape[0]*nq*dq, dtype=FP8_DTYPE, device="cuda")

        if use_nonpersist:
            ns = _get_splits_nonpersist(bs, total_kv)
            indptr = torch.arange(0, (bs+1)*ns, ns, dtype=torch.int32, device=q.device)
            if ns == 1:
                logits = o.view(q.shape[0], 1, nq, dv)
            else:
                logits = torch.empty(q.shape[0], ns, nq, dv, dtype=torch.float32, device="cuda")
            lse = torch.empty(q.shape[0], ns, nq, 1, dtype=torch.float32, device="cuda")
            _cache[ck] = ("nonpersist", kv_indptr_pg, kv_last, kv_indices,
                          o, amax_buf, scale_buf, q_fp8_flat,
                          ns, indptr, logits, lse)
        else:
            num_kv_splits = 16 if total_kv > 8192 else 8
            info = get_mla_metadata_info_v1(
                bs, 1, nq, FP8_DTYPE if not use_a16w8 else BF16, FP8_DTYPE,
                is_sparse=False, fast_mode=False,
                num_kv_splits=num_kv_splits, intra_batch_mode=True)
            work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
            (wm, wi, wis, ri, rfm, rpm) = work
            get_mla_metadata_v1(
                qo_indptr, kv_indptr_pg, kv_last,
                nq//nkv, nkv, True, wm, wis, wi, ri, rfm, rpm,
                page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE,16),
                max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
                max_split_per_batch=num_kv_splits, intra_batch_mode=True,
                dtype_q=FP8_DTYPE if not use_a16w8 else BF16, dtype_kv=FP8_DTYPE)
            np = rpm.size(0)
            logits = torch.empty(np, 1, nq, dv, dtype=torch.float32, device="cuda")
            lse = torch.empty(np, 1, nq, 1, dtype=torch.float32, device="cuda")
            _cache[ck] = ("persistent", kv_indptr_pg, kv_last, kv_indices,
                          o, amax_buf, scale_buf, q_fp8_flat,
                          wm, wi, wis, ri, rfm, rpm, logits, lse, use_a16w8)

    entry = _cache[ck]
    kv_4d = kv_fp8.view(-1, PAGE_SIZE, nkv, kv_fp8.shape[-1])

    if entry[0] == "nonpersist":
        _, kv_indptr_pg, kv_last, kv_indices, o, amax_buf, scale_buf, q_fp8_flat, ns, indptr, logits, lse = entry
        # Q to fp8
        N = q.numel(); BLOCK = 4096; grid = ((N+BLOCK-1)//BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf, FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
        q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)

        aiter.mla_decode_stage1_asm_fwd(
            q_fp8, kv_4d, qo_indptr, kv_indptr_pg, kv_indices, kv_last,
            indptr, None, None, None,
            1, PAGE_SIZE, nkv, sm_scale,
            logits, lse, o, scale_buf, kv_scale)
        # splits=1 → output already in o
    else:
        _, kv_indptr_pg, kv_last, kv_indices, o, amax_buf, scale_buf, q_fp8_flat, wm, wi, wis, ri, rfm, rpm, logits, lse, is_a16w8 = entry
        if is_a16w8:
            aiter.mla_decode_stage1_asm_fwd(
                q, kv_4d, qo_indptr, kv_indptr_pg, kv_indices, kv_last,
                None, wm, wi, wis, 1, PAGE_SIZE, nkv, sm_scale,
                logits, lse, o, None, kv_scale)
        else:
            N = q.numel(); BLOCK = 4096; grid = ((N+BLOCK-1)//BLOCK,)
            amax_buf.zero_()
            _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
            _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf, FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
            q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)
            aiter.mla_decode_stage1_asm_fwd(
                q_fp8, kv_4d, qo_indptr, kv_indptr_pg, kv_indices, kv_last,
                None, wm, wi, wis, 1, PAGE_SIZE, nkv, sm_scale,
                logits, lse, o, scale_buf, kv_scale)
        aiter.mla_reduce_v1(logits, lse, ri, rfm, rpm, 1, o, None)

    return o

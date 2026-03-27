#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Direct stage1+reduce + pg2 + aggressive split tuning v2.
Changes from v1:
- More aggressive per-size split tuning
- kv_granularity tuning (try 8 instead of 16 for pg2)
- Larger BLOCK for Q quant (less kernel launch overhead)
"""
import torch
import triton
import triton.language as tl
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 2
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)

_meta_cache = {}
_alloc_cache = {}

# Per-size tuned splits: (bs, kv_seq_len) → num_kv_splits
# Goal: balance stage1 parallelism vs reduce overhead
# Total work items = bs × num_kv_splits × nhead
# MI355X has 304 CUs. Target: 2-4 waves per CU = 600-1200 work items
_SPLIT_MAP = {
    (4, 1024): 16,    # 4*16*16 = 1024 work items
    (4, 8192): 32,    # 4*32*16 = 2048 work items
    (32, 1024): 8,    # 32*8*16 = 4096 work items
    (32, 8192): 16,   # 32*16*16 = 8192 work items
    (64, 1024): 8,    # 64*8*16 = 8192 work items
    (64, 8192): 8,    # 64*8*16 = 8192 (fewer splits to reduce overhead)
    (256, 1024): 4,   # 256*4*16 = 16384 work items
    (256, 8192): 4,   # 256*4*16 = 16384 (reduce is bottleneck at high bs)
}


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


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    total_kv = batch_size * kv_seq_len
    num_kv_splits = _SPLIT_MAP.get((batch_size, kv_seq_len), 16)
    num_pages = total_kv // PAGE_SIZE

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        kv_indptr_pages = kv_indptr // PAGE_SIZE
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % PAGE_SIZE).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, PAGE_SIZE, kv_last_page_len)

        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, FP8_DTYPE, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (work_metadata, work_indptr, work_info_set,
         reduce_indptr, reduce_final_map, reduce_partial_map) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_pages, kv_last_page_len,
            nq // nkv, nkv, True,
            work_metadata, work_info_set, work_indptr,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            page_size=PAGE_SIZE,
            kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=FP8_DTYPE,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")

        n_partials = reduce_partial_map.size(0)
        logits = torch.empty(
            (n_partials * q_seq_len, 1, nq, dv),
            dtype=torch.float32, device="cuda",
        )
        attn_lse = torch.empty(
            (n_partials * q_seq_len, 1, nq, 1),
            dtype=torch.float32, device="cuda",
        )

        _meta_cache[cache_key] = (
            work_metadata, work_indptr, work_info_set,
            reduce_indptr, reduce_final_map, reduce_partial_map,
            kv_indices, kv_last_page_len, kv_indptr_pages,
            logits, attn_lse,
        )

    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map,
     kv_indices, kv_last_page_len, kv_indptr_pages,
     logits, attn_lse) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv, dq)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = (
            torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda"),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
        )
    o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    N = q.numel()
    BLOCK = 8192  # Larger block = fewer blocks = less atomic contention
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                           FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)

    q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)

    aiter.mla_decode_stage1_asm_fwd(
        q_fp8, kv_buffer_4d,
        qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
        None, work_metadata, work_indptr, work_info_set,
        q_seq_len, PAGE_SIZE, nkv, sm_scale,
        logits, attn_lse, o,
        scale_buf, kv_scale,
    )

    aiter.mla_reduce_v1(
        logits, attn_lse,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        q_seq_len, o, None,
    )

    return o

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Decode — a16w8 kernel (bf16 Q + fp8 KV) for ALL shapes.
Avoids Q->fp8 quantization overhead entirely by passing bf16 Q directly.
Uses the mla_a16w8 ASM kernel path.
"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16_DTYPE = torch.bfloat16
PAGE_SIZE = 1

_meta_cache = {}
_alloc_cache = {}


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

    num_kv_splits = 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        total_kv_len = batch_size * kv_seq_len
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, BF16_DTYPE, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE,
            kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=BF16_DTYPE,
            dtype_kv=FP8_DTYPE,
        )

        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty(
            (q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda"
        )
    o = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(
        kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1]
    )

    # Pass bf16 Q directly — no quantization needed for a16w8 path
    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o

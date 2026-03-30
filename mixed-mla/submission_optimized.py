#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Decode — Use aiter's optimized mla_decode_fwd persistent kernel.

The baseline submission.py uses torch._scaled_mm in a Python loop which is very slow.
This submission uses the same optimized mla_decode_fwd kernel as the reference,
with fp8 Q + fp8 KV (a8w8 mode), which is the fastest available aiter kernel.

This should match the reference performance. To beat it, we'd need a custom
kernel with mxfp4 KV cache (future optimization).
"""
import torch
from task import input_t, output_t

from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 1


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def _choose_num_kv_splits(batch_size, kv_seq_len):
    """Adaptive split-K for MLA decode based on problem size."""
    # More splits = more parallelism but more reduction overhead
    # For small kv_seq_len, fewer splits avoid wasted work
    # For large batch, fewer splits since batches already provide parallelism
    total_kv_work = batch_size * kv_seq_len
    if total_kv_work >= 1024 * 1024:  # bs=256 * kv=8192
        return 16
    elif total_kv_work >= 64 * 1024:
        return 32
    else:
        return 32  # default for small problems


def _make_mla_decode_metadata(
    batch_size, max_q_len, nhead, nhead_kv,
    q_dtype, kv_dtype,
    qo_indptr, kv_indptr, kv_last_page_len,
    num_kv_splits=32,
):
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, nhead, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (work_metadata, work_indptr, work_info_set,
     reduce_indptr, reduce_final_map, reduce_partial_map) = work

    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        nhead // nhead_kv,
        nhead_kv,
        True,  # is_causal
        work_metadata, work_info_set, work_indptr,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=max_q_len,
        uni_seqlen_qo=max_q_len,
        fast_mode=False,
        max_split_per_batch=num_kv_splits,
        intra_batch_mode=True,
        dtype_q=q_dtype,
        dtype_kv=kv_dtype,
    )

    return {
        "work_meta_data": work_metadata,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]

    # Quantize Q to fp8
    q_fp8, q_scale = quantize_fp8(q)

    # Use fp8 KV cache
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    total_kv_len = int(kv_indptr[-1].item())
    kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")

    # Reshape KV for aiter: (total_kv, page_size, nhead_kv, dim)
    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    max_q_len = q_seq_len
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

    # Adaptive split-K
    kv_seq_len = config["kv_seq_len"]
    num_kv_splits = _choose_num_kv_splits(batch_size, kv_seq_len)

    # Build metadata
    meta = _make_mla_decode_metadata(
        batch_size, max_q_len, nq, nkv,
        q_fp8.dtype, kv_buffer_fp8.dtype,
        qo_indptr, kv_indptr, kv_last_page_len,
        num_kv_splits=num_kv_splits,
    )

    # Run optimized decode
    o = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q_fp8.view(-1, nq, dq),
        kv_buffer_4d,
        o,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        max_q_len,
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=sm_scale,
        logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=q_scale,
        kv_scale=kv_scale,
        intra_batch_mode=True,
        **meta,
    )
    return o

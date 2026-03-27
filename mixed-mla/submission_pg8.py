"""MLA with page_size=8 for kv=8192 — Danishlynx #1 approach"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

NUM_HEADS = 16
NUM_KV_HEADS = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
FP8_DTYPE = aiter_dtypes.fp8

_meta_cache = {}

def _quantize_fp8(tensor):
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8 = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8, scale.to(torch.float32).reshape(1)

def _make_meta(batch_size, max_q_len, q_dtype, kv_dtype, qo_indptr, kv_indptr, kv_last_page_len, page_size, num_kv_splits):
    key = (batch_size, max_q_len, q_dtype, kv_dtype, page_size, num_kv_splits)
    if key in _meta_cache:
        return _meta_cache[key]
    
    kv_gran = max(page_size, 16)
    info = get_mla_metadata_info_v1(
        batch_size, max_q_len, NUM_HEADS, q_dtype, kv_dtype,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work
    get_mla_metadata_v1(
        qo_indptr, kv_indptr, kv_last_page_len,
        NUM_HEADS // NUM_KV_HEADS, NUM_KV_HEADS, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size, kv_granularity=kv_gran,
        max_seqlen_qo=max_q_len, uni_seqlen_qo=max_q_len,
        fast_mode=False, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=q_dtype, dtype_kv=kv_dtype,
    )
    meta = {"work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
            "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm}
    _meta_cache[key] = meta
    return meta

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_len = config["kv_seq_len"]
    
    # Quantize Q to fp8
    q_fp8, q_scale = _quantize_fp8(q)
    kv_fp8, kv_scale = kv_data["fp8"]
    
    # KEY: use page_size=8 for kv=8192, page_size=1 for small kv
    if kv_len >= 8192:
        page_size = 8
        num_splits = 16
    elif kv_len >= 1024:
        page_size = 2
        num_splits = 16
    else:
        page_size = 1
        num_splits = 8
    
    total_kv = int(kv_indptr[-1].item())
    kv_4d = kv_fp8.view(kv_fp8.shape[0], 1, NUM_KV_HEADS, kv_fp8.shape[-1])
    
    # Build paged indices
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    
    meta = _make_meta(batch_size, 1, q_fp8.dtype, kv_fp8.dtype,
                      qo_indptr, kv_indptr, kv_last_page_len, page_size, num_splits)
    
    o = torch.empty(q.shape[0], NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    mla_decode_fwd(
        q_fp8, kv_4d, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        1, page_size=page_size, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
        logit_cap=0.0, num_kv_splits=num_splits,
        q_scale=q_scale, kv_scale=kv_scale, intra_batch_mode=True, **meta,
    )
    return o

"""MLA: pg4 for kv=1024, pg8 for kv=8192 — push further than pg8_v2"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
import triton, triton.language as tl

NUM_HEADS, NUM_KV_HEADS = 16, 1
QK_HEAD_DIM, V_HEAD_DIM = 576, 512
SM_SCALE = 1.0 / (QK_HEAD_DIM ** 0.5)
FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_meta_cache = {}

@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.atomic_max(amax_ptr, tl.max(tl.abs(x)))

@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, N, FP8_MAX: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    scale = tl.load(scale_ptr)
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(tl.float8e4nv), mask=mask)

_alloc = {}
def _fast_fp8_quant(q):
    N = q.numel()
    key = N
    if key not in _alloc:
        _alloc[key] = (torch.zeros(1, dtype=torch.float32, device=q.device),
                       torch.empty(1, dtype=torch.float32, device=q.device),
                       torch.empty(q.shape, dtype=FP8_DTYPE, device=q.device))
    amax_buf, scale_buf, q_fp8 = _alloc[key]
    amax_buf.zero_()
    BLK = 1024
    grid = ((N + BLK - 1) // BLK,)
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLK)
    scale_buf.copy_(amax_buf.clamp(min=1e-12) / torch.finfo(FP8_DTYPE).max)
    _q_to_fp8_kernel[grid](q, q_fp8, scale_buf, N, FP8_MAX=torch.finfo(FP8_DTYPE).max, BLOCK=BLK)
    return q_fp8.view(q.shape), scale_buf.reshape(1)

def _build_meta(bs, kv_len, ps, ns, dtype_q, dtype_kv, qo_ind, kv_ind):
    key = (bs, kv_len, ps, ns, dtype_q, dtype_kv)
    if key in _meta_cache:
        return _meta_cache[key]
    kv_gran = max(1, 16 // ps)
    info = get_mla_metadata_info_v1(bs, 1, NUM_HEADS, dtype_q, dtype_kv,
        is_sparse=False, fast_mode=False, num_kv_splits=ns, intra_batch_mode=True)
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    wm, wi, wis, ri, rfm, rpm = work
    total_kv = bs * kv_len
    kv_last = (kv_ind[1:] - kv_ind[:-1]).to(torch.int32)
    if ps > 1:
        kv_last_pg = (kv_last % ps).to(torch.int32)
        kv_last_pg = torch.where(kv_last_pg == 0, ps, kv_last_pg)
    else:
        kv_last_pg = kv_last
    get_mla_metadata_v1(qo_ind, kv_ind if ps == 1 else kv_ind // ps, kv_last_pg,
        NUM_HEADS, NUM_KV_HEADS, True, wm, wis, wi, ri, rfm, rpm,
        page_size=ps, kv_granularity=kv_gran, max_seqlen_qo=1, uni_seqlen_qo=1,
        fast_mode=False, max_split_per_batch=ns, intra_batch_mode=True,
        dtype_q=dtype_q, dtype_kv=dtype_kv)
    meta = {"work_meta_data": wm, "work_indptr": wi, "work_info_set": wis,
            "reduce_indptr": ri, "reduce_final_map": rfm, "reduce_partial_map": rpm}
    _meta_cache[key] = meta
    return meta

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    kv_len = config["kv_seq_len"]
    kv_fp8, kv_scale = kv_data["fp8"]
    
    if kv_len >= 8192:
        ps, ns = 8, 16
        q_in, q_sc = _fast_fp8_quant(q)
        dtype_q = FP8_DTYPE
    elif kv_len >= 1024:
        ps, ns = 4, 16       # pg4 for kv=1024!
        q_in, q_sc = _fast_fp8_quant(q)
        dtype_q = FP8_DTYPE
    else:
        ps, ns = 1, 8
        q_in, q_sc = q, None
        dtype_q = BF16
    
    total_kv = int(kv_indptr[-1].item())
    n_pages = total_kv // ps
    kv_4d = kv_fp8.view(-1, ps, NUM_KV_HEADS, QK_HEAD_DIM)
    kv_idx = torch.arange(n_pages, dtype=torch.int32, device="cuda")
    kv_last = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    if ps > 1:
        kv_last_pg = (kv_last % ps).to(torch.int32)
        kv_last_pg = torch.where(kv_last_pg == 0, ps, kv_last_pg)
    else:
        kv_last_pg = kv_last
    
    meta = _build_meta(bs, kv_len, ps, ns, dtype_q, kv_fp8.dtype, qo_indptr, kv_indptr)
    o = torch.empty(q.shape[0], NUM_HEADS, V_HEAD_DIM, dtype=BF16, device="cuda")
    
    mla_decode_fwd(q_in, kv_4d, o, qo_indptr, kv_indptr // ps if ps > 1 else kv_indptr,
        kv_idx, kv_last_pg, 1, page_size=ps, nhead_kv=NUM_KV_HEADS, sm_scale=SM_SCALE,
        logit_cap=0.0, num_kv_splits=ns, q_scale=q_sc, kv_scale=kv_scale,
        intra_batch_mode=True, **meta)
    return o

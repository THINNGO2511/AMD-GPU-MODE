#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: Try the HIGH-LEVEL mla_decode_fwd wrapper instead of direct ASM calls.
The wrapper may auto-select persistent vs non-persistent mode optimally.
Our direct calls always use persistent — maybe non-persistent is faster for small kv."""
import torch
import aiter
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    
    kv_fp8, kv_scale = kv_data["fp8"]
    
    # Pre-allocate output
    key = (q.shape[0], nq, dv)
    if key not in _cache:
        _cache[key] = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device=q.device)
    output = _cache[key]
    
    # Use the HIGH-LEVEL wrapper — let it decide persistent vs non-persistent
    # page_size=1 (no paging), let it auto-select num_kv_splits
    mla_decode_fwd(
        q, kv_fp8,
        qo_indptr, kv_indptr,
        output=output,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        page_size=1,
        num_kv_splits=0,  # 0 = auto-select
        intra_batch_mode=True,
    )
    return output

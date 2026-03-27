#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: Use mla_decode_fwd wrapper with pg2.
The wrapper handles metadata internally — maybe it does pg2 better than us."""
import torch
import aiter
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

PAGE_SIZE = 2
_cache = {}

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = bs * kv_seq_len
    
    kv_fp8, kv_scale = kv_data["fp8"]
    
    # Reshape KV for pg2
    kv_4d = kv_fp8.view(-1, PAGE_SIZE, nkv, kv_fp8.shape[-1])
    
    # Build pg2 indptrs
    key = (bs, kv_seq_len)
    if key not in _cache:
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        num_pages = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr_paged = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
        kv_indptr_paged[1:] = torch.cumsum(num_pages, 0)
        total_pages = kv_indptr_paged[-1].item()
        kv_last = seq_lens % PAGE_SIZE
        kv_last[kv_last == 0] = PAGE_SIZE
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=q.device)
        output = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device=q.device)
        _cache[key] = (kv_indptr_paged, kv_last, kv_indices, output)
    
    kv_indptr_paged, kv_last, kv_indices, output = _cache[key]
    
    # Let the wrapper handle everything including metadata + dispatch
    mla_decode_fwd(
        q, kv_4d,
        qo_indptr, kv_indptr_paged,
        kv_page_indices=kv_indices,
        kv_last_page_lens=kv_last,
        output=output,
        sm_scale=sm_scale,
        kv_scale=kv_scale,
        page_size=PAGE_SIZE,
        num_kv_splits=0,  # auto
        intra_batch_mode=True,
    )
    return output

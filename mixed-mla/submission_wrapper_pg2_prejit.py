#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: Use mla_decode_fwd wrapper with pg2 + pre-JIT warmup.
The wrapper auto-selects persistent vs non-persistent and optimal splits."""
import torch
import aiter
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes

# PRE-JIT: Trigger all module builds at import time
try:
    _dq = torch.randn(1, 16, 576, dtype=torch.bfloat16, device="cuda")
    _dk = torch.randn(1, 1, 1, 576, dtype=aiter_dtypes.fp8, device="cuda")
    _do = torch.empty(1, 16, 512, dtype=torch.bfloat16, device="cuda")
    _qi = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    _ki = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    _kx = torch.tensor([0], dtype=torch.int32, device="cuda")
    _kl = torch.tensor([1], dtype=torch.int32, device="cuda")
    mla_decode_fwd(_dq, _dk, _do, _qi, _ki, _kx, _kl, 1,
                   page_size=1, sm_scale=1.0/576**0.5,
                   kv_scale=torch.ones(1, device="cuda"))
except Exception:
    pass
try:
    del _dq, _dk, _do, _qi, _ki, _kx, _kl
except NameError:
    pass

from task import input_t, output_t

PAGE_SIZE = 2
_cache = {}

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dv = config["v_head_dim"]
    sm = config["sm_scale"]
    kvl = config["kv_seq_len"]
    tkv = bs * kvl

    # Use bf16 KV for kv<=1024 (zero quant error), fp8 for kv>1024
    use_bf16 = kvl <= 1024
    if use_bf16:
        kv_buf = kv_data["bf16"]
        kv_sc = None
    else:
        kv_fp8, kv_sc = kv_data["fp8"]
        kv_buf = kv_fp8

    ck = (bs, kvl, use_bf16)
    if ck not in _cache:
        sl = kv_indptr[1:] - kv_indptr[:-1]
        np_ = (sl + PAGE_SIZE - 1) // PAGE_SIZE
        ki = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
        ki[1:] = torch.cumsum(np_, 0)
        tp = ki[-1].item()
        kl = sl % PAGE_SIZE
        kl[kl == 0] = PAGE_SIZE
        kx = torch.arange(tp, dtype=torch.int32, device=q.device)
        o = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device="cuda")
        _cache[ck] = (ki, kl, kx, o)

    ki, kl, kx, o = _cache[ck]
    kv_4d = kv_buf.view(-1, PAGE_SIZE, nkv, kv_buf.shape[-1])

    # Let the wrapper handle everything
    mla_decode_fwd(
        q, kv_4d, o,
        qo_indptr, ki, kx, kl,
        1,  # max_seqlen_q
        page_size=PAGE_SIZE,
        nhead_kv=nkv,
        sm_scale=sm,
        kv_scale=kv_sc,
        intra_batch_mode=True,
    )
    return o

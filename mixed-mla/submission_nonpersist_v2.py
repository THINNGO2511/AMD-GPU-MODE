#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Non-persistent mode based on ACTUAL mla.py source code.
Key: pass num_kv_splits_indptr (NOT None), work_meta_data=None.
Auto-select num_kv_splits via get_meta_param formula.
pg2 for 28% KV bandwidth reduction.
"""
import torch
import triton
import triton.language as tl
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes

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
    if tl.program_id(0) == 0:
        tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


def _get_splits(bs, total_kv, nhead, max_seqlen_q):
    """Replicate aiter's get_meta_param auto-split selection."""
    cu_num = 304  # MI355X
    avg_kv = total_kv / bs
    overhead = 84.1
    tmp = [(bs * i / ((bs * i + cu_num - 1) // cu_num * cu_num) * avg_kv / (avg_kv + overhead * i), i)
           for i in range(1, 17)]
    num_kv_splits = sorted(tmp, key=lambda x: x[0], reverse=True)[0][1]
    
    # fp8 min_block_n constraint
    get_block_n = {16: 128, 32: 128, 48: 64, 64: 64, 128: 32, 256: 32, 384: 32, 512: 32}
    min_block_n = get_block_n.get(int(nhead * max_seqlen_q), 32)
    num_kv_splits = min(num_kv_splits, int(total_kv / bs + min_block_n - 1) // min_block_n)
    if num_kv_splits > 1:
        num_kv_splits = min(num_kv_splits, int(abs(total_kv / bs - max_seqlen_q) // min_block_n + 1))
    
    return max(1, num_kv_splits)


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
    kv_4d = kv_fp8.view(-1, PAGE_SIZE, nkv, kv_fp8.shape[-1])

    cache_key = (bs, kv_seq_len)
    if cache_key not in _cache:
        # pg2 setup
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        num_pages = (seq_lens + PAGE_SIZE - 1) // PAGE_SIZE
        kv_indptr_pg = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
        kv_indptr_pg[1:] = torch.cumsum(num_pages, 0)
        total_pages = kv_indptr_pg[-1].item()
        kv_last = seq_lens % PAGE_SIZE
        kv_last[kv_last == 0] = PAGE_SIZE
        kv_indices = torch.arange(total_pages, dtype=torch.int32, device=q.device)

        # Auto-select splits (from actual aiter source)
        num_splits = _get_splits(bs, total_kv, nq, 1)
        
        # Non-persistent needs num_kv_splits_indptr
        num_kv_splits_indptr = torch.arange(
            0, (bs + 1) * num_splits, num_splits,
            dtype=torch.int32, device=q.device)

        # When splits=1 AND fp8: logits = o.view(...) — output written directly!
        if num_splits == 1:
            o = torch.empty(q.shape[0], nq, dv, dtype=BF16, device="cuda")
            logits = o.view(q.shape[0], 1, nq, dv)
        else:
            o = torch.empty(q.shape[0], nq, dv, dtype=BF16, device="cuda")
            logits = torch.empty(q.shape[0], num_splits, nq, dv, dtype=torch.float32, device="cuda")
        attn_lse = torch.empty(q.shape[0], num_splits, nq, 1, dtype=torch.float32, device="cuda")

        amax_buf = torch.zeros(1, dtype=torch.float32, device="cuda")
        scale_buf = torch.empty(1, dtype=torch.float32, device="cuda")
        q_fp8_flat = torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda")

        _cache[cache_key] = (kv_indptr_pg, kv_last, kv_indices, num_splits,
                             num_kv_splits_indptr, o, logits, attn_lse,
                             amax_buf, scale_buf, q_fp8_flat)

    (kv_indptr_pg, kv_last, kv_indices, num_splits,
     num_kv_splits_indptr, o, logits, attn_lse,
     amax_buf, scale_buf, q_fp8_flat) = _cache[cache_key]

    # Quantize Q to fp8
    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                           FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
    q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)

    # NON-PERSISTENT: work_meta_data=None, pass num_kv_splits_indptr
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8, kv_4d,
        qo_indptr, kv_indptr_pg, kv_indices, kv_last,
        num_kv_splits_indptr,  # CRITICAL: needed for non-persistent
        None,  # work_meta_data = None → non-persistent
        None,  # work_indptr = None
        None,  # work_info_set = None
        1,     # max_seqlen_q
        PAGE_SIZE,
        nkv,
        sm_scale,
        logits, attn_lse, o,
        scale_buf, kv_scale,
    )

    # When splits=1 and fp8, output is already in o (logits aliases o)
    if num_splits > 1:
        # Triton reduce
        Lv = dv
        BLOCK_DV = triton.next_power_of_2(Lv)
        _fwd_reduce[bs, nq](
            logits, attn_lse, o,
            qo_indptr, kv_indptr_pg, num_kv_splits_indptr,
            attn_lse.stride(0), attn_lse.stride(2), attn_lse.stride(1),
            o.stride(0), o.stride(1),
            MAYBE_FINAL_OUT=False, BATCH_NUM=bs,
            BLOCK_DV=BLOCK_DV, Lv=Lv, mgc=64,
            num_warps=4, num_stages=2, waves_per_eu=4,
        )

    return o


@triton.jit
def _fwd_reduce(
    Mid_O, Mid_lse, O,
    qo_indptr, kv_indptr, num_kv_splits_indptr,
    stride_mid_ob: tl.int64, stride_mid_oh: tl.int64, stride_mid_os: tl.int64,
    stride_obs: tl.int64, stride_oh: tl.int64,
    MAYBE_FINAL_OUT: tl.constexpr, BATCH_NUM: tl.constexpr,
    BLOCK_DV: tl.constexpr, Lv: tl.constexpr, mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    num_valid = tl.minimum(cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc))
    
    offs_logic = cur_qo_start * stride_mid_ob + cur_head * stride_mid_oh
    offs_v = offs_logic * Lv + offs_d
    
    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros((BLOCK_DV,), dtype=tl.float32)
    for split_id in range(0, num_valid):
        tv = tl.load(Mid_O + offs_v + split_id * stride_mid_os * Lv, mask=mask_d, other=0.0)
        tlogic = tl.load(Mid_lse + offs_logic + split_id * stride_mid_os)
        n_e_max = tl.maximum(tlogic, e_max)
        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv
        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max
    
    tl.store(O + cur_qo_start * stride_obs + cur_head * stride_oh + offs_d,
             acc / e_sum, mask=mask_d)

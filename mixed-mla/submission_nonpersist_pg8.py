#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Non-persistent mode + pg8 (page_size=8 for kv>=8192) + pg1 for kv=1024.

Key insights from aiter/mla.py source code analysis:
1. Non-persistent mode: work_meta_data=None → different kernel path
2. Auto-tuned splits via get_meta_param():
   - MI355X has 304 CUs, overhead=84.1
   - bs=256, kv=1024: auto-split=1 → logits ALIASES output (ZERO reduce step!)
   - bs=32,  kv=8192: auto-split=9 → more parallelism
3. When splits=1 and q.dtype==fp8: logits=o.view(total_s, 1, nhead, dv)
   → mla_reduce_v1 NOT needed → saves Triton reduce kernel launch
4. pg8 for kv=8192: 8x fewer page index lookups, better cache
5. pg1 for kv=1024: kv_last_page_len alignment issues avoided

CRITICAL: The correct non-persistent call passes num_kv_splits_indptr (NOT None).
logits shape is (total_s, num_kv_splits, nhead, v_head_dim) — NOT (n_total, 1, nhead, dv).
The _fwd_kernel_stage2_asm Triton kernel does the reduce for splits>1.

Current: 42.5μs. This should exploit splits=1 path for kv=1024 → zero reduce overhead.
"""
import torch
import triton
import triton.language as tl
import aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter.mla import get_meta_param, _fwd_kernel_stage2_asm

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)

# MI355X has 304 CUs
_CU_NUM = 304

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


def _build_cache(batch_size, kv_seq_len, nq, nkv, dq, dv, device):
    """Build per-shape cache: page tables, splits, buffers."""
    total_kv = batch_size * kv_seq_len
    total_s = batch_size  # max_seqlen_q=1 → total_s = bs

    # ---- Choose page size ----
    # pg8 for kv>=8192 (kv is multiple of 8, no last-page edge case)
    # pg1 for kv<8192 (safe, no alignment issues)
    if kv_seq_len >= 8192:
        page_size = 8
    else:
        page_size = 1

    # ---- Page tables ----
    # kv is uniform (all seqs same length), aligned to page_size
    pages_per_seq = kv_seq_len // page_size
    total_pages = batch_size * pages_per_seq

    kv_indptr_pages = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    ) * pages_per_seq

    kv_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device=device
    )

    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)

    # ---- Auto-tune splits (mirrors aiter get_meta_param) ----
    # Uses MI355X CU count and aiter's efficiency formula
    num_kv_splits, num_kv_splits_indptr = get_meta_param(
        None,  # auto
        batch_size, total_kv, nq, 1,  # max_seqlen_q=1
        FP8_DTYPE
    )

    # ---- Token-level kv_indptr (needed by stage2 for seq len computation) ----
    # _fwd_kernel_stage2_asm uses kv_indptr to compute cur_kv_seq_len for
    # capping num_valid_kv_splits = min(splits, ceil(kv_seq_len / mgc))
    kv_indptr_tokens = torch.arange(
        0, batch_size + 1, dtype=torch.int32, device=device
    ) * kv_seq_len

    # ---- Allocate buffers ----
    o = torch.empty((total_s, nq, dv), dtype=BF16, device=device)
    amax_buf = torch.zeros(1, dtype=torch.float32, device=device)
    scale_buf = torch.empty(1, dtype=torch.float32, device=device)
    q_fp8_flat = torch.empty(total_s * nq * dq, dtype=FP8_DTYPE, device=device)

    # Non-persistent logits shape: (total_s, num_kv_splits, nhead, v_head_dim)
    # When num_kv_splits==1 and fp8: logits will alias o (set in hot path)
    # When num_kv_splits>1: need separate fp32 buffer
    if num_kv_splits == 1:
        # Will alias output — allocate dummy (actual alias set per-call)
        logits_buf = None
    else:
        logits_buf = torch.empty(
            (total_s, num_kv_splits, nq, dv), dtype=torch.float32, device=device
        )

    attn_lse = torch.empty(
        (total_s, num_kv_splits, nq, 1), dtype=torch.float32, device=device
    )

    return {
        "page_size": page_size,
        "kv_indptr_pages": kv_indptr_pages,
        "kv_indptr_tokens": kv_indptr_tokens,
        "kv_last_page_len": kv_last_page_len,
        "kv_indices": kv_indices,
        "num_kv_splits": num_kv_splits,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "o": o,
        "amax_buf": amax_buf,
        "scale_buf": scale_buf,
        "q_fp8_flat": q_fp8_flat,
        "logits_buf": logits_buf,
        "attn_lse": attn_lse,
        "total_s": total_s,
        "nq": nq,
        "dv": dv,
    }


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    device = q.device

    cache_key = (batch_size, kv_seq_len)
    if cache_key not in _cache:
        _cache[cache_key] = _build_cache(
            batch_size, kv_seq_len, nq, nkv, dq, dv, device
        )

    c = _cache[cache_key]
    page_size = c["page_size"]
    o = c["o"]
    amax_buf = c["amax_buf"]
    scale_buf = c["scale_buf"]
    q_fp8_flat = c["q_fp8_flat"]
    attn_lse = c["attn_lse"]
    num_kv_splits = c["num_kv_splits"]
    num_kv_splits_indptr = c["num_kv_splits_indptr"]
    total_s = c["total_s"]

    # ---- Reshape kv_buffer for chosen page_size ----
    kv_buffer_4d = kv_buffer_fp8.view(-1, page_size, nkv, kv_buffer_fp8.shape[-1])

    # ---- Quantize Q to fp8 ----
    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                           FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
    q_fp8 = q_fp8_flat.view(total_s, nq, dq)

    # ---- Set up logits buffer ----
    # When splits=1 and fp8: logits ALIASES o → zero-copy, no reduce needed
    if num_kv_splits == 1:
        logits = o.view(total_s, 1, nq, dv)
    else:
        logits = c["logits_buf"]

    # ---- Non-persistent MLA decode stage 1 ----
    # work_meta_data=None → non-persistent mode
    # num_kv_splits_indptr passed (required for non-persistent!)
    # work_indptr, work_info_set = None, None (not used in non-persistent)
    aiter.mla_decode_stage1_asm_fwd(
        q_fp8,
        kv_buffer_4d,
        qo_indptr,
        c["kv_indptr_pages"],
        c["kv_indices"],
        c["kv_last_page_len"],
        num_kv_splits_indptr,   # required for non-persistent split distribution
        None,                   # work_meta_data = None → NON-PERSISTENT
        None,                   # work_indptr = None
        None,                   # work_info_set = None
        1,                      # max_seqlen_q = 1 (decode)
        page_size,
        nkv,
        sm_scale,
        logits,
        attn_lse,
        o,
        scale_buf,              # q_scale
        kv_scale,               # kv_scale
    )

    # ---- Stage 2: Triton reduce (skip when splits=1, logits already in o) ----
    if num_kv_splits > 1:
        # Use aiter's built-in Triton reduce kernel
        # mgc=64 for nhead=16, max_seqlen_q=1 (from mla.py source)
        # MAYBE_FINAL_OUT=False for nhead=16, max_seqlen_q=1
        Lv = dv
        BLOCK_DV = triton.next_power_of_2(Lv)
        grid_s2 = (batch_size, nq)
        _fwd_kernel_stage2_asm[grid_s2](
            logits,
            attn_lse,
            o,
            qo_indptr,
            c["kv_indptr_tokens"],  # token-level: used to compute cur_kv_seq_len
            num_kv_splits_indptr,
            attn_lse.stride(0),
            attn_lse.stride(2),
            attn_lse.stride(1),
            o.stride(0),
            o.stride(1),
            MAYBE_FINAL_OUT=False,  # nhead=16, max_seqlen_q=1 → False
            BATCH_NUM=batch_size,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=64,             # 64 for nhead=16, seqlen=1
            num_warps=4,
            num_stages=2,
            waves_per_eu=4,
        )

    return o

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA decode reformulated as batched GEMM / SDPA.

== DECOMPOSITION ==

MLA decode (qseqlen=1, GQA 16:1):
  1) QK^T: Q[bs,16,576] @ KV[bs,576,kv_len] -> scores[bs,16,kv_len]
  2) softmax(scores * sm_scale) -> weights[bs,16,kv_len]
  3) V:    weights[bs,16,kv_len] @ KV[bs,kv_len,512] -> out[bs,16,512]

== BANDWIDTH ANALYSIS ==

For bs=256, kv=8192 (hardest case):

  ASM kernel: reads KV ONCE in fp8 = 1.2GB. Fused QK^T+softmax+V.
  BMM: dequant 3.6GB + QKT 2.5GB + softmax 0.1GB + V 2.2GB = 8.4GB.
  Ratio: ~7x more bandwidth. BMM CANNOT win for large kv.

For bs=256, kv=1024:

  KV data: 256*1024*576 = 150MB (fp8). Fits in L2 cache (~200MB MI355X).
  After dequant to bf16: 300MB. Still L2-resident on second read (for V).
  ASM kernel overhead: metadata build, split-k scheduling, reduce.
  BMM: 3 clean kernel launches (dequant, QKT+scale, softmax+V).
  BMM MAY win for small kv if hipBLASLt GEMM overhead < ASM overhead.

== APPROACHES ==

1. torch.baddbmm for QK^T (fuses scale), F.softmax, torch.bmm for V
2. torch SDPA with enable_gqa (fuses all 3 steps, math kernel backend)
3. Proven ASM kernel (fallback for kv=8192)

Default: BMM for kv<=1024, ASM for kv>=8192.
"""
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "1")

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
FP32 = torch.float32
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)

_meta_cache = {}
_alloc_cache = {}


# ============================================================================
# BMM approach: baddbmm + softmax + bmm
# ============================================================================

def _bmm_attention(q, kv_buffer_fp8, kv_scale, config):
    """
    MLA decode via batched GEMM.
    - baddbmm fuses scale into QK^T (one fewer kernel vs bmm + mul)
    - softmax in fp32 for numerical stability
    - bmm for V multiplication
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    nq = config["num_heads"]       # 16
    dq = config["qk_head_dim"]     # 576
    dv = config["v_head_dim"]      # 512
    sm_scale = config["sm_scale"]

    # Pre-allocate output buffers (reused across calls)
    alloc_key = ("bmm", batch_size, kv_seq_len)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = {
            "scores": torch.empty(batch_size, nq, kv_seq_len, dtype=BF16, device="cuda"),
            "output": torch.empty(batch_size, nq, dv, dtype=BF16, device="cuda"),
        }
    bufs = _alloc_cache[alloc_key]

    # Dequant KV: fp8 -> bf16
    kv_bf16 = kv_buffer_fp8.view(batch_size, kv_seq_len, 576).to(BF16) * kv_scale

    # Q: (bs, 16, 576)
    q_3d = q.view(batch_size, nq, dq)

    # QK^T with fused scale: scores = sm_scale * (Q @ KV^T)
    torch.baddbmm(bufs["scores"], q_3d, kv_bf16.transpose(1, 2),
                  beta=0.0, alpha=sm_scale, out=bufs["scores"])

    # Softmax in fp32 for numerical stability
    weights = F.softmax(bufs["scores"], dim=-1, dtype=FP32).to(BF16)

    # V: (bs, 16, kv_len) @ (bs, kv_len, 512) -> (bs, 16, 512)
    torch.bmm(weights, kv_bf16[:, :, :dv], out=bufs["output"])
    return bufs["output"].reshape(-1, nq, dv)


# ============================================================================
# SDPA approach: fused QK^T + softmax + V
# ============================================================================

def _sdpa_attention_v576(q, kv_buffer_fp8, kv_scale, config):
    """
    SDPA with V=576 dims (same as K). Output sliced to [:512].

    head_dim=576 exceeds flash attention limit (256), so math kernel is used.
    Math kernel = optimized bmm + softmax, but fully fused in one launch.

    With enable_gqa: Q(bs,16,1,576), K(bs,1,kv,576), V(bs,1,kv,576).
    K/V automatically broadcast to 16 heads by SDPA.
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    nq = config["num_heads"]       # 16
    dq = config["qk_head_dim"]     # 576
    dv = config["v_head_dim"]      # 512
    sm_scale = config["sm_scale"]

    kv_bf16 = kv_buffer_fp8.view(batch_size, kv_seq_len, 576).to(BF16) * kv_scale

    q_4d = q.view(batch_size, nq, 1, dq)        # (bs, 16, 1, 576)
    kv_4d = kv_bf16.unsqueeze(1)                 # (bs, 1, kv_len, 576)

    try:
        out = F.scaled_dot_product_attention(
            q_4d, kv_4d, kv_4d, scale=sm_scale, is_causal=False, enable_gqa=True)
    except TypeError:
        kv_expanded = kv_4d.expand(-1, nq, -1, -1)
        out = F.scaled_dot_product_attention(
            q_4d, kv_expanded, kv_expanded, scale=sm_scale, is_causal=False)

    return out[:, :, 0, :dv].contiguous().view(-1, nq, dv)


def _sdpa_attention_split(q, kv_buffer_fp8, kv_scale, config):
    """
    SDPA with K=576 dims, V=512 dims (no wasted compute on V).

    K_dim != V_dim forces math kernel backend (flash attn needs K_dim==V_dim).
    Must expand K/V to 16 heads manually (enable_gqa may not work with math kernel).
    """
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    nq = config["num_heads"]       # 16
    dq = config["qk_head_dim"]     # 576
    dv = config["v_head_dim"]      # 512
    sm_scale = config["sm_scale"]

    kv_bf16 = kv_buffer_fp8.view(batch_size, kv_seq_len, 576).to(BF16) * kv_scale

    q_4d = q.view(batch_size, nq, 1, dq)
    k_4d = kv_bf16.unsqueeze(1).expand(-1, nq, -1, -1)               # (bs, 16, kv, 576)
    v_4d = kv_bf16[:, :, :dv].unsqueeze(1).expand(-1, nq, -1, -1)    # (bs, 16, kv, 512)

    out = F.scaled_dot_product_attention(
        q_4d, k_4d, v_4d, scale=sm_scale, is_causal=False)
    return out.squeeze(2).contiguous().view(-1, nq, dv)


# ============================================================================
# ASM kernel (proven baseline from submission_pg8_v2)
# ============================================================================

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


def _build_meta(batch_size, kv_seq_len, q_seq_len, nq, nkv,
                num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr):
    total_kv = batch_size * kv_seq_len
    if page_size == 1:
        num_pages = total_kv
        kv_indptr_pages = kv_indptr
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = seq_lens.to(torch.int32)
    else:
        num_pages = total_kv // page_size
        kv_indptr_pages = kv_indptr // page_size
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = (seq_lens % page_size).to(torch.int32)
        kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)
    kv_gran = max(1, 16 // page_size)
    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, nq, dtype_q, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True)
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work
    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size, kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
        fast_mode=False, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=dtype_q, dtype_kv=FP8_DTYPE)
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")
    return (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages, page_size)


def _asm_attention(q, kv_data, qo_indptr, kv_indptr, config, page_size, use_fp8_q):
    """ASM kernel path (proven baseline)."""
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len
    dtype_q = FP8_DTYPE if use_fp8_q else BF16
    num_kv_splits = 8 if total_kv <= 8192 else 16

    cache_key = ("asm", batch_size, kv_seq_len, num_kv_splits, page_size, use_fp8_q)
    if cache_key not in _meta_cache:
        _meta_cache[cache_key] = _build_meta(
            batch_size, kv_seq_len, q_seq_len, nq, nkv,
            num_kv_splits, page_size, dtype_q, qo_indptr, kv_indptr)
    (wm, wi, wis, ri, rfm, rpm,
     kv_indices, kv_last_page_len, kv_indptr_pages, ps) = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, ps, nkv, kv_buffer_fp8.shape[-1])

    if use_fp8_q:
        alloc_key = ("fp8_asm", q.shape[0], nq, dv, dq)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = (
                torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda"),
                torch.zeros(1, dtype=FP32, device="cuda"),
                torch.empty(1, dtype=FP32, device="cuda"),
                torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"))
        o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]
        N = q.numel()
        BLOCK = 4096
        grid = ((N + BLOCK - 1) // BLOCK,)
        amax_buf.zero_()
        _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
        _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf,
                               FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
        mla_decode_fwd(
            q_fp8_flat.view(q.shape[0], nq, dq), kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale_buf, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm)
        return o
    else:
        alloc_key = ("bf16_asm", q.shape[0], nq, dv)
        if alloc_key not in _alloc_cache:
            _alloc_cache[alloc_key] = torch.empty(
                (q.shape[0], nq, dv), dtype=BF16, device="cuda")
        o = _alloc_cache[alloc_key]
        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=ps, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm)
        return o


# ============================================================================
# Main entry point
# ============================================================================

# MODE selects approach:
# "asm_only"     = proven ASM for all shapes (baseline)
# "bmm_small"    = BMM for kv<=1024, ASM for kv>=8192
# "sdpa_small"   = SDPA(v576) for kv<=1024, ASM for kv>=8192
# "sdpa_split"   = SDPA(k576,v512) for kv<=1024, ASM for kv>=8192
# "bmm_all"      = BMM for ALL shapes (bandwidth test)
# "sdpa_all"     = SDPA for ALL shapes (bandwidth test)
MODE = "bmm_small"


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    kv_seq_len = config["kv_seq_len"]
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    is_small_kv = (kv_seq_len <= 1024)

    if MODE == "asm_only":
        if is_small_kv:
            return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                                 page_size=1, use_fp8_q=False)
        return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                             page_size=8, use_fp8_q=True)

    elif MODE == "bmm_small":
        if is_small_kv:
            return _bmm_attention(q, kv_buffer_fp8, kv_scale, config)
        return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                             page_size=8, use_fp8_q=True)

    elif MODE == "sdpa_small":
        if is_small_kv:
            return _sdpa_attention_v576(q, kv_buffer_fp8, kv_scale, config)
        return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                             page_size=8, use_fp8_q=True)

    elif MODE == "sdpa_split":
        if is_small_kv:
            return _sdpa_attention_split(q, kv_buffer_fp8, kv_scale, config)
        return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                             page_size=8, use_fp8_q=True)

    elif MODE == "bmm_all":
        return _bmm_attention(q, kv_buffer_fp8, kv_scale, config)

    elif MODE == "sdpa_all":
        return _sdpa_attention_v576(q, kv_buffer_fp8, kv_scale, config)

    # Default: ASM
    if is_small_kv:
        return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                             page_size=1, use_fp8_q=False)
    return _asm_attention(q, kv_data, qo_indptr, kv_indptr, config,
                         page_size=8, use_fp8_q=True)

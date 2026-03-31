#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Custom Triton Flash-Decoding Kernel for AMD MI355X (gfx950, CDNA4).

Two-stage flash-decoding bypassing aiter's 4-kernel pipeline.
Based on SGLang's decode_attention.py grouped kernel architecture.

Architecture:
  Stage 1: Split KV across thread blocks, compute partial softmax + weighted V
    - Grid: (batch, 1, num_kv_splits) -- 1 group covers all 16 heads via BLOCK_H=16
    - Each block: loads Q [16,512]+[16,64], iterates KV in chunks of BLOCK_N
    - Uses tl.dot for batched matmul: [16,512]x[512,BLOCK_N] -> [16,BLOCK_N]
    - Online softmax: track running max + exp sum per head
    - Stores: partial_out [16,512] + LSE [16] per split

  Stage 2: Reduce partial results across splits
    - Grid: (batch, 16) -- one thread block per (batch, head)
    - Loads all partial outputs + LSEs, merges via online softmax

Key AMD HIP optimizations (from SGLang):
  - BLOCK_N=16 for Lk>=576 (SGPR limit workaround on MI3xx)
  - num_stages=1 (not 2) on HIP
  - waves_per_eu=1 for stage1, =4 for stage2
  - matrix_instr_nonkdim=16, kpack=2
  - dim=576 split into 512+64 (both power-of-2) for tl.dot

Data flow:
  Q: (bs, 16, 576) bf16 -- no fp8 quant needed (saves 2 kernel launches)
  KV: (total_kv, 1, 576) fp8 + scalar scale
  Output: (bs, 16, 512) bf16
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

# Pre-allocated tensor caches
_alloc_cache = {}


# ============================================================================
# Stage 1: Grouped Flash-Decoding Attention
# ============================================================================
# Grid: (batch_size, 1, MAX_KV_SPLITS)
# All 16 query heads processed in one thread block (BLOCK_H=16, kv_group=16)
# ============================================================================

@triton.jit
def _mla_stage1(
    Q,              # (bs, 16, 576) bf16 -- total_q = bs for decode
    KV,             # (total_kv, 576) fp8 -- flattened from (total_kv, 1, 576)
    KV_SCALE,       # (1,) float32 -- per-tensor scale
    Att_Out,        # (bs, 16, max_splits, 512) float32 -- partial output
    Att_Lse,        # (bs, 16, max_splits) float32 -- log-sum-exp
    kv_indptr,      # (bs+1,) int32 -- KV boundaries per batch
    sm_scale,       # float32 -- 1/sqrt(576)
    # Strides
    stride_q_b,     # Q batch stride
    stride_q_h,     # Q head stride
    stride_kv_t,    # KV token stride (=576)
    stride_ao_b,    # Att_Out batch stride
    stride_ao_h,    # Att_Out head stride
    stride_ao_s,    # Att_Out split stride
    stride_lse_b,   # Att_Lse batch stride
    stride_lse_h,   # Att_Lse head stride
    # Compile-time constants
    BLOCK_DMODEL: tl.constexpr,  # 512
    BLOCK_DPE: tl.constexpr,     # 64
    BLOCK_DV: tl.constexpr,      # 512
    BLOCK_N: tl.constexpr,       # 16 (KV tokens per iteration)
    BLOCK_H: tl.constexpr,       # 16 (all query heads)
    MAX_KV_SPLITS: tl.constexpr,
    Lk: tl.constexpr,            # 576
    Lv: tl.constexpr,            # 512
):
    cur_batch = tl.program_id(0)
    split_kv_id = tl.program_id(2)

    # Head indices [0..15]
    cur_head = tl.arange(0, BLOCK_H)
    mask_h = cur_head < BLOCK_H  # all True when BLOCK_H == num_heads

    # KV boundaries for this batch element
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    kv_len = kv_end - kv_start

    # Compute this split's KV range
    # Round up to BLOCK_N alignment (MIN_BLOCK_KV equivalent)
    kv_per_split = tl.cdiv(tl.cdiv(kv_len, MAX_KV_SPLITS), BLOCK_N) * BLOCK_N
    split_start = kv_per_split * split_kv_id
    split_end = tl.minimum(split_start + kv_per_split, kv_len)

    # Dimension offsets
    offs_d = tl.arange(0, BLOCK_DMODEL)                    # [0..511]
    offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)      # [512..575]
    offs_dv = tl.arange(0, BLOCK_DV)                        # [0..511]

    # Online softmax state (per head)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_end > split_start:
        # Load Q main part: [BLOCK_H, BLOCK_DMODEL] = [16, 512]
        offs_q = (
            cur_batch * stride_q_b
            + cur_head[:, None] * stride_q_h
            + offs_d[None, :]
        )
        q = tl.load(Q + offs_q, mask=mask_h[:, None], other=0.0)

        # Load Q PE part: [BLOCK_H, BLOCK_DPE] = [16, 64]
        offs_qpe = (
            cur_batch * stride_q_b
            + cur_head[:, None] * stride_q_h
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q + offs_qpe, mask=mask_h[:, None], other=0.0)

        # Load KV scale once
        kv_scale_val = tl.load(KV_SCALE)

        # Iterate over KV tokens in this split
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_mask_n = offs_n < split_end

            # Absolute KV token indices
            kv_ids = kv_start + offs_n

            # Load K main: [BLOCK_DMODEL, BLOCK_N] = [512, 16] (transposed layout)
            offs_k = kv_ids[None, :] * stride_kv_t + offs_d[:, None]
            k = tl.load(
                KV + offs_k,
                mask=kv_mask_n[None, :],
                other=0.0,
            )

            # Load K PE: [BLOCK_DPE, BLOCK_N] = [64, 16]
            offs_kpe = kv_ids[None, :] * stride_kv_t + offs_dpe[:, None]
            kpe = tl.load(
                KV + offs_kpe,
                mask=kv_mask_n[None, :],
                other=0.0,
            )

            # QK^T = Q @ K^T: [16,512] x [512,16] -> [16,16]
            # Cast K (fp8) to bf16 for MFMA matmul. q is already bf16.
            qk = tl.dot(q, k.to(q.dtype))
            # Add PE contribution: [16,64] x [64,16] -> [16,16]
            qk += tl.dot(qpe, kpe.to(qpe.dtype))

            # Apply scale: QK = (Q_bf16 @ K_fp8_as_bf16) * kv_scale * sm_scale
            qk = qk.to(tl.float32) * (sm_scale * kv_scale_val)

            # Mask invalid positions
            qk = tl.where(kv_mask_n[None, :], qk, float("-inf"))

            # Load V: [BLOCK_N, BLOCK_DV] = [16, 512]
            # V = first 512 dims of KV (same data as K_main but in row-major layout)
            offs_v = kv_ids[:, None] * stride_kv_t + offs_dv[None, :]
            v = tl.load(
                KV + offs_v,
                mask=kv_mask_n[:, None],
                other=0.0,
            )

            # Online softmax update
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])

            # Rescale accumulator and add: acc = acc * rescale + P @ V
            acc = acc * re_scale[:, None]
            # P @ V: [16, BLOCK_N] x [BLOCK_N, 512] -> [16, 512]
            # Cast V (fp8) to bf16 for MFMA, scale by kv_scale after
            pv = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16))
            acc = acc + pv.to(tl.float32) * kv_scale_val

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

    # Store partial output: normalized by e_sum
    offs_out = (
        cur_batch * stride_ao_b
        + cur_head[:, None] * stride_ao_h
        + split_kv_id * stride_ao_s
        + offs_dv[None, :]
    )
    # Avoid division by zero for empty splits
    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    tl.store(
        Att_Out + offs_out,
        acc / safe_sum[:, None],
        mask=mask_h[:, None],
    )

    # Store LSE: e_max + log(e_sum)
    offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_kv_id
    lse = tl.where(e_sum > 0.0, e_max + tl.log(e_sum), float("-inf"))
    tl.store(Att_Lse + offs_lse, lse, mask=mask_h)


# ============================================================================
# Stage 2: Reduce across KV splits
# ============================================================================
# Grid: (batch_size, 16)
# One thread block per (batch, head) -- reduces all split partial results
# ============================================================================

@triton.jit
def _mla_stage2(
    Att_Out,        # (bs, 16, max_splits, 512) float32
    Att_Lse,        # (bs, 16, max_splits) float32
    O,              # (bs, 16, 512) bf16
    kv_indptr,      # (bs+1,) int32
    # Strides
    stride_ao_b,
    stride_ao_h,
    stride_ao_s,
    stride_lse_b,
    stride_lse_h,
    stride_o_b,
    stride_o_h,
    # Constants
    BLOCK_DV: tl.constexpr,      # 512
    BLOCK_N: tl.constexpr,       # same BLOCK_N as stage1
    MAX_KV_SPLITS: tl.constexpr,
    Lv: tl.constexpr,            # 512
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    offs_dv = tl.arange(0, BLOCK_DV)

    # Reduction state
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    # Compute split validity (same formula as stage1)
    kv_start = tl.load(kv_indptr + cur_batch)
    kv_end = tl.load(kv_indptr + cur_batch + 1)
    kv_len = kv_end - kv_start
    kv_per_split = tl.cdiv(tl.cdiv(kv_len, MAX_KV_SPLITS), BLOCK_N) * BLOCK_N

    for split_id in range(MAX_KV_SPLITS):
        s_start = kv_per_split * split_id
        s_end = tl.minimum(s_start + kv_per_split, kv_len)

        if s_end > s_start:
            # Load partial output
            offs_v = (
                cur_batch * stride_ao_b
                + cur_head * stride_ao_h
                + split_id * stride_ao_s
                + offs_dv
            )
            tv = tl.load(Att_Out + offs_v)

            # Load LSE
            offs_lse = cur_batch * stride_lse_b + cur_head * stride_lse_h + split_id
            tlogic = tl.load(Att_Lse + offs_lse)

            # Online softmax merge
            n_e_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp(e_max - n_e_max)
            exp_logic = tl.exp(tlogic - n_e_max)

            acc = acc * old_scale + exp_logic * tv
            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    # Final normalize and store
    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    result = acc / safe_sum

    offs_o = cur_batch * stride_o_b + cur_head * stride_o_h + offs_dv
    tl.store(O + offs_o, result.to(tl.bfloat16))


# ============================================================================
# Configuration tuning
# ============================================================================

def _get_config(batch_size, kv_seq_len):
    """Return (num_splits, BLOCK_N) tuned for MI355X 304 CUs.

    CRITICAL: We use a SINGLE num_splits value (16) for all shapes to avoid
    multiple Triton JIT compilations. Each unique (BLOCK_N, MAX_KV_SPLITS)
    combination triggers a separate compilation (10-30s each). With 8 benchmark
    shapes and 900s timeout, we cannot afford many compilations.

    Using 16 splits for all shapes:
      bs=4,   kv=1024: 4*16=64 programs,  64 tokens/split, 4 iterations/split
      bs=4,   kv=8192: 4*16=64 programs,  512 tokens/split, 32 iterations/split
      bs=32,  kv=1024: 32*16=512 programs, 64 tokens/split, 4 iterations/split
      bs=32,  kv=8192: 32*16=512 programs, 512 tokens/split, 32 iterations/split
      bs=64,  kv=1024: 64*16=1024 programs, 64 tokens/split, 4 iterations/split
      bs=64,  kv=8192: 64*16=1024 programs, 512 tokens/split, 32 iterations/split
      bs=256, kv=1024: 256*16=4096 programs, 64 tokens/split, 4 iterations/split
      bs=256, kv=8192: 256*16=4096 programs, 512 tokens/split, 32 iterations/split

    This covers 304 CUs well for all shapes (>= 64 programs).
    """
    BLOCK_N = 16
    num_splits = 16
    return num_splits, BLOCK_N


# ============================================================================
# Entry point
# ============================================================================

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq = config["num_heads"]        # 16
    dq = config["qk_head_dim"]      # 576
    dv = config["v_head_dim"]       # 512
    sm_scale = config["sm_scale"]   # 1/sqrt(576)
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # Use fp8 KV cache (1 byte/elem vs 2 for bf16 -- 2x bandwidth savings)
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Flatten: (total_kv, 1, 576) -> (total_kv, 576)
    kv_flat = kv_buffer_fp8.view(total_kv, dq)

    # Tuned configuration
    num_splits, BLOCK_N = _get_config(batch_size, kv_seq_len)

    BLOCK_DMODEL = 512
    BLOCK_DPE = 64
    BLOCK_DV = 512
    BLOCK_H = 16

    # Allocate (cached across calls with same shape)
    alloc_key = (batch_size, nq, num_splits, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = (
            torch.empty((batch_size, nq, num_splits, dv), dtype=torch.float32, device="cuda"),
            torch.empty((batch_size, nq, num_splits), dtype=torch.float32, device="cuda"),
            torch.empty((batch_size, nq, dv), dtype=torch.bfloat16, device="cuda"),
        )
    att_out, att_lse, o = _alloc_cache[alloc_key]

    # ---- Stage 1 ----
    grid_s1 = (batch_size, 1, num_splits)
    _mla_stage1[grid_s1](
        q, kv_flat, kv_scale,
        att_out, att_lse, kv_indptr,
        sm_scale,
        # Strides
        q.stride(0), q.stride(1),
        kv_flat.stride(0),
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        att_lse.stride(0), att_lse.stride(1),
        # Constants
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        BLOCK_H=BLOCK_H,
        MAX_KV_SPLITS=num_splits,
        Lk=dq,
        Lv=dv,
        # AMD HIP tuning (from SGLang)
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
        matrix_instr_nonkdim=16,
        kpack=2,
    )

    # ---- Stage 2 ----
    grid_s2 = (batch_size, nq)
    _mla_stage2[grid_s2](
        att_out, att_lse, o, kv_indptr,
        # Strides
        att_out.stride(0), att_out.stride(1), att_out.stride(2),
        att_lse.stride(0), att_lse.stride(1),
        o.stride(0), o.stride(1),
        # Constants
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK_N,
        MAX_KV_SPLITS=num_splits,
        Lv=dv,
        # AMD HIP tuning
        num_warps=4,
        num_stages=1,
        waves_per_eu=4,
        matrix_instr_nonkdim=16,
        kpack=2,
    )

    return o

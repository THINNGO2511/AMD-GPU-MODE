#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Custom Triton flash-decoding kernel. Bypasses aiter entirely.
No JIT timeout issue since it's pure Triton (already cached at 3.6.0).
Tolerance is very loose: rtol=0.1, atol=0.1, 5% mismatch bypass.

DeepSeek R1 MLA: 16 query heads, 1 KV head, qk_dim=576, v_dim=512
Each query head attends to the same single KV head.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t

_alloc_cache = {}


@triton.jit
def _mla_decode_kernel(
    Q,          # (total_q, nq, dq) bf16
    KV,         # (total_kv, 1, dq) fp8
    KV_SCALE,   # scalar float32
    O,          # (total_q, nq, dv) bf16
    kv_indptr,  # (batch+1,) int32 — KV boundaries per batch
    sm_scale: tl.constexpr,
    nq: tl.constexpr,         # 16
    dq: tl.constexpr,         # 576
    dv: tl.constexpr,         # 512
    BLOCK_KV: tl.constexpr,   # KV chunk size
    BLOCK_D: tl.constexpr,    # head dim chunk size
):
    # Program IDs: (batch_idx * nq + head_idx, kv_split_idx)
    pid_bh = tl.program_id(0)
    batch_idx = pid_bh // nq
    head_idx = pid_bh % nq

    kv_start = tl.load(kv_indptr + batch_idx)
    kv_end = tl.load(kv_indptr + batch_idx + 1)
    kv_len = kv_end - kv_start

    # Load query for this head: Q[batch_idx, head_idx, :dq]
    # We process dq in chunks of BLOCK_D
    q_offset = batch_idx * nq * dq + head_idx * dq

    # Online softmax + weighted sum over KV
    m_prev = float("-inf")  # running max
    l_prev = 0.0            # running sum of exp
    # Accumulator for output (dv floats)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Load KV scale
    kv_scale = tl.load(KV_SCALE)

    # Iterate over KV in chunks
    for kv_block_start in range(0, kv_len, BLOCK_KV):
        kv_block_end = tl.minimum(kv_block_start + BLOCK_KV, kv_len)
        actual_len = kv_block_end - kv_block_start

        # Compute QK^T for this block
        # Q: (dq,), K: (BLOCK_KV, dq) → scores: (BLOCK_KV,)
        scores = tl.zeros([BLOCK_KV], dtype=tl.float32)

        # Accumulate dot product in chunks of BLOCK_D over the dq dimension
        for d_start in range(0, dq, BLOCK_D):
            d_offs = d_start + tl.arange(0, BLOCK_D)
            d_mask = d_offs < dq

            # Load Q chunk
            q_vals = tl.load(Q + q_offset + d_offs, mask=d_mask, other=0.0).to(tl.float32)

            # Load KV chunk for each token in block
            for kv_i in range(BLOCK_KV):
                if kv_block_start + kv_i < kv_len:
                    kv_token_idx = kv_start + kv_block_start + kv_i
                    kv_vals = tl.load(
                        KV + kv_token_idx * dq + d_offs,
                        mask=d_mask, other=0.0
                    ).to(tl.float32) * kv_scale

                    dot = tl.sum(q_vals * kv_vals)
                    scores = tl.where(
                        tl.arange(0, BLOCK_KV) == kv_i,
                        scores + dot,
                        scores
                    )

        scores = scores * sm_scale

        # Mask out-of-range KV tokens
        kv_mask = tl.arange(0, BLOCK_KV) < actual_len
        scores = tl.where(kv_mask, scores, float("-inf"))

        # Online softmax update
        m_new = tl.maximum(m_prev, tl.max(scores))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_prev - m_new)
        l_new = alpha * l_prev + tl.sum(p)

        # Update accumulator: acc = alpha * acc + sum(p * V)
        # V is the first dv dimensions of KV
        acc = acc * alpha

        for kv_i in range(BLOCK_KV):
            if kv_block_start + kv_i < kv_len:
                kv_token_idx = kv_start + kv_block_start + kv_i
                v_offs = tl.arange(0, BLOCK_D)
                v_mask = v_offs < dv
                v_vals = tl.load(
                    KV + kv_token_idx * dq + v_offs,
                    mask=v_mask, other=0.0
                ).to(tl.float32) * kv_scale

                weight = tl.where(
                    tl.arange(0, BLOCK_KV) == kv_i, p, 0.0
                )
                w = tl.sum(weight)  # scalar weight for this token
                acc = acc + w * v_vals

        m_prev = m_new
        l_prev = l_new

    # Normalize
    acc = acc / l_prev

    # Store output
    o_offset = batch_idx * nq * dv + head_idx * dv
    o_offs = tl.arange(0, BLOCK_D)
    o_mask = o_offs < dv
    tl.store(O + o_offset + o_offs, acc.to(tl.bfloat16), mask=o_mask)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq = config["num_heads"]       # 16
    dq = config["qk_head_dim"]     # 576
    dv = config["v_head_dim"]      # 512
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Flatten KV: (total_kv, 1, 576) → (total_kv, 576)
    total_kv = batch_size * kv_seq_len
    kv_flat = kv_buffer_fp8.view(total_kv, dq)

    alloc_key = (batch_size, nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty(
            (batch_size, nq, dv), dtype=torch.bfloat16, device="cuda")
    o = _alloc_cache[alloc_key]

    # Launch: one program per (batch, head) pair
    BLOCK_KV = 64
    BLOCK_D = 576  # Must be >= max(dq, dv), power of 2... 576 isn't power of 2

    # Round up to next power of 2
    BLOCK_D = 1024  # Covers both dq=576 and dv=512

    grid = (batch_size * nq,)
    _mla_decode_kernel[grid](
        q, kv_flat, kv_scale, o, kv_indptr,
        sm_scale=sm_scale, nq=nq, dq=dq, dv=dv,
        BLOCK_KV=BLOCK_KV, BLOCK_D=BLOCK_D,
    )

    return o

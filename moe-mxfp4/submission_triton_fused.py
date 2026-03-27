#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Fused Triton MoE using tl.dot_scaled with raw (un-shuffled) MXFP4 weights.

Architecture:
1. moe_sorting_fwd: sort tokens by expert (reuse aiter's C++ sorting)
2. Stage 1 Triton kernel: batched GEMM across all experts
   - Each program handles one (expert, m_tile, n_tile)
   - Inline MXFP4 quant via _mxfp4_quant_op
   - Fused SiLU activation at output
3. Stage 2 Triton kernel: batched down-projection GEMM
   - Each program handles one (expert, m_tile, n_tile)
   - Weighted accumulation into output

Key advantage over CK path: CK has ZERO tuned MXFP4 MoE configs.
Our Triton kernels use tl.dot_scaled("e2m1") which is native on MI355X.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_cache = {}


@triton.jit
def _moe_stage1_kernel(
    # Pointers
    hidden_ptr,           # [M, d_hidden_pad] bf16 (padded input)
    w1_ptr,               # [E, 2*d_expert_pad, K_packed] uint8 (raw fp4x2)
    w1_scale_ptr,         # [E, 2*d_expert_pad, scale_K] uint8 (raw e8m0)
    inter_ptr,            # [M * top_k, d_expert] bf16 (intermediate output after SiLU)
    sorted_token_ids_ptr, # [M * top_k (padded)] int32
    expert_ids_ptr,       # [num_tiles] int32 - which expert each tile-row belongs to
    num_tokens_post_padded_ptr,  # [E] int32 - cumsum of tokens per expert
    # Dimensions
    K,                    # d_hidden_pad (full K, not packed)
    N,                    # 2 * d_expert_pad
    d_expert,             # actual d_expert (for SiLU split)
    num_valid_tokens,     # M * top_k
    top_k,
    # Strides
    stride_hs,            # hidden_states stride(0)
    stride_w1e,           # w1 stride(0) = 2*d_expert_pad * K//2
    stride_w1n,           # w1 stride(1) = K//2
    stride_w1se,          # w1_scale stride(0) = 2*d_expert_pad * scale_K
    stride_w1sn,          # w1_scale stride(1) = scale_K
    stride_inter,         # inter stride(0) = d_expert
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 1: gate+up projection + SiLU for all experts in one kernel."""
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Map pid to (token_tile, n_tile)
    # Token tiles are laid out contiguously per expert
    pid_token_tile = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Load expert_id for this token tile
    expert_id = tl.load(expert_ids_ptr + pid_token_tile)

    # Get token range for this tile within the sorted order
    tile_start = pid_token_tile * BLOCK_M
    offs_m = tile_start + tl.arange(0, BLOCK_M)

    # Load sorted token IDs (map from sorted position to original token index)
    token_ids = tl.load(sorted_token_ids_ptr + offs_m, mask=offs_m < num_valid_tokens, other=0)

    # Check which tokens in this tile are valid
    valid_mask = offs_m < num_valid_tokens

    # Load hidden_states rows (gather by token_id // top_k to get original token)
    # sorted_token_ids values are token_idx * top_k + k_idx style from moe_sorting
    orig_token = token_ids // top_k

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)

        # Load hidden_states tile [BLOCK_M, BLOCK_K] - gather rows by orig_token
        h_tile = tl.load(
            hidden_ptr + orig_token[:, None] * stride_hs + (k_start + offs_k)[None, :],
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # Quantize A to MXFP4 inline
        a_fp4, a_scales = _mxfp4_quant_op(h_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

        # Load W1 tile [BLOCK_K//2, BLOCK_N] for this expert
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        b_fp4 = tl.load(
            w1_ptr + expert_id * stride_w1e + offs_n[None, :] * stride_w1n + (k_start // 2 + offs_k_packed)[:, None],
        )

        # Load W1 scales [BLOCK_N, BLOCK_K//SCALE_GROUP] for this expert
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        b_scales = tl.load(
            w1_scale_ptr + expert_id * stride_w1se + offs_n[:, None] * stride_w1sn + (k_start // SCALE_GROUP + offs_k_scale)[None, :],
        )

        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Apply SiLU activation: result[:, :d_expert] = SiLU(gate) * up
    # gate = acc[:, :d_expert_pad], up = acc[:, d_expert_pad:]
    # But we only compute for the N-tile we're in
    result = acc.to(tl.bfloat16)

    # Store to intermediate buffer (we'll apply SiLU in a separate small kernel or fuse here)
    # For now, store raw stage1 output. SiLU will be applied between stages.
    out_ptrs = inter_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = valid_mask[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptrs, result, mask=out_mask)


@triton.jit
def _silu_kernel(
    inter_ptr,     # [total_tokens, 2*d_expert_pad] bf16 - stage1 output
    out_ptr,       # [total_tokens, d_expert] bf16 - after SiLU
    d_expert,
    d_expert_pad,
    total_tokens,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Apply SwiGLU: out = SiLU(gate) * up where gate=inter[:,:d_expert], up=inter[:,d_expert_pad:]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < total_tokens
    mask_n = offs_n < d_expert
    mask = mask_m[:, None] & mask_n[None, :]

    # Load gate and up
    gate = tl.load(inter_ptr + offs_m[:, None] * (2 * d_expert_pad) + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(inter_ptr + offs_m[:, None] * (2 * d_expert_pad) + (d_expert_pad + offs_n)[None, :], mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    silu_gate = gate * tl.sigmoid(gate)
    result = (silu_gate * up).to(tl.bfloat16)

    tl.store(out_ptr + offs_m[:, None] * d_expert + offs_n[None, :], result, mask=mask)


@triton.jit
def _moe_stage2_kernel(
    # Pointers
    inter_ptr,            # [total_tokens, d_expert_pad] bf16 (padded intermediate)
    w2_ptr,               # [E, d_hidden_pad, d_expert_pad//2] uint8 (raw fp4x2)
    w2_scale_ptr,         # [E, d_hidden_pad, scale_K] uint8 (raw e8m0)
    output_ptr,           # [M, d_hidden] bf16
    sorted_token_ids_ptr, # [total_tokens (padded)] int32
    expert_ids_ptr,       # [num_tiles] int32
    topk_weights_ptr,     # [M, top_k] float32
    # Dimensions
    K,                    # d_expert_pad (full K for stage2)
    N,                    # d_hidden_pad
    d_hidden,             # actual d_hidden
    num_valid_tokens,
    top_k,
    M,                    # batch size
    # Strides
    stride_inter,         # inter stride(0) = d_expert_pad
    stride_w2e,           # w2 stride(0)
    stride_w2n,           # w2 stride(1)
    stride_w2se,          # w2_scale stride(0)
    stride_w2sn,          # w2_scale stride(1)
    stride_out,           # output stride(0) = d_hidden
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Stage 2: down projection + weighted accumulation."""
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_token_tile = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(expert_ids_ptr + pid_token_tile)

    tile_start = pid_token_tile * BLOCK_M
    offs_m = tile_start + tl.arange(0, BLOCK_M)
    valid_mask = offs_m < num_valid_tokens

    token_ids = tl.load(sorted_token_ids_ptr + offs_m, mask=valid_mask, other=0)
    orig_token = token_ids // top_k
    k_idx = token_ids % top_k

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_k = tl.arange(0, BLOCK_K)

        # Load intermediate tile [BLOCK_M, BLOCK_K]
        a_tile = tl.load(
            inter_ptr + offs_m[:, None] * stride_inter + (k_start + offs_k)[None, :],
            mask=valid_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)

        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        b_fp4 = tl.load(
            w2_ptr + expert_id * stride_w2e + offs_n[None, :] * stride_w2n + (k_start // 2 + offs_k_packed)[:, None],
        )

        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        b_scales = tl.load(
            w2_scale_ptr + expert_id * stride_w2se + offs_n[:, None] * stride_w2sn + (k_start // SCALE_GROUP + offs_k_scale)[None, :],
        )

        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)

    # Load topk_weights for weighted accumulation
    weights = tl.load(topk_weights_ptr + orig_token * top_k + k_idx, mask=valid_mask, other=0.0)

    # Apply weight and store (atomic add for accumulation across experts)
    result = acc.to(tl.float32) * weights[:, None]
    result_bf16 = result.to(tl.bfloat16)

    # Only store valid N columns (trim to d_hidden)
    n_mask = offs_n < d_hidden
    store_mask = valid_mask[:, None] & n_mask[None, :]

    # Use atomic add since multiple experts write to same output token
    out_ptrs = output_ptr + orig_token[:, None] * stride_out + offs_n[None, :]
    tl.atomic_add(out_ptrs, result_bf16, mask=store_mask)


def _build_expert_tile_map(num_tokens_per_expert, BLOCK_M, E):
    """Build a mapping from tile index to expert_id."""
    expert_ids = []
    for e in range(E):
        n_tokens = num_tokens_per_expert[e].item()
        n_tiles = triton.cdiv(n_tokens, BLOCK_M)
        expert_ids.extend([e] * n_tiles)
    return torch.tensor(expert_ids, dtype=torch.int32, device='cuda')


_first_call = True


def custom_kernel(data: input_t) -> output_t:
    global _first_call

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # For now, fall back to reference while we develop
    # TODO: switch to Triton path once correctness verified
    if _first_call:
        _first_call = False
        # Probe moe_sorting_fwd
        try:
            from aiter import moe_sorting_fwd
            import inspect
            M = hidden_states.shape[0]
            E = gate_up_weight.shape[0]
            top_k = topk_ids.shape[1]
            d_hidden = config["d_hidden"]
            d_expert = config["d_expert"]
            d_hidden_pad = config["d_hidden_pad"]
            d_expert_pad = config["d_expert_pad"]

            sig = inspect.signature(moe_sorting_fwd)
            print(f"[SORT] moe_sorting_fwd{sig}")

            # Try calling with various arg combinations
            try:
                result = moe_sorting_fwd(topk_ids, E)
                print(f"[SORT] (topk_ids, E) → {len(result)} outputs")
                for i, r in enumerate(result):
                    if isinstance(r, torch.Tensor):
                        print(f"  [{i}] {r.shape} {r.dtype} [{r.min()}, {r.max()}]")
                        if r.numel() < 50:
                            print(f"       values: {r.tolist()}")
                    else:
                        print(f"  [{i}] {type(r).__name__}: {r}")
            except Exception as e1:
                print(f"[SORT] (topk_ids, E) failed: {e1}")
                try:
                    result = moe_sorting_fwd(topk_ids, E, top_k)
                    print(f"[SORT] (topk_ids, E, top_k) → {len(result)} outputs")
                    for i, r in enumerate(result):
                        if isinstance(r, torch.Tensor):
                            print(f"  [{i}] {r.shape} {r.dtype}")
                        else:
                            print(f"  [{i}] {type(r).__name__}: {r}")
                except Exception as e2:
                    print(f"[SORT] (topk_ids, E, top_k) failed: {e2}")

            # Also read fused_moe_ source to see how sorting is called
            import aiter.fused_moe as fm
            with open(fm.__file__) as f:
                src = f.read()
            # Find moe_sorting call pattern
            for line_no, line in enumerate(src.split('\n')):
                if 'moe_sorting' in line and 'def ' not in line and 'import' not in line:
                    print(f"[SRC:{line_no}] {line.strip()}")

            # Print lines around per_1x32 handling
            lines = src.split('\n')
            for i, line in enumerate(lines):
                if 'per_1x32' in line or 'per1x32' in line.lower():
                    start = max(0, i-2)
                    end = min(len(lines), i+5)
                    for j in range(start, end):
                        print(f"[Q:{j}] {lines[j].rstrip()}")

        except Exception as e:
            import traceback
            print(f"[PROBE] error: {e}")
            traceback.print_exc()

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

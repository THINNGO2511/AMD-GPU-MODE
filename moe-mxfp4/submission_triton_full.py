#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Full Triton MoE pipeline: stage1 + SiLU + stage2 + weighted reduce.
All stages use tl.dot_scaled with raw (un-shuffled) MXFP4 weights.
Uses moe_align_block_size for per-block expert IDs.
"""
import torch
import triton
import triton.language as tl
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.ops.triton.quant import dynamic_mxfp4_quant

_cache = {}


@triton.jit
def _triton_moe_stage1(
    a_ptr, stride_am, stride_ak,
    as_ptr, stride_asm, stride_ask,
    w_ptr, stride_we, stride_wn, stride_wk,
    ws_ptr, stride_wse, stride_wsn, stride_wsk,
    out_ptr, stride_om, stride_on,
    sorted_ids_ptr, expert_ids_ptr,
    N, K, num_valid, top_k: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Stage1: gate+up projection. A gathered from original order."""
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_ids_ptr + offs_block)
    token_mask = token_ids < num_valid
    # Clamp orig_token to prevent OOB A reads
    orig_token = tl.where(token_mask, token_ids // top_k, 0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_kp = tl.arange(0, BLOCK_K // 2)
        offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)

        a = tl.load(a_ptr + orig_token[:, None] * stride_am + (k_start // 2 + offs_kp)[None, :] * stride_ak,
                     mask=token_mask[:, None], other=0)
        a_s = tl.load(as_ptr + orig_token[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_ks)[None, :] * stride_ask,
                       mask=token_mask[:, None], other=0)
        w = tl.load(w_ptr + expert_id * stride_we + (k_start // 2 + offs_kp)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_s = tl.load(ws_ptr + expert_id * stride_wse + offs_n[:, None] * stride_wsn + (k_start // SCALE_GROUP + offs_ks)[None, :] * stride_wsk)

        acc = tl.dot_scaled(a, a_s, "e2m1", w, w_s, "e2m1", acc)

    result = acc.to(tl.bfloat16)
    out_mask = token_mask[:, None] & (offs_n[None, :] < N)
    # Clamp token_ids for store to prevent OOB
    safe_ids = tl.where(token_mask, token_ids, 0)
    tl.store(out_ptr + safe_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
             result, mask=out_mask)


@triton.jit
def _silu_kernel(
    inp_ptr, out_ptr,
    d_expert, d_expert_pad, num_tokens,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """SwiGLU: out[m,n] = SiLU(gate[m,n]) * up[m,n] where gate=inp[:,:d_expert_pad], up=inp[:,d_expert_pad:]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < num_tokens) & (offs_n[None, :] < d_expert)

    gate = tl.load(inp_ptr + offs_m[:, None] * 2 * d_expert_pad + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
    up = tl.load(inp_ptr + offs_m[:, None] * 2 * d_expert_pad + (d_expert_pad + offs_n)[None, :], mask=mask, other=0.0).to(tl.float32)

    silu = gate * tl.sigmoid(gate)
    result = (silu * up).to(tl.bfloat16)
    tl.store(out_ptr + offs_m[:, None] * d_expert + offs_n[None, :], result, mask=mask)


@triton.jit
def _triton_moe_stage2(
    a_ptr, stride_am, stride_ak,
    as_ptr, stride_asm, stride_ask,
    w_ptr, stride_we, stride_wn, stride_wk,
    ws_ptr, stride_wse, stride_wsn, stride_wsk,
    out_ptr, stride_om, stride_on,
    sorted_ids_ptr, expert_ids_ptr,
    weights_ptr,  # topk_weights flattened [M*topk]
    N, K, num_valid, top_k: tl.constexpr, M,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Stage2: down projection + weighted accumulation into output."""
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    expert_id = tl.load(expert_ids_ptr + pid_m)
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_ids_ptr + offs_block)
    token_mask = token_ids < num_valid
    # Clamp orig_token to prevent OOB output writes
    orig_token = tl.where(token_mask, token_ids // top_k, 0)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    safe_tids = tl.where(token_mask, token_ids, 0)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_kp = tl.arange(0, BLOCK_K // 2)
        offs_ks = tl.arange(0, BLOCK_K // SCALE_GROUP)

        # A is intermediate indexed by (clamped) token_ids
        a = tl.load(a_ptr + safe_tids[:, None] * stride_am + (k_start // 2 + offs_kp)[None, :] * stride_ak,
                     mask=token_mask[:, None], other=0)
        a_s = tl.load(as_ptr + safe_tids[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_ks)[None, :] * stride_ask,
                       mask=token_mask[:, None], other=0)

        w = tl.load(w_ptr + expert_id * stride_we + (k_start // 2 + offs_kp)[:, None] * stride_wk + offs_n[None, :] * stride_wn)
        w_s = tl.load(ws_ptr + expert_id * stride_wse + offs_n[:, None] * stride_wsn + (k_start // SCALE_GROUP + offs_ks)[None, :] * stride_wsk)

        acc = tl.dot_scaled(a, a_s, "e2m1", w, w_s, "e2m1", acc)

    # Apply topk_weight (use clamped token_ids)
    weight = tl.load(weights_ptr + safe_tids, mask=token_mask, other=0.0)
    result = acc * weight[:, None]  # Keep float32

    n_mask = offs_n < N
    out_mask = token_mask[:, None] & n_mask[None, :]
    # Scatter-add to float32 output[orig_token, :]
    tl.atomic_add(out_ptr + orig_token[:, None] * stride_om + offs_n[None, :] * stride_on,
                  result.to(tl.float32), mask=out_mask)


def _triton_moe_full(data):
    """Full Triton MoE pipeline."""
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    M = hidden_states.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]
    BLOCK_M = 32

    # Cache setup tensors by shape
    cache_key = (M, E, topk, d_hidden_pad, d_expert_pad)
    if cache_key not in _cache:
        max_padded = M * topk + E * BLOCK_M
        _cache[cache_key] = {
            'sorted_ids': torch.empty(max_padded, dtype=torch.int32, device='cuda'),
            'expert_ids': torch.empty(max_padded // BLOCK_M, dtype=torch.int32, device='cuda'),
            'token_nums': torch.empty(E, dtype=torch.int32, device='cuda'),
            'ntp_tensor': torch.empty(1, dtype=torch.int32, device='cuda'),
        }
    c = _cache[cache_key]

    # 1. Sorting — use moe_align_block_size (correct per-block expert IDs)
    aiter.moe_align_block_size(topk_ids, E, BLOCK_M,
        c['sorted_ids'], c['expert_ids'], c['token_nums'], c['ntp_tensor'])
    ntp = c['ntp_tensor'].item()
    num_blocks = ntp // BLOCK_M
    num_valid = M * topk

    # 2. Quantize A (original order)
    a_fp4, a_scale = dynamic_mxfp4_quant(hidden_states)
    a_u8 = a_fp4.view(torch.uint8)
    as_u8 = a_scale.view(torch.uint8)

    # No need to pad A — orig_token is clamped to 0 for invalid tokens

    # 3. Stage 1: gate+up projection
    N1 = 2 * d_expert_pad
    K1 = d_hidden_pad
    w1 = gate_up_weight.view(torch.uint8)
    w1s = gate_up_weight_scale.view(torch.uint8).view(E, N1, K1 // 32)
    stage1_out = torch.zeros((num_valid, N1), dtype=torch.bfloat16, device='cuda')

    BLOCK_N, BLOCK_K = 128, 128
    grid1 = (num_blocks * triton.cdiv(N1, BLOCK_N),)
    _triton_moe_stage1[grid1](
        a_u8, a_u8.stride(0), a_u8.stride(1),
        as_u8, as_u8.stride(0), as_u8.stride(1),
        w1, w1.stride(0), w1.stride(1), w1.stride(2),
        w1s, w1s.stride(0), w1s.stride(1), w1s.stride(2),
        stage1_out, stage1_out.stride(0), stage1_out.stride(1),
        c['sorted_ids'], c['expert_ids'],
        N1, K1, num_valid, top_k=topk,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # 4. SiLU activation — applied to stage1_out[0:num_valid, :]
    # stage1_out layout: rows indexed by token_id (0..num_valid-1), cols = [gate|up]
    inter = torch.empty((num_valid, d_expert), dtype=torch.bfloat16, device='cuda')
    silu_grid = (triton.cdiv(num_valid, 32), triton.cdiv(d_expert, 128))
    _silu_kernel[silu_grid](stage1_out, inter, d_expert, d_expert_pad, num_valid, BLOCK_M=32, BLOCK_N=128)

    # Pad intermediate to d_expert_pad for quantization
    if d_expert_pad > d_expert:
        inter_padded = torch.zeros((num_valid, d_expert_pad), dtype=torch.bfloat16, device='cuda')
        inter_padded[:, :d_expert] = inter
    else:
        inter_padded = inter

    # 5. Quantize intermediate to MXFP4
    inter_fp4, inter_scale = dynamic_mxfp4_quant(inter_padded)
    inter_u8 = inter_fp4.view(torch.uint8)  # [num_valid, K2//2]
    inters_u8 = inter_scale.view(torch.uint8)  # [num_valid, K2//32]

    # No need to pad intermediate — token_ids clamped to 0 for invalid tokens

    # 6. Stage 2: down projection + weighted accumulation
    N2 = d_hidden_pad
    K2 = d_expert_pad
    w2 = down_weight.view(torch.uint8)
    w2s = down_weight_scale.view(torch.uint8).view(E, N2, K2 // 32)

    # Flatten topk_weights — token_ids clamped so no padding needed
    tw_flat = topk_weights.reshape(-1)  # [M*topk]

    # Output: pad to M + some extra for safety (invalid tokens masked anyway)
    output = torch.zeros((M + 1, d_hidden), dtype=torch.float32, device='cuda')
    grid2 = (num_blocks * triton.cdiv(d_hidden, BLOCK_N),)
    _triton_moe_stage2[grid2](
        inter_u8, inter_u8.stride(0), inter_u8.stride(1),
        inters_u8, inters_u8.stride(0), inters_u8.stride(1),
        w2, w2.stride(0), w2.stride(1), w2.stride(2),
        w2s, w2s.stride(0), w2s.stride(1), w2s.stride(2),
        output, output.stride(0), output.stride(1),
        c['sorted_ids'], c['expert_ids'],
        tw_flat,
        d_hidden, K2, num_valid, top_k=topk, M=M,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return output[:M, :d_hidden].to(torch.bfloat16)


_first = True


def custom_kernel(data: input_t) -> output_t:
    global _first
    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Use Triton for E=33 cases only (fewer experts = faster JIT + more impact)
    E = gate_up_weight.shape[0]
    if E <= 64:
        try:
            return _triton_moe_full(data)
        except:
            pass

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

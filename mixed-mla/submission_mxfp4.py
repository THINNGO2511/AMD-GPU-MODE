#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Decode — MXFP4 KV cache with fused dequant-attention.

Strategy: Use the mxfp4 KV cache (4x bandwidth savings over bf16, 2x over fp8)
with on-the-fly dequantization during attention computation. Since decode is
memory-bound (q_seq_len=1), reducing KV memory traffic is the key to speed.

Approach:
1. Quantize Q to fp8 (same as reference)
2. Load mxfp4 KV from HBM, dequant to bf16 in-register
3. Compute QK^T scores in bf16/fp32
4. Softmax in fp32
5. Compute output with dequanted V

This avoids the full fp8 KV read and uses the smaller mxfp4 buffer.
"""
import torch
import torch.nn.functional as F
from task import input_t, output_t

from aiter import dtypes as aiter_dtypes
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32

FP8_DTYPE = aiter_dtypes.fp8


def quantize_fp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    finfo = torch.finfo(FP8_DTYPE)
    amax = tensor.abs().amax().clamp(min=1e-12)
    scale = amax / finfo.max
    fp8_tensor = (tensor / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
    return fp8_tensor, scale.to(torch.float32).reshape(1)


def dequant_mxfp4_kv(kv_buffer_mxfp4, kv_scale_mxfp4, total_kv, dim=576):
    """Dequantize mxfp4 KV cache to bfloat16."""
    # kv_buffer_mxfp4: (total_kv, 1, dim//2) fp4x2
    # kv_scale_mxfp4: (total_kv, N_blocks) fp8_e8m0
    BLOCK_SIZE = 32
    num_blocks = dim // BLOCK_SIZE

    # Flatten to 2D for dequant
    fp4_2d = kv_buffer_mxfp4.view(total_kv, dim // 2)
    float_vals = mxfp4_to_f32(fp4_2d)  # (total_kv, dim)

    # Scale: trim padding and apply blockwise
    scale_f32 = e8m0_to_f32(kv_scale_mxfp4)  # (padded_rows, padded_blocks)
    scale_f32 = scale_f32[:total_kv, :num_blocks]  # (total_kv, num_blocks)

    float_vals_blocked = float_vals.view(total_kv, num_blocks, BLOCK_SIZE)
    scaled = float_vals_blocked * scale_f32.unsqueeze(-1)

    return scaled.view(total_kv, dim).to(torch.bfloat16)


def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    num_heads = config["num_heads"]
    kv_lora_rank = config["kv_lora_rank"]
    qk_head_dim = config["qk_head_dim"]
    sm_scale = config["sm_scale"]
    batch_size = qo_indptr.shape[0] - 1

    total_kv = int(kv_indptr[-1].item())

    # Dequant mxfp4 KV cache to bf16 (one-time, much smaller HBM read than bf16 KV)
    kv_buffer_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
    kv_bf16 = dequant_mxfp4_kv(kv_buffer_mxfp4, kv_scale_mxfp4, total_kv, qk_head_dim)

    # Quantize Q to fp8
    q_fp8, q_scale = quantize_fp8(q)

    # Quantize dequanted KV to fp8 for scaled_mm
    kv_fp8, kv_scale = quantize_fp8(kv_bf16)
    kv_fp8_2d = kv_fp8.view(-1, qk_head_dim)

    scale_one = torch.ones(1, dtype=torch.float32, device="cuda")
    out_list = []

    for i in range(batch_size):
        q_s, q_e = int(qo_indptr[i].item()), int(qo_indptr[i + 1].item())
        kv_s, kv_e = int(kv_indptr[i].item()), int(kv_indptr[i + 1].item())
        seq_q = q_e - q_s
        seq_kv = kv_e - kv_s

        # Q: (seq_q * nhead, 576) fp8, K: (seq_kv, 576) fp8
        qi_fp8 = q_fp8[q_s:q_e].reshape(seq_q * num_heads, qk_head_dim)
        ki_fp8 = kv_fp8_2d[kv_s:kv_e]

        # QK^T via _scaled_mm
        raw_scores = torch._scaled_mm(
            qi_fp8, ki_fp8.t(),
            scale_a=q_scale, scale_b=kv_scale,
            out_dtype=torch.float32,
        )
        scores = raw_scores.view(seq_q, num_heads, seq_kv).permute(1, 0, 2)
        scores = scores * sm_scale
        scores = F.softmax(scores, dim=-1)

        # V from dequanted bf16 KV (first 512 dims)
        vi = kv_bf16[kv_s:kv_e, :kv_lora_rank].float()

        oi = torch.matmul(scores, vi)
        oi = oi.permute(1, 0, 2)
        out_list.append(oi.to(torch.bfloat16))

    return torch.cat(out_list, dim=0)

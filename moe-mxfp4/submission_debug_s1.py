#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Debug: compare Triton stage1 output vs gemm_afp4wfp4 per-expert."""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

_probed = False

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
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    expert_id = tl.load(expert_ids_ptr + pid_m)
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_ids_ptr + offs_block)
    token_mask = token_ids < num_valid
    orig_token = token_ids // top_k
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
    tl.store(out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on, result, mask=out_mask)


def custom_kernel(data: input_t) -> output_t:
    global _probed
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled,
     down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _probed:
        _probed = True
        M = hidden_states.shape[0]
        E = gate_up_weight.shape[0]
        topk = topk_ids.shape[1]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]
        N = 2 * d_expert_pad
        K = d_hidden_pad
        BLOCK_M = 32

        try:
            # 1. Manual per-expert GEMM using gemm_afp4wfp4
            # Pick first token's first expert
            test_expert = topk_ids[0, 0].item()
            a_fp4, a_scale = dynamic_mxfp4_quant(hidden_states[:1])  # [1, K//2]
            w_e = gate_up_weight[test_expert].view(torch.uint8)  # [N, K//2]
            ws_e = gate_up_weight_scale.view(torch.uint8)
            # Need to extract scale for this expert
            ws_3d = ws_e.view(E, N, K // 32)
            ws_expert = ws_3d[test_expert]  # [N, K//32]

            ref_out = gemm_afp4wfp4(a_fp4, w_e, a_scale, ws_expert, dtype=torch.bfloat16)
            print(f"[DBG] gemm_afp4wfp4 expert {test_expert}: {ref_out.shape}, range [{ref_out.min():.4f}, {ref_out.max():.4f}]")
            print(f"[DBG] ref_out[:, :5]: {ref_out[0, :5].tolist()}")

            # 2. Triton stage1 with single token
            max_padded = M * topk + E * BLOCK_M
            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device='cuda')
            expert_ids = torch.empty(max_padded // BLOCK_M, dtype=torch.int32, device='cuda')
            token_nums = torch.empty(E, dtype=torch.int32, device='cuda')
            ntp_tensor = torch.empty(1, dtype=torch.int32, device='cuda')
            aiter.moe_align_block_size(topk_ids, E, BLOCK_M, sorted_ids, expert_ids, token_nums, ntp_tensor)
            ntp = ntp_tensor.item()
            num_blocks = ntp // BLOCK_M
            num_valid = M * topk

            # Pad A
            a_u8 = a_fp4.view(torch.uint8)
            as_u8 = a_scale.view(torch.uint8)
            # For single-token test, pad to handle large sorted_ids
            max_orig = sorted_ids[:ntp].max().item() // topk + 1
            a_u8_pad = torch.zeros(max(max_orig, M), a_u8.shape[1], dtype=torch.uint8, device='cuda')
            a_u8_pad[:1] = a_u8
            as_u8_pad = torch.zeros(max(max_orig, M), as_u8.shape[1], dtype=torch.uint8, device='cuda')
            as_u8_pad[:1] = as_u8

            w1 = gate_up_weight.view(torch.uint8)
            w1s = ws_3d

            triton_out = torch.zeros((num_valid, N), dtype=torch.bfloat16, device='cuda')
            BLOCK_N, BLOCK_K = 128, 128
            grid = (num_blocks * triton.cdiv(N, BLOCK_N),)

            _triton_moe_stage1[grid](
                a_u8_pad, a_u8_pad.stride(0), a_u8_pad.stride(1),
                as_u8_pad, as_u8_pad.stride(0), as_u8_pad.stride(1),
                w1, w1.stride(0), w1.stride(1), w1.stride(2),
                w1s, w1s.stride(0), w1s.stride(1), w1s.stride(2),
                triton_out, triton_out.stride(0), triton_out.stride(1),
                sorted_ids, expert_ids,
                N, K, num_valid, top_k=topk,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            torch.cuda.synchronize()

            # Find which row in triton_out corresponds to token 0, expert test_expert
            # token 0 with expert slot 0 → token_id = 0*topk + 0 = 0
            triton_row = triton_out[0]  # token_id 0
            print(f"[DBG] triton stage1 token0: range [{triton_row.min():.4f}, {triton_row.max():.4f}]")
            print(f"[DBG] triton_out[0, :5]: {triton_row[:5].tolist()}")

            # Compare
            diff = (triton_row[:N] - ref_out[0, :N]).abs()
            print(f"[DBG] Diff token0: max={diff.max():.4f}, mean={diff.mean():.4f}")

        except Exception as e:
            import traceback
            print(f"[DBG] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

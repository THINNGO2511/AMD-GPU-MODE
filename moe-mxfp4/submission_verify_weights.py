#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Verify raw weight format: use our proven GEMM kernel on single expert weights."""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_probed = False

@triton.jit
def _fused_gemm(a_ptr, b_ptr, c_ptr, b_scales_ptr, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn,
                stride_cm, stride_cn, stride_bsn, stride_bsk,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Our proven fused quant+GEMM kernel."""
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)
        b_fp4 = tl.load(b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_scales = tl.load(b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk)
        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c,
             mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


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

        try:
            test_expert = topk_ids[0, 0].item()
            h = hidden_states[:1]  # single token

            # Method 1: gemm_afp4wfp4 (known correct)
            a_fp4, a_scale = dynamic_mxfp4_quant(h)
            w_e = gate_up_weight[test_expert].view(torch.uint8)  # [N, K//2]
            ws_3d = gate_up_weight_scale.view(torch.uint8).view(E, N, K // 32)
            ws_e = ws_3d[test_expert]  # [N, K//32]
            ref = gemm_afp4wfp4(a_fp4, w_e, a_scale, ws_e, dtype=torch.bfloat16)
            print(f"[V] gemm_afp4wfp4: {ref[0,:5].tolist()}")

            # Method 2: Our fused GEMM kernel (same kernel as in GEMM submission)
            C = torch.empty(1, N, dtype=torch.bfloat16, device='cuda')
            grid = (triton.cdiv(1, 32) * triton.cdiv(N, 128),)
            _fused_gemm[grid](
                h, w_e, C, ws_e, 1, N, K,
                h.stride(0), h.stride(1),
                w_e.stride(1), w_e.stride(0),  # B is [N, K//2], stride_bk=stride(1)=1, stride_bn=stride(0)=K//2
                C.stride(0), C.stride(1),
                ws_e.stride(0), ws_e.stride(1),
                BLOCK_M=32, BLOCK_N=128, BLOCK_K=128,
            )
            print(f"[V] fused_gemm: {C[0,:5].tolist()}")

            # Compare
            diff = (ref - C).abs()
            print(f"[V] Diff: max={diff.max():.6f}, mean={diff.mean():.6f}")
            print(f"[V] Expert {test_expert}: N={N}, K={K}")

            # Method 3: Our fused GEMM with 3D weight (expert offset)
            w_3d = gate_up_weight.view(torch.uint8)  # [E, N, K//2]
            # Create a view starting at the expert's offset
            w_e_from_3d = w_3d[test_expert]  # Same as w_e
            C2 = torch.empty(1, N, dtype=torch.bfloat16, device='cuda')
            _fused_gemm[grid](
                h, w_e_from_3d, C2, ws_e, 1, N, K,
                h.stride(0), h.stride(1),
                w_e_from_3d.stride(1), w_e_from_3d.stride(0),
                C2.stride(0), C2.stride(1),
                ws_e.stride(0), ws_e.stride(1),
                BLOCK_M=32, BLOCK_N=128, BLOCK_K=128,
            )
            print(f"[V] fused_3d: {C2[0,:5].tolist()}")
            diff2 = (ref - C2).abs()
            print(f"[V] Diff2: max={diff2.max():.6f}")

        except Exception as e:
            import traceback
            print(f"[V] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

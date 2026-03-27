#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Simple Triton MoE stage1 with pre-gathered activations.
Instead of gather-during-GEMM (which caused memory faults),
use fused_dynamic_mxfp4_quant_moe_sort to pre-gather+quantize,
then run a simple batched GEMM kernel over the sorted data.
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
from aiter.fused_moe import (
    moe_sorting, get_padded_M, get_inter_dim,
    fused_dynamic_mxfp4_quant_moe_sort,
)
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
from aiter.utility import fp4_utils

_tested = False
_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched: return
    _patched = True
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
    try: fm._USE_OPUS_MOE_SORTING = True
    except: pass
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
        if expert <= 64 and q_type == QuantType.per_1x32 and not r.run_1stage and inter_dim < 2048:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(functools.partial(fm.ck_moe_stage1, kernelName=kn1, activation=activation, quant_type=q_type, dtype=dtype, splitk=0, use_non_temporal_load=False), functools.partial(aiter.ck_moe_stage2_fwd, kernelName=STAGE2_32, activation=activation, quant_type=q_type, use_non_temporal_load=False), 32, 0, False)
            except: pass
        return r
    fm.get_2stage_cfgs = new
    fm.cfg_2stages = None


@triton.jit
def _simple_moe_stage1(
    # Pre-sorted+quantized activations [sorted_len, K//2] uint8
    a_ptr,
    # Pre-sorted activation scales [sorted_len, scale_K] uint8
    a_scale_ptr,
    # Weights [E, N, K//2] uint8 (raw, un-shuffled)
    w_ptr,
    # Weight scales [E*N, scale_K] uint8 (raw)
    w_scale_ptr,
    # Output [sorted_len, N] bf16
    out_ptr,
    # Expert IDs per block
    expert_ids_ptr,
    # Dimensions
    N, K, sorted_len,
    # Strides
    stride_am, stride_ak,
    stride_asm, stride_ask,
    stride_we, stride_wn, stride_wk,
    stride_wse, stride_wsn, stride_wsk,
    stride_om, stride_on,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Simple MoE GEMM: sorted activations × per-expert weights."""
    SCALE_GROUP: tl.constexpr = 32
    NUM_XCDS: tl.constexpr = 8

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(sorted_len, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # XCD swizzle
    total = num_pid_m * num_pid_n
    pids_per_xcd = total // NUM_XCDS
    extra = total % NUM_XCDS
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    pid = xcd * pids_per_xcd + tl.minimum(xcd, extra) + local_pid

    # Group ordering
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Get expert for this M-block
    expert_id = tl.load(expert_ids_ptr + pid_m)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < sorted_len

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_k_packed = tl.arange(0, BLOCK_K // 2)

        # Load pre-sorted A [BLOCK_M, BLOCK_K//2]
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k_start // 2 + offs_k_packed)[None, :] * stride_ak,
                     mask=m_mask[:, None], other=0)

        # Load A scales [BLOCK_M, BLOCK_K//32]
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        a_scales = tl.load(a_scale_ptr + offs_m[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_ask,
                           mask=m_mask[:, None], other=0)

        # Load W [BLOCK_K//2, BLOCK_N] for this expert
        w = tl.load(w_ptr + expert_id * stride_we + (k_start // 2 + offs_k_packed)[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Load W scales [BLOCK_N, BLOCK_K//32] for this expert
        w_scales = tl.load(w_scale_ptr + expert_id * stride_wse + offs_n[:, None] * stride_wsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_wsk)

        acc = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", acc)

    # Store output
    out = acc.to(tl.bfloat16)
    out_mask = m_mask[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             out, mask=out_mask)


def custom_kernel(data: input_t) -> output_t:
    global _tested
    _patch()
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _tested:
        _tested = True
        M = hidden_states.shape[0]
        E = gate_up_weight.shape[0]
        topk = topk_ids.shape[1]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]
        _, model_dim, inter_dim = get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)

        try:
            BLOCK_M = 32
            # 1. Sorting
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
                topk_ids, topk_weights, E, model_dim, torch.bfloat16, BLOCK_M, None, None, 0)

            # 2. Fused quant + sort (produces pre-gathered MXFP4 activations)
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=M, topk=1, block_size=BLOCK_M)

            print(f"[TEST] a1: {a1.shape} {a1.dtype}")
            print(f"[TEST] a1_scale: {a1_scale.shape} {a1_scale.dtype}")
            print(f"[TEST] sorted_ids: {sorted_ids.shape}")
            print(f"[TEST] sorted_expert_ids: {sorted_expert_ids.shape}")

            # The a1 output from fused_dynamic_mxfp4_quant_moe_sort is already
            # sorted and quantized. It has shape compatible with CK stage1.
            # Let's see if we can feed it to our Triton kernel.

            # Weight tensors (raw, un-shuffled)
            N = 2 * d_expert_pad
            K = d_hidden_pad
            w1 = gate_up_weight.view(torch.uint8)  # [E, N, K//2]
            w1_scale = gate_up_weight_scale.view(torch.uint8)  # [E*N, scale_K]
            scale_K = K // 32
            # Reshape to 3D
            w1_scale_3d = w1_scale.view(E, N, scale_K)

            print(f"[TEST] w1: {w1.shape}, w1_scale_3d: {w1_scale_3d.shape}")
            print(f"[TEST] N={N}, K={K}, scale_K={scale_K}")

            # Output buffer
            sorted_len = sorted_ids.shape[0]
            out = torch.zeros((sorted_len, N), dtype=torch.bfloat16, device='cuda')

            # Expert IDs per M-block
            num_blocks = sorted_len // BLOCK_M
            expert_ids = sorted_expert_ids[:num_blocks]

            print(f"[TEST] sorted_len={sorted_len}, num_blocks={num_blocks}")
            print(f"[TEST] a1 view as uint8: {a1.view(torch.uint8).shape}")

            # Launch Triton kernel
            a1_u8 = a1.view(torch.uint8)
            a1_scale_u8 = a1_scale.view(torch.uint8)

            BLOCK_N = 128
            BLOCK_K = 128
            grid = (num_blocks * triton.cdiv(N, BLOCK_N),)

            print(f"[TEST] Grid: {grid}")
            print(f"[TEST] a1_u8 strides: {a1_u8.stride()}, shape: {a1_u8.shape}")
            print(f"[TEST] a1_scale_u8 strides: {a1_scale_u8.stride()}, shape: {a1_scale_u8.shape}")
            print(f"[TEST] w1 strides: {w1.stride()}")
            print(f"[TEST] w1_scale_3d strides: {w1_scale_3d.stride()}")
            print(f"[TEST] expert_ids[:10]: {expert_ids[:10].tolist()}")
            print(f"[TEST] expert_ids range: [{expert_ids.min()}, {expert_ids.max()}]")
            print(f"[TEST] SKIPPING kernel launch for diagnostic")

        except Exception as e:
            import traceback
            print(f"[TEST] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

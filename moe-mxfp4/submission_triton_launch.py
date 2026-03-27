#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Actually launch the Triton MoE MXFP4 kernel for stage 1.
Uses moe_sorting from aiter + _fused_moe_kernel_mxfp4_silu.
"""
import torch
import triton
import triton.language as tl
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_padded_M, get_inter_dim
import aiter.fused_moe as fm
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import _fused_moe_kernel_mxfp4_silu

_patched = False
_tested = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
    try:
        fm._USE_OPUS_MOE_SORTING = True
    except:
        pass
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _test_triton_stage1(data):
    """Test launching the Triton MoE kernel for stage 1."""
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight.shape[0]
    topk = topk_ids.shape[1]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]
    N = 2 * d_expert_pad  # stage 1 output dim (gate + up)
    K = d_hidden_pad

    # Step 1: Use moe_sorting (same as fused_moe pipeline)
    BLOCK_M = 32
    _, model_dim, inter_dim = get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, E, model_dim, torch.bfloat16, BLOCK_M, None, None, 0)

    # For Triton kernel, need expert_ids per M-block (not per token)
    # sorted_expert_ids has one entry per BLOCK_M-aligned token block
    # Pad to BLOCK_M alignment
    raw_len = sorted_token_ids.shape[0]
    num_blocks = (raw_len + BLOCK_M - 1) // BLOCK_M
    num_tokens_post_padded = num_blocks * BLOCK_M

    # Pad sorted_token_ids to block alignment (fill with M*topk = invalid token id)
    if raw_len < num_tokens_post_padded:
        pad = torch.full((num_tokens_post_padded - raw_len,), M * topk, dtype=sorted_token_ids.dtype, device='cuda')
        sorted_token_ids = torch.cat([sorted_token_ids, pad])

    num_tokens_post_padded_tensor = torch.tensor([num_tokens_post_padded], dtype=torch.int32, device='cuda')
    num_valid_tokens = M * topk

    # expert_ids: one per M-block — pad with -1 for empty blocks
    expert_ids = sorted_expert_ids
    if expert_ids.shape[0] < num_blocks:
        pad_ids = torch.full((num_blocks - expert_ids.shape[0],), -1, dtype=expert_ids.dtype, device='cuda')
        expert_ids = torch.cat([expert_ids, pad_ids])

    print(f"[S1] sorted_ids: {sorted_token_ids.shape}, expert_ids: {expert_ids.shape}")
    print(f"[S1] num_tokens_post_padded: {num_tokens_post_padded}, num_valid: {num_valid_tokens}")
    print(f"[S1] num_valid_ids: {num_valid_ids}")

    # Step 2: Quantize activations to MXFP4
    a_fp4, a_mx_scale = dynamic_mxfp4_quant(hidden_states)
    a_fp4_u8 = a_fp4.view(torch.uint8)
    a_mx_scale_u8 = a_mx_scale.view(torch.uint8)

    # Pad A tensors: kernel accesses a_ptr[sorted_token_id // top_k]
    # Padded sorted_token_ids can be very large (up to num_tokens_post_padded-1)
    # Need: max_sorted_id // topk + 1 rows in A
    max_sorted_id = num_tokens_post_padded - 1
    max_a_row = max_sorted_id // topk + 1
    if a_fp4_u8.shape[0] < max_a_row:
        pad_rows = max_a_row - a_fp4_u8.shape[0]
        a_fp4_u8 = torch.cat([a_fp4_u8, torch.zeros(pad_rows, a_fp4_u8.shape[1], dtype=torch.uint8, device='cuda')])
        a_mx_scale_u8 = torch.cat([a_mx_scale_u8, torch.zeros(pad_rows, a_mx_scale_u8.shape[1], dtype=torch.uint8, device='cuda')])

    print(f"[S1] a_fp4: {a_fp4_u8.shape}, a_mx_scale: {a_mx_scale_u8.shape}")

    # Step 3: Prepare weight tensors (raw, un-shuffled)
    # gate_up_weight: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
    # We need [E, N, K//2] = [E, 2*d_expert_pad, d_hidden_pad//2] — already correct!
    w1_u8 = gate_up_weight.view(torch.uint8)
    # gate_up_weight_scale: [E*2*d_expert_pad, scale_K] e8m0 (flattened)
    # Need [E, N, scale_K] — reshape
    scale_K = K // 32  # MXFP4 block size = 32
    w1_scale_u8 = gate_up_weight_scale.view(torch.uint8)

    print(f"[S1] w1: {w1_u8.shape}, w1_scale: {w1_scale_u8.shape}")
    print(f"[S1] N={N}, K={K}, scale_K={scale_K}")

    # Step 4: Output buffer
    # The kernel writes to c_ptr with sorted token ordering
    # Output: [num_tokens_post_padded, N//2] (SiLU halves N: gate*up → d_expert)
    # Actually, looking at the kernel, it writes to c_ptr[offs_token, offs_n]
    # The SiLU is applied at the end: output = SiLU(gate) * up, halving N
    # Output shape: [num_tokens_post_padded, N//2] = [ntp, d_expert_pad]
    c = torch.zeros((num_tokens_post_padded, N // 2), dtype=torch.bfloat16, device='cuda')

    # Also pad topk_weights: kernel accesses topk_weights[sorted_token_id]
    # topk_weights is [M, topk] = M*topk elements. Padded IDs go beyond.
    tw_flat = topk_weights.view(-1)
    if tw_flat.shape[0] < num_tokens_post_padded:
        tw_flat = torch.cat([tw_flat, torch.zeros(num_tokens_post_padded - tw_flat.shape[0],
                                                   dtype=tw_flat.dtype, device='cuda')])

    # Step 5: Per-tensor scales (not used for microscaled, but kernel reads them)
    a_scale_dummy = torch.tensor(1.0, dtype=torch.float32, device='cuda')
    b_scale_dummy = torch.ones(E, dtype=torch.float32, device='cuda')
    # Flatten w1_scale to [E, N, scale_K] view for proper striding
    w1_scale_3d = w1_scale_u8.view(E, N, scale_K) if w1_scale_u8.shape[0] == E * N else w1_scale_u8

    # Step 6: Compute grid
    BLOCK_N = 128  # Common block size
    BLOCK_K = 128
    num_pid_m = num_tokens_post_padded // BLOCK_M
    num_pid_n = triton.cdiv(N, BLOCK_N)
    grid = (num_pid_m * num_pid_n,)

    print(f"[S1] Grid: {grid} = {num_pid_m} * {num_pid_n}")
    print(f"[S1] Launching Triton stage 1 kernel...")

    try:
        _fused_moe_kernel_mxfp4_silu[grid](
            # Pointers
            a_fp4_u8,           # a_ptr
            w1_u8,              # b_ptr
            c,                  # c_ptr
            a_scale_dummy,      # a_scale_ptr
            b_scale_dummy,      # b_scale_ptr
            a_mx_scale_u8,      # a_mx_scale_ptr
            w1_scale_u8,        # b_mx_scale_ptr
            tw_flat,            # topk_weights_ptr
            sorted_token_ids,   # sorted_token_ids_ptr
            expert_ids,         # expert_ids_ptr
            num_tokens_post_padded_tensor,  # num_tokens_post_padded_ptr
            # Dimensions
            N, K, num_valid_tokens,
            # Strides for A [M, K//2]
            a_fp4_u8.stride(0), a_fp4_u8.stride(1),
            # Strides for B [E, N, K//2]
            w1_u8.stride(0), w1_u8.stride(2), w1_u8.stride(1),
            # Strides for C [ntp, N//2]
            c.stride(0), c.stride(1),
            # Strides for A mx scale [M, scale_K]
            a_mx_scale_u8.stride(0), a_mx_scale_u8.stride(1),
            # Strides for B mx scale [E, N, scale_K]
            w1_scale_3d.stride(0) if w1_scale_3d.dim() == 3 else w1_scale_u8.stride(0) * N,  # stride_bmxe
            w1_scale_3d.stride(2) if w1_scale_3d.dim() == 3 else w1_scale_u8.stride(1),       # stride_bmxk
            w1_scale_3d.stride(1) if w1_scale_3d.dim() == 3 else w1_scale_u8.stride(0),       # stride_bmxn
            # Meta-parameters
            A_DTYPE_FORMAT="e2m1",
            B_DTYPE_FORMAT="e2m1",
            BLOCK_SIZE_M=BLOCK_M,
            BLOCK_SIZE_N=BLOCK_N,
            BLOCK_SIZE_K=BLOCK_K,
            GROUP_SIZE_M=8,
            MUL_ROUTED_WEIGHT=True,
            top_k=topk,
            compute_type=tl.bfloat16,
            SWIZZLE_MX_A=False,
            SWIZZLE_MX_B=False,
        )
        torch.cuda.synchronize()
        print(f"[S1] SUCCESS! Output range: [{c.min():.4f}, {c.max():.4f}]")
    except Exception as e:
        import traceback
        print(f"[S1] KERNEL ERROR: {e}")
        traceback.print_exc()


def custom_kernel(data: input_t) -> output_t:
    global _tested
    _patch()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _tested:
        _tested = True
        try:
            _test_triton_stage1(data)
        except Exception as e:
            import traceback
            print(f"[TEST] error: {e}")
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

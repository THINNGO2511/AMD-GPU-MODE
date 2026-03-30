#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Invoke aiter's Triton MoE MXFP4 kernel directly.
Uses _fused_moe_kernel_mxfp4_silu from moe_op_mxfp4_silu_fused.py for stage 1,
and _fused_moe_kernel_mxfp4 from moe_op_mxfp4.py for stage 2.

The Triton kernels use tl.dot_scaled with raw (un-shuffled) weights,
bypassing the CK path entirely. This could be faster since:
1. Native MXFP4 support via tl.dot_scaled on MI355X
2. XCD swizzle optimized for 8 XCDs
3. No shuffle overhead
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_probed = False

# Best CK configs (fallback)
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


def custom_kernel(data: input_t) -> output_t:
    global _probed
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

    if not _probed:
        _probed = True
        # Probe: try to import and call the Triton MoE kernel
        try:
            from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import _fused_moe_kernel_mxfp4_silu
            print(f"[TRITON] Imported _fused_moe_kernel_mxfp4_silu")

            # Check its parameters
            import inspect
            # Can't get sig of Triton JIT function easily, but we know from reading the source:
            # a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr,
            # a_mx_scale_ptr, b_mx_scale_ptr,
            # topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
            # N, K, num_valid_tokens,
            # stride_am, stride_ak, stride_be, stride_bk, stride_bn,
            # stride_cm, stride_cn, stride_amxm, stride_amxk, stride_bmxe, stride_bmxk, stride_bmxn,
            # BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, EVEN_K,
            # MUL_ROUTED_WEIGHT, top_k, compute_type, SWIZZLE_MX_A, SWIZZLE_MX_B

            # Also read moe_op_mxfp4.py for stage 2
            from aiter.ops.triton.moe.moe_op_mxfp4 import _fused_moe_kernel_mxfp4
            print(f"[TRITON] Imported _fused_moe_kernel_mxfp4 (stage 2)")

            # Read moe_align_block_size
            from aiter.ops.triton.moe.moe_align_block_size import moe_align_block_size
            print(f"[TRITON] Imported moe_align_block_size")
            sig = inspect.signature(moe_align_block_size)
            print(f"  moe_align_block_size{sig}")

            # Read the quant_moe module
            from aiter.ops.triton.moe.quant_moe import _compute_static_fp8_quant
            print(f"[TRITON] Imported _compute_static_fp8_quant")

            # Test: set up a small Triton MoE call
            M = hidden_states.shape[0]
            E = gate_up_weight.shape[0]
            topk = topk_ids.shape[1]
            d_hidden = config["d_hidden"]
            d_expert = config["d_expert"]
            d_hidden_pad = config["d_hidden_pad"]
            d_expert_pad = config["d_expert_pad"]

            # The Triton kernel needs:
            # - sorted_token_ids, expert_ids from moe_align_block_size
            # - a: input activations (MXFP4 quantized)
            # - b: expert weights (raw, [E, N, K//2])
            # - scales for both

            # Step 1: Align block size (Triton-based sorting)
            BLOCK_SIZE_M = 32
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                topk_ids, BLOCK_SIZE_M, E)
            print(f"[TEST] sorted_token_ids: {sorted_token_ids.shape} {sorted_token_ids.dtype}")
            print(f"[TEST] expert_ids: {expert_ids.shape} {expert_ids.dtype}")
            print(f"[TEST] num_tokens_post_padded: {num_tokens_post_padded}")

            # Step 2: Quantize input to MXFP4
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            a_fp4, a_scale = dynamic_mxfp4_quant(hidden_states)
            print(f"[TEST] a_fp4: {a_fp4.shape} {a_fp4.dtype}")
            print(f"[TEST] a_scale: {a_scale.shape} {a_scale.dtype}")

            # Weight layout for Triton kernel:
            # b_ptr expects [E, N, K//2] with strides
            # gate_up_weight (raw): [E, 2*d_expert_pad, d_hidden_pad//2]
            # This IS [E, N, K//2] where N=2*d_expert_pad, K=d_hidden_pad
            w1 = gate_up_weight.view(torch.uint8)
            w1_scale = gate_up_weight_scale.view(torch.uint8)
            print(f"[TEST] w1 (raw uint8): {w1.shape}")
            print(f"[TEST] w1_scale (raw uint8): {w1_scale.shape}")

            # The kernel also needs a_scale_ptr and b_scale_ptr (per-tensor fp32 scales)
            # For MXFP4, these are the e8m0 microscales, not per-tensor
            # a_scale_ptr = None (or dummy), b_scale_ptr = None (or dummy)
            # a_mx_scale_ptr = a_scale (e8m0), b_mx_scale_ptr = w1_scale (e8m0)

            N = 2 * d_expert_pad
            K = d_hidden_pad

            print(f"[TEST] Stage 1: M={M}, N={N}, K={K}, E={E}, topk={topk}")
            print(f"[TEST] Expected grid: ({num_tokens_post_padded} // {BLOCK_SIZE_M}) * cdiv({N}, BLOCK_SIZE_N)")

        except Exception as e:
            import traceback
            print(f"[TRITON] error: {e}")
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

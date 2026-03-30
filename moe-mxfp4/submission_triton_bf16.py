"""
MoE — Triton MXFP4 kernels with bf16 input (NO quantization!).
Skip both quant stages by using bf16 activations directly with the Triton
fused_moe_mxfp4_silu (stage1) and fused_moe_mxfp4 (stage2) kernels.
These kernels use tl.dot_scaled which handles bf16×fp4 natively on MI355X.

Expected benefit: eliminates ~45µs quant overhead (35% of total).
Risk: accuracy might differ, Triton might be slower than CK for GEMM.
"""
import torch
import triton
import triton.language as tl
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_triton_ok = None  # None = not tested, True = works, False = fails

# CK kernel names for fallback
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _triton_moe(hidden_states, w1_raw, w2_raw, w1_scale_raw, w2_scale_raw,
                topk_weights, topk_ids, config):
    """
    Run MoE using Triton kernels with bf16 input — NO quantization!
    Uses RAW (non-shuffled) weights and scales.
    """
    from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
    from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

    M = hidden_states.shape[0]
    E = w1_raw.shape[0]
    topk = topk_ids.shape[1]
    d_hidden = hidden_states.shape[1]
    # w1_raw shape: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2
    # After unpack: N_stage1 = 2*d_expert_pad, K_stage1 = d_hidden_pad
    N_stage1 = w1_raw.shape[1]  # 2*d_expert_pad (gate + up combined)
    d_inter = N_stage1 // 2  # d_expert_pad (after SiLU gate×up)

    # w2_raw shape: [E, d_hidden_pad, d_expert_pad//2] fp4x2
    N_stage2 = w2_raw.shape[1]  # d_hidden_pad

    block_m = 32  # Safe default

    # Sort tokens
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
        topk_ids, topk_weights, E, d_hidden, torch.bfloat16, block_m)

    # Dummy per-tensor scales (required by kernel, but value is 1.0 = no-op)
    device = hidden_states.device
    a_scale = torch.ones(1, dtype=torch.float32, device=device)
    b1_scale = torch.ones(1, dtype=torch.float32, device=device)
    b2_scale = torch.ones(1, dtype=torch.float32, device=device)

    # View weights as uint8 (Triton kernel expects uint8 for fp4x2)
    w1_u8 = w1_raw.view(torch.uint8)
    w2_u8 = w2_raw.view(torch.uint8)

    # E8M0 scales — raw (not shuffled)
    w1_sc = w1_scale_raw.view(torch.uint8)
    w2_sc = w2_scale_raw.view(torch.uint8)

    # Stage 1: gate+up + SiLU (bf16 input, no quant!)
    # Output shape: [sorted_tokens, d_inter] where d_inter = d_expert_pad (after gate×up+SiLU)
    # The SiLU-fused kernel handles the gate×up internally
    a1_out = torch.empty((sorted_ids.shape[0], d_inter), dtype=torch.bfloat16, device=device)

    triton_config_s1 = {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 128,
        'GROUP_SIZE_M': 8,
    }

    fused_moe_mxfp4_silu(
        A=hidden_states,
        B=w1_u8,
        C=a1_out,
        A_scale=a_scale,
        B_scale=b1_scale,
        A_mx_scale=None,  # bf16 input — no microscale needed
        B_mx_scale=w1_sc,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=sorted_expert_ids,
        num_tokens_post_padded=num_valid_ids,
        mul_routed_weight=False,
        top_k=topk,
        swizzle_mx_a=False,
        swizzle_mx_b=False,
        config=triton_config_s1,
        compute_type=tl.bfloat16,
    )

    # Stage 2: down projection (bf16 intermediate, no inter-stage quant!)
    # Output shape needs to match moe_buf layout for weighted accumulation
    # moe_buf shape: [M, d_hidden]
    # But the kernel writes to C with shape [sorted_tokens, top_k, N_stage2]
    # We need the output to be accumulated per-token with routing weights

    # Create output buffer matching kernel expectation
    # The kernel handles mul_routed_weight=True to apply topk_weights
    c_out = torch.zeros((M, N_stage2), dtype=torch.bfloat16, device=device)

    # For stage2, we need to handle the output accumulation differently
    # The MoE kernel with mul_routed_weight=True accumulates with routing weights
    # But it writes to C[token_idx, :] directly
    # The sorted_token_ids maps sorted positions → (token_idx * topk + topk_idx)

    triton_config_s2 = {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': 64,
        'BLOCK_SIZE_K': 128,
        'GROUP_SIZE_M': 8,
    }

    # For stage2, C output should be [num_sorted_tokens, N_stage2]
    # with mul_routed_weight=True, the kernel multiplies by topk_weights
    c_stage2 = torch.empty((sorted_ids.shape[0], N_stage2), dtype=torch.bfloat16, device=device)

    fused_moe_mxfp4(
        A=a1_out,
        B=w2_u8,
        C=c_stage2.unsqueeze(1),  # Add topk dimension: [sorted, 1, N]
        A_scale=a_scale,
        B_scale=b2_scale,
        A_mx_scale=None,  # bf16 intermediate — no microscale
        B_mx_scale=w2_sc,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_ids,
        expert_ids=sorted_expert_ids,
        num_tokens_post_padded=num_valid_ids,
        mul_routed_weight=True,
        top_k=topk,
        swizzle_mx_a=False,
        swizzle_mx_b=False,
        config=triton_config_s2,
        compute_type=tl.bfloat16,
    )

    # The output is accumulated in moe_buf by the kernel? No, the kernel writes to c_stage2
    # Need to manually scatter-add back to output
    # Actually, the CK stage2 writes directly to moe_buf with atomic accumulation
    # The Triton kernel might not do this — need to check

    # For now, use moe_buf from sorting (which is zeroed?) and return
    # This approach might be WRONG — need to understand how Triton kernel accumulates

    hidden_pad = config.get("d_hidden_pad", d_hidden) - config.get("d_hidden", d_hidden)
    return moe_buf[:M, :d_hidden - hidden_pad]


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

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
    global _call_count, _triton_ok
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
    _call_count += 1

    # Always use CK path (proven, stable)
    # The Triton bf16 approach has too many unknowns with output accumulation
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

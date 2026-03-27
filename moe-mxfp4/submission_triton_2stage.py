#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Use Triton MXFP4 kernels directly for 2-stage MoE.
Key advantages:
1. bf16 input (no input quantization!)
2. No inter-stage quantization (bf16 intermediate!)
3. Triton autotuning can find optimal configs
4. Uses RAW (non-shuffled) weights
"""
import torch
import triton.language as tl
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe

_patched = False
_triton_ok = {}  # Track which (E, inter_dim) combos work

# CK kernels for fallback
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    import functools
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
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _triton_moe(hidden_states, w1_raw, w2_raw, w1_scale, w2_scale,
                topk_weights, topk_ids, config):
    """Run MoE using Triton kernels with bf16 input (no quantization!)."""
    from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
    from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4

    M = hidden_states.shape[0]
    E = w1_raw.shape[0]
    topk = topk_ids.shape[1]
    d_hidden = hidden_states.shape[1]
    d_inter_2x = w1_raw.shape[1]  # 2 * d_expert_pad
    d_inter = d_inter_2x // 2  # d_expert_pad (after SiLU gate)

    block_m = fm.get_block_size_M(fm.get_padded_M(M), topk, E, d_inter)

    # Convert fp4x2 weights to uint8 view (Triton kernel expects uint8)
    w1_u8 = w1_raw.view(torch.uint8)
    w2_u8 = w2_raw.view(torch.uint8)
    # Convert e8m0 scales to uint8 view if needed
    w1_sc = w1_scale.view(torch.uint8) if w1_scale.dtype != torch.uint8 else w1_scale
    w2_sc = w2_scale.view(torch.uint8) if w2_scale.dtype != torch.uint8 else w2_scale

    # 1. Sort
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
        topk_ids, topk_weights, E, d_hidden, torch.bfloat16, block_m)

    # 2. Stage 1: gate+up + SiLU (bf16 input, no quant!)
    a1_out = torch.empty((sorted_ids.shape[0], d_inter), dtype=torch.bfloat16,
                         device=hidden_states.device)

    triton_config = {
        'BLOCK_SIZE_M': block_m,
        'BLOCK_SIZE_N': 128,
        'BLOCK_SIZE_K': 128,
        'GROUP_SIZE_M': 8,
    }

    a_scale = torch.ones(1, dtype=torch.float32, device=hidden_states.device)
    b_scale = torch.ones(1, dtype=torch.float32, device=hidden_states.device)

    fused_moe_mxfp4_silu(
        A=hidden_states,
        B=w1_u8,
        C=a1_out,
        A_scale=a_scale,
        B_scale=b_scale,
        A_mx_scale=None,
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
        config=triton_config,
        compute_type=tl.bfloat16,
    )

    # 3. Stage 2: down projection (bf16 intermediate, no inter-stage quant!)
    fused_moe_mxfp4(
        A=a1_out,
        B=w2_u8,
        C=moe_buf.unsqueeze(1),
        A_scale=a_scale,
        B_scale=b_scale,
        A_mx_scale=None,
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
        config=triton_config,
        compute_type=tl.bfloat16,
    )

    return moe_buf


def custom_kernel(data: input_t) -> output_t:
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
    E = gate_up_weight.shape[0]
    _, _, inter_dim = fm.get_inter_dim(gate_up_weight.shape, down_weight.shape)

    # Try Triton for E<=64 cases
    key = (E, inter_dim)
    if E <= 64 and _triton_ok.get(key, True):
        try:
            result = _triton_moe(
                hidden_states, gate_up_weight, down_weight,
                gate_up_weight_scale, down_weight_scale,
                topk_weights, topk_ids, config)
            _triton_ok[key] = True
            return result
        except Exception as e:
            import traceback
            print(f"[TRITON ERR] E={E} d={inter_dim}: {e}")
            traceback.print_exc()
            _triton_ok[key] = False

    # Fallback to CK 2-stage
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

"""
MoE — CK stage1 + FlyDSL stage2.
FlyDSL stage1 crashes (b_scale bug in compile_mixed_moe_gemm1).
But stage2 uses compile_mixed_moe_gemm2 which might work.
If FlyDSL stage2 is faster than CK stage2, we save time on the GEMM phase.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_flydsl_s2_ok = None
_call_count = 0

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# FlyDSL stage2 kernel names
FLYDSL_S2_32 = "flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce"
FLYDSL_S2_64 = "flydsl_moe2_afp4_wfp4_bf16_t64x128x256_reduce"


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
        # Replace CK stage2 with FlyDSL stage2 for ALL non-1stage cases
        if (q_type == QuantType.per_1x32 and not result.run_1stage):
            try:
                # Determine block_m
                if expert <= 64:
                    est_m = token * topk // expert
                    block_m = 32 if est_m < 50 else 64
                else:
                    block_m = result.block_m if hasattr(result, 'block_m') else 32

                flydsl_s2 = FLYDSL_S2_64 if block_m == 64 else FLYDSL_S2_32
                print(f"[FLYDSL] Injecting stage2={flydsl_s2} E={expert} d={inter_dim} bm={block_m}", flush=True)

                # Keep original stage1, replace stage2 with FlyDSL
                return fm.MOEMetadata(
                    result.stage1,  # Keep CK/CSV stage1 as-is
                    functools.partial(fm._flydsl_stage2_wrapper,
                        kernelName=flydsl_s2),
                    block_m, result.ksplit, False)
            except Exception as e:
                print(f"[FLYDSL ERR] {e}", flush=True)
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

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

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Force specific CK kernel tile configs for E=33 cases.
The CK module has multiple tile variants. Default selection may not be optimal.
Try the proven tile configs from DSv3 CSV (which work well for E=257).
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Best kernel names from DSv3 CSV (proven for E=257)
KN1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
KN1_256_32 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
KN1_256_128 = "moe_ck2stages_gemm1_256x128x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"

KN2_64 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"
KN2_128 = "moe_ck2stages_gemm2_64x128x128x128_1x1_MulABScaleExpertWeightShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    original_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def patched_get_2stage_cfgs(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type,
        use_g1u1, activation, doweight_stage1,
        hidden_pad, intermediate_pad, is_shuffled=True,
    ):
        result = original_get_2stage(
            token, model_dim, inter_dim, expert, topk,
            dtype, q_dtype_a, q_dtype_w, q_type,
            use_g1u1, activation, doweight_stage1,
            hidden_pad, intermediate_pad, is_shuffled,
        )

        # For untuned cases (E=33), try injecting kernel names
        if expert <= 64 and q_type == QuantType.per_1x32 and not result.run_1stage:
            try:
                stage1_kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                kn1 = stage1_kw.get('kernelName', '')
                if not kn1:
                    est_m = token * topk // expert
                    # Select kernel based on batch size
                    if est_m < 50:
                        # Small batch: use 64-tile kernels
                        new_kn1 = KN1_64
                        new_kn2 = KN2_64
                        new_block_m = 32
                    elif est_m < 200:
                        # Medium: try 256x32 tile
                        new_kn1 = KN1_256_32
                        new_kn2 = KN2_64
                        new_block_m = 32
                    else:
                        # Large: use 256x128 tiles
                        new_kn1 = KN1_256_128
                        new_kn2 = KN2_128
                        new_block_m = 64

                    use_nt = False  # Disable NT load for E=33

                    print(f"[INJECT] E={expert}, token={token}, est_m={est_m}: block_m={new_block_m}, kn1={new_kn1.split('_')[3]}")

                    return fm.MOEMetadata(
                        functools.partial(
                            fm.ck_moe_stage1,
                            kernelName=new_kn1,
                            activation=activation,
                            quant_type=q_type,
                            dtype=dtype,
                            splitk=0,
                            use_non_temporal_load=use_nt,
                        ),
                        functools.partial(
                            aiter.ck_moe_stage2_fwd,
                            kernelName=new_kn2,
                            activation=activation,
                            quant_type=q_type,
                            use_non_temporal_load=use_nt,
                        ),
                        new_block_m,
                        0,  # ksplit
                        False,  # run_1stage
                    )
            except Exception as e:
                print(f"[INJECT] error: {e}")

        return result

    fm.get_2stage_cfgs = patched_get_2stage_cfgs
    fm.cfg_2stages = None


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

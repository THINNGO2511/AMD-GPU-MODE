#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try cktile_moe_stage1/stage2 path for E=33 cases.
The cktile path supports padding and may perform better than default CK path.
For E=257 cases, use the existing tuned CK configs (which already work).

Also try different block_m sizes for E=33 cases.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import functools

_patched = False


def _patch_configs():
    global _patched
    if _patched:
        return
    _patched = True

    # Monkey-patch get_2stage_cfgs to use cktile for per_1x32 + Silu (not just Swiglu)
    original_get_2stage = fm.get_2stage_cfgs.__wrapped__  # unwrap lru_cache

    @functools.lru_cache(maxsize=2048)
    def patched_get_2stage_cfgs(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type,
        use_g1u1, activation, doweight_stage1,
        hidden_pad, intermediate_pad, is_shuffled=True,
    ):
        # First try original (will match DSv3 CSV for E=257 cases)
        result = original_get_2stage(
            token, model_dim, inter_dim, expert, topk,
            dtype, q_dtype_a, q_dtype_w, q_type,
            use_g1u1, activation, doweight_stage1,
            hidden_pad, intermediate_pad, is_shuffled,
        )

        # For E=33 (TP=4 and EP) cases that fall back to default, try cktile path
        if expert <= 64 and q_type == QuantType.per_1x32:
            # Check if it used default (kernelName would be empty in the metadata)
            try:
                stage1_keywords = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                kn1 = stage1_keywords.get('kernelName', '')
                if not kn1:
                    # Default path - try cktile instead
                    # cktile supports padding and may have better perf
                    from aiter.fused_moe import cktile_moe_stage1, cktile_moe_stage2, MOEMetadata

                    # Select block_m based on estimated tokens per expert
                    est_m = token * topk // expert
                    if est_m < 10:
                        block_m = 16
                    elif est_m < 100:
                        block_m = 32
                    else:
                        block_m = 64

                    print(f"[PATCH] E={expert}: using cktile path with block_m={block_m}, est_m={est_m}")
                    return MOEMetadata(
                        functools.partial(
                            cktile_moe_stage1,
                            n_pad_zeros=intermediate_pad // 64 * 64 * (2 if use_g1u1 else 1),
                            k_pad_zeros=hidden_pad // 128 * 128,
                            activation=activation,
                        ),
                        functools.partial(
                            cktile_moe_stage2,
                            n_pad_zeros=hidden_pad // 64 * 64,
                            k_pad_zeros=intermediate_pad // 128 * 128,
                            activation=activation,
                        ),
                        block_m,
                        0,  # ksplit
                        False,  # run_1stage
                    )
            except Exception as e:
                print(f"[PATCH] error: {e}")

        return result

    fm.get_2stage_cfgs = patched_get_2stage_cfgs
    fm.cfg_2stages = None
    print("[PATCH] Installed cktile path for E<=64 cases")


def custom_kernel(data: input_t) -> output_t:
    _patch_configs()

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

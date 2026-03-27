#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try 1-stage kernel path (fused gate+up+SiLU+down in one dispatch).
Research found aiter has fmoe_g1u1 for per_1x32 MXFP4 1-stage.
Eliminates inter-stage quantization + extra memory traffic + 2nd kernel launch.
Try by forcing use_g1u1=True via monkey-patch.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    fm.use_nt = lambda t, k, e: False

    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    orig = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
            q_type, use_g1u1, activation, doweight_stage1,
            hidden_pad, intermediate_pad, is_shuffled=True):
        # Try 1-stage path first
        try:
            r_1stage = orig(token, model_dim, inter_dim, expert, topk, dtype,
                           q_dtype_a, q_dtype_w, q_type,
                           True,  # use_g1u1=True for 1-stage
                           activation, doweight_stage1,
                           hidden_pad, intermediate_pad, is_shuffled)
            if r_1stage.run_1stage:
                return r_1stage
        except:
            pass

        # Fallback to 2-stage with CK kernel injection for E<=64
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
                 q_type, use_g1u1, activation, doweight_stage1,
                 hidden_pad, intermediate_pad, is_shuffled)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not r.run_1stage and inter_dim < 2048):
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
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
        return r

    fm.get_2stage_cfgs = new
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

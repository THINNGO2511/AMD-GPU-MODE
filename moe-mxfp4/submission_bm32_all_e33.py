#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Per-case optimal CK kernel selection.
- E=257: auto-matched from dsv3_fp4_tuned_fmoe.csv (already optimal)
- E=33, d=512: inject 256x32x128x128 stage1 kernel (15% faster for bs=512)
- E=33, d=2048: use default kernel with use_nt=False + block_m=64
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Kernel names from DSv3 CSV (proven on MI355X)
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch use_nt: disable for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda token, topk, expert: False if expert <= 64 else orig_use_nt(token, topk, expert)

    # Patch block_m: 32 for small batches, 64 for large
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs: inject kernel names for d=512 E=33
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

        # Inject for E<=64 with d<2048, or try overriding E=257 tuned configs
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    # Use 256x32 for large batches, 64x32 for small
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
            except Exception as e:
                print(f"[ERR] {e}")
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


# Override block_m for large est_m (d=2048 uses 128, try 64)
import aiter.fused_moe as _fm2
_orig_bm = getattr(_fm2, "get_block_size_M", None)
if _orig_bm:
    def _custom_bm(token, topk, expert, inter_dim):
        result = _orig_bm(token, topk, expert, inter_dim)
        # For d=2048 (inter_dim=2048), use block_m=64 instead of 128
        if expert <= 64 and result > 32:
            return 32
        return result
    _fm2.get_block_size_M = _custom_bm

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

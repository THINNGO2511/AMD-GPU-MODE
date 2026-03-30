#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Split shared expert from routed experts.
Shared expert (last col of topk_ids) processes ALL tokens — compute as dense GEMM.
Routed experts (first 8 cols) go through fused_moe with E-1 experts.
Avoids sorting overhead for the shared expert since all tokens use it.
Competitor "split_shared_route46" at rank ~10 confirms this works.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_cache = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    fm.use_nt = lambda t, k, e: False if e <= 64 else False
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
    try:
        fm._USE_OPUS_MOE_SORTING = True
    except:
        pass
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w,
            q_type, use_g1u1, activation, doweight_stage1,
            hidden_pad, intermediate_pad, is_shuffled=True):
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
                        functools.partial(fm.ck_moe_stage1, kernelName=kn1,
                            activation=activation, quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=STAGE2_32,
                            activation=activation, quant_type=q_type,
                            use_non_temporal_load=False),
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

    M = hidden_states.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]
    hidden_pad = d_hidden_pad - d_hidden
    intermediate_pad = d_expert_pad - d_expert
    n_routed = config.get("nroutedexperts", gate_up_weight.shape[0] - 1)
    n_shared = config.get("nsharedexperts", 1)

    # If no shared expert or E=33 (small), just use standard fused_moe
    if n_shared == 0 or n_routed <= 64:
        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids,
            expert_mask=None, activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32, doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            a1_scale=None, a2_scale=None,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )

    # Split: routed experts via fused_moe, shared expert via dense GEMM
    routed_ids = topk_ids[:, :-n_shared]  # [M, 8]
    routed_weights = topk_weights[:, :-n_shared]  # [M, 8]
    shared_id = topk_ids[0, -1].item()  # shared expert index

    # 1. Routed experts via fused_moe
    routed_out = fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        routed_weights, routed_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    # 2. Shared expert via fused_moe with topk=1
    # Instead of manually computing dense GEMM (scale format issues),
    # run fused_moe for just the shared expert with all tokens routed to it.
    shared_ids = topk_ids[:, -n_shared:]  # [M, 1]
    shared_weights = topk_weights[:, -n_shared:]  # [M, 1]

    shared_out = fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        shared_weights, shared_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    # 3. Combine routed + shared
    return routed_out + shared_out

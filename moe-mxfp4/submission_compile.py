#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — torch.compile with reduce-overhead mode for kernel fusion.
Based on proven submission_inject_metadata approach + compile wrapper.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_compiled_fn = None

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    fm.use_nt = lambda token, topk, expert: False
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50: return 32
            elif inter_dim >= 2048 and est_m >= 100: return 128
            else: return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, qa, qw, qt, g1, act, dw, hp, ip, sh)
        if expert <= 64 and qt == QuantType.per_1x32 and not r.run_1stage:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est = token * topk // expert
                    if inter_dim >= 2048: return r
                    elif est >= 100: kn = S1_256
                    else: kn = S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1, kernelName=kn, activation=act, quant_type=qt, dtype=dtype, splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd, kernelName=S2_V1, activation=act, quant_type=qt, use_non_temporal_load=False),
                        32, 0, False)
            except: pass
        return r
    fm.get_2stage_cfgs = new_g2s
    fm.cfg_2stages = None


def _call_moe(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s):
    return fused_moe(
        hidden_states, w1, w2, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1s, w2_scale=w2s,
        a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip,
    )


def custom_kernel(data: input_t) -> output_t:
    global _compiled_fn
    _patch()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]

    # Try compiled version (reduce-overhead mode for graph capture)
    if _compiled_fn is None:
        try:
            _compiled_fn = torch.compile(_call_moe, mode="reduce-overhead", fullgraph=False)
        except Exception:
            _compiled_fn = _call_moe

    try:
        return _compiled_fn(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, hp, ip,
            gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        )
    except Exception:
        return _call_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, hp, ip,
            gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        )

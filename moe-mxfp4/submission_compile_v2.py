#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE -- torch.compile v2: CORRECTED for ROCm.

Session 14 used mode="reduce-overhead" which is BROKEN on ROCm (65x slowdown).
This version uses mode="default" with proper ROCm inductor configs.

ANALYSIS: This will NOT help because:
1. fused_moe calls torch.ops.aiter.fused_moe_ (C++ custom op -> graph break)
2. All compute is inside opaque CK ASM + Triton JIT kernels (not traceable)
3. No pure PyTorch ops exist to fuse
4. The 5-6 kernel launches are GPU-compute-bound, not CPU-launch-bound

This submission exists to PROVE the analysis -- expect identical perf to v2.
"""
import torch
import torch._inductor.config as inductor_config
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

# === ROCm-safe inductor config (from AMD-Skills reference) ===
# MUST disable CUDAGraph features on ROCm
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False
# Disable autotuning (can hang on ROCm)
inductor_config.max_autotune = False
inductor_config.coordinate_descent_tuning = False
# Disable memory planning (broken on ROCm)
inductor_config.memory_planning = False
# Enable fusion features that DO work on ROCm
inductor_config.epilogue_fusion = True
inductor_config.pattern_matcher = True
inductor_config.reorder_for_locality = True

_patched = False
_compiled_fns = {}  # per-shape compiled functions

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # All proven optimizations from submission_optimized_v2
    fm.use_nt = lambda token, topk, expert: False

    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk,
                dtype, qa, qw, qt, g1, act, dw, hp, ip, sh=True):
        r = orig(token, model_dim, inter_dim, expert, topk,
                 dtype, qa, qw, qt, g1, act, dw, hp, ip, sh)
        if expert <= 64 and qt == QuantType.per_1x32 and not r.run_1stage and inter_dim < 2048:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est = token * topk // expert
                    kn = S1_256 if est >= 100 else S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn, activation=act,
                            quant_type=qt, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_V1, activation=act,
                            quant_type=qt, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return r
    fm.get_2stage_cfgs = new_g2s
    fm.cfg_2stages = None


def _call_moe(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s):
    """Thin wrapper for torch.compile to trace."""
    return fused_moe(
        hidden_states, w1, w2, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1s, w2_scale=w2s,
        a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip,
    )


def custom_kernel(data: input_t) -> output_t:
    global _compiled_fns
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

    # Per-shape compiled function (avoids recompilation across different shapes)
    shape_key = (hidden_states.shape[0], gate_up_weight_shuffled.shape[0],
                 gate_up_weight_shuffled.shape[1])
    if shape_key not in _compiled_fns:
        try:
            # mode="default" is the ONLY safe mode on ROCm
            # reduce-overhead triggers CUDAGraph capture which HANGS on ROCm
            _compiled_fns[shape_key] = torch.compile(
                _call_moe,
                mode="default",
                fullgraph=False,  # allow graph breaks (C++ custom ops)
                dynamic=False,    # static shapes for this shape_key
            )
        except Exception:
            _compiled_fns[shape_key] = _call_moe

    fn = _compiled_fns[shape_key]
    try:
        return fn(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, hp, ip,
            gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        )
    except Exception:
        # Fallback to eager if compile fails
        return _call_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, hp, ip,
            gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        )

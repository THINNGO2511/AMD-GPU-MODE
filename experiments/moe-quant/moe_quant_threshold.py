#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE: Monkey-patch token_num_quant_moe_sort_switch to 8192.
Forces fused quant+sort path for ALL token counts (currently only <=1024).
For d=2048 bs=512: token_num=4608 > 1024, so currently uses separate path.
Hypothesis: fused path may be faster due to better memory reuse.
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
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Monkey-patch fused_moe_2stages to change quant threshold
    import aiter.fused_moe as _fm
    orig_2stages = _fm.fused_moe_2stages

    def patched_2stages(*args, **kwargs):
        # The function has token_num_quant_moe_sort_switch as a local var
        # We can't change locals, but we can wrap the quant functions
        # Actually, let's just use our proven optimizations
        return orig_2stages(*args, **kwargs)

    # Instead of complex patching, just use v2 approach with use_nt=False + CK injection
    fm.use_nt = lambda token, topk, expert: False

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

    # CK injection for E<=64 d<2048
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
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    # NOW the key change: patch fused_moe_2stages to use higher threshold
    # Read the source, find the threshold, and monkey-patch
    try:
        import inspect
        src = inspect.getsource(_fm.fused_moe_2stages)
        # Can't modify local vars directly. Instead, patch the fused quant function
        # to handle larger token counts
        print("Attempting quant threshold patch...", flush=True)

        # The threshold controls which quant path is used:
        # token_num <= 1024: fused_dynamic_mxfp4_quant_moe_sort (fused)
        # token_num > 1024: separate quant + moe_mxfp4_sort

        # We can't change the local var, but we CAN rewrite fused_moe_2stages
        # via exec with the threshold changed
        new_src = src.replace(
            "token_num_quant_moe_sort_switch = 1024",
            "token_num_quant_moe_sort_switch = 8192"
        )
        if new_src != src:
            # Compile and replace
            local_ns = {}
            # Need all imports available
            exec_globals = _fm.__dict__.copy()
            exec(compile(new_src, "<patched>", "exec"), exec_globals, local_ns)
            if 'fused_moe_2stages' in local_ns:
                _fm.fused_moe_2stages = local_ns['fused_moe_2stages']
                print("SUCCESS: token_num_quant_moe_sort_switch = 8192", flush=True)
            else:
                print("FAILED: function not found in exec result", flush=True)
        else:
            print("FAILED: threshold string not found in source", flush=True)
    except Exception as e:
        print(f"Threshold patch FAILED: {e}", flush=True)


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

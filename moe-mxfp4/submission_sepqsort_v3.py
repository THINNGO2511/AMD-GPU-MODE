#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — sepqsort v3: Force separate quant + sort for ALL shapes.
Key insight: token_num (16/128/512) is always < 1024, so the fused
fused_dynamic_mxfp4_quant_moe_sort is ALWAYS used. Competitor "sepqsort"
suggests separate path is faster. We patch fused_moe_2stages via exec
to set token_num_quant_moe_sort_switch = 0, forcing the else branch.
Combined with all proven optimizations from optimized_v2.
"""
import torch
import functools
import inspect
import textwrap
import sys
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

    # 1. use_nt=False for ALL shapes
    fm.use_nt = lambda token, topk, expert: False

    # 2. OPUS sorting if available
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # 3. block_m tuning for E<=64
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

    # 4. Inject CK kernels for E<=64 d<2048
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

    # 5. Force separate quant + sort (the "sepqsort" approach)
    # token_num_quant_moe_sort_switch is a LOCAL var in fused_moe_2stages.
    # Method A: exec patched source to set threshold=0
    sepqsort_ok = False
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        src = textwrap.dedent(src)
        if 'token_num_quant_moe_sort_switch' in src:
            # Replace threshold: force else branch (separate quant + sort)
            src = src.replace('token_num_quant_moe_sort_switch = 1024',
                              'token_num_quant_moe_sort_switch = 0')
            # Also try other possible values
            for old_val in ['= 512', '= 2048', '= 256']:
                src = src.replace(f'token_num_quant_moe_sort_switch {old_val}',
                                  'token_num_quant_moe_sort_switch = 0')
            ns = dict(fm.__dict__)
            exec(compile(src, 'sepqsort_patch', 'exec'), ns)
            fm.fused_moe_2stages = ns['fused_moe_2stages']
            sepqsort_ok = True
            print("[sepqsort] Patched fused_moe_2stages threshold=0", file=sys.stderr)
        else:
            print("[sepqsort] threshold variable not found in source", file=sys.stderr)
    except Exception as e:
        print(f"[sepqsort] exec patch failed: {e}", file=sys.stderr)

    # Method B fallback: replace fused_dynamic_mxfp4_quant_moe_sort
    if not sepqsort_ok:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.utility.fp4_utils import moe_mxfp4_sort

            def sep_quant_sort(x, sorted_ids=None, num_valid_ids=None,
                               token_num=None, topk=1, block_size=32,
                               scaling_mode='even'):
                a1, a1_scale = dynamic_mxfp4_quant(x)
                a1_scale = moe_mxfp4_sort(
                    a1_scale, sorted_ids=sorted_ids,
                    num_valid_ids=num_valid_ids,
                    token_num=token_num, topk=topk,
                    block_size=block_size)
                return a1, a1_scale

            fm.fused_dynamic_mxfp4_quant_moe_sort = sep_quant_sort
            print("[sepqsort] Replaced fused_dynamic_mxfp4_quant_moe_sort", file=sys.stderr)
        except Exception as e:
            print(f"[sepqsort] replacement also failed: {e}", file=sys.stderr)


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

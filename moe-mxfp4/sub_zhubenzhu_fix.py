#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — zhubenzhu quant fix (Discord Mar 30).

Root cause: aiter's HIP quant kernel rounds scale differently from pytorch.
When scale is in [1.5, 1.75)*2^E, HIP rounds UP to 2 while python rounds DOWN to 1.
This causes subtle accuracy differences in the fused Triton quant path.

Fix (from zhubenzhu):
1. Patch get_quant to return pytorch quant function (get_torch_quant)
2. Set token_num_quant_moe_sort_switch = -1 to NEVER use fused Triton quant
   (all token_num > -1 is true → always takes the quant_func path)

The key problem with previous attempts:
- token_num_quant_moe_sort_switch is a LOCAL variable in fused_moe_2stages
  → cannot be patched via fm.token_num_quant_moe_sort_switch = X
- get_quant is bound at import time as a local name in fused_moe module scope
  → fm.get_quant = X only changes the module attr, not the closure reference

Solution: exec-patch the source of fused_moe_2stages with BOTH changes,
in a namespace where get_quant resolves to get_torch_quant.

Combined with all proven optimizations from submission_optimized_v2.
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

    # === CORE FIX: Patch fused_moe_2stages via exec ===
    # This is the ONLY way to change the LOCAL variable token_num_quant_moe_sort_switch
    # AND replace get_quant in the function's closure.
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        src = textwrap.dedent(src)

        # Change 1: Set threshold to -1 so ALL token counts use quant_func path
        # (token_num > -1 is always true → never uses fused Triton quant)
        patched = False
        if 'token_num_quant_moe_sort_switch' in src:
            # Try exact match first
            for old_val in ['= 1024', '= 512', '= 2048', '= 256']:
                needle = f'token_num_quant_moe_sort_switch {old_val}'
                if needle in src:
                    src = src.replace(needle,
                                      'token_num_quant_moe_sort_switch = -1')
                    patched = True
                    break
            if not patched:
                # Fallback: replace any assignment
                import re
                src = re.sub(
                    r'token_num_quant_moe_sort_switch\s*=\s*\d+',
                    'token_num_quant_moe_sort_switch = -1',
                    src)
                patched = True
            print(f"[zhubenzhu] Patched threshold to -1", file=sys.stderr)
        else:
            print(f"[zhubenzhu] WARNING: threshold var not found in source",
                  file=sys.stderr)

        # Build namespace with ALL globals from fused_moe module
        # but override get_quant to return pytorch version
        ns = dict(fm.__dict__)

        # Change 2: Replace get_quant with get_torch_quant
        # This makes quant_func = get_quant(quant_type) return pytorch quant
        try:
            ns['get_quant'] = aiter.get_torch_quant
            print(f"[zhubenzhu] Replaced get_quant with get_torch_quant",
                  file=sys.stderr)
        except AttributeError:
            print(f"[zhubenzhu] WARNING: aiter.get_torch_quant not found",
                  file=sys.stderr)

        # Also add common imports that might be needed
        ns['torch'] = torch
        ns['aiter'] = aiter
        ns['functools'] = functools
        ns['os'] = __import__('os')
        ns['__builtins__'] = __builtins__

        # Compile and execute the patched function
        code = compile(src, '<zhubenzhu_patch>', 'exec')
        exec(code, ns)

        # Replace the function in the module
        if 'fused_moe_2stages' in ns:
            fm.fused_moe_2stages = ns['fused_moe_2stages']
            print(f"[zhubenzhu] Successfully patched fused_moe_2stages",
                  file=sys.stderr)
        else:
            print(f"[zhubenzhu] WARNING: fused_moe_2stages not found in exec namespace",
                  file=sys.stderr)
            # Try alternative names
            for key in ns:
                if 'fused_moe_2stages' in key or '_fused_moe_ck2stages' in key:
                    fm.fused_moe_2stages = ns[key]
                    print(f"[zhubenzhu] Found and patched via key: {key}",
                          file=sys.stderr)
                    break

    except Exception as e:
        print(f"[zhubenzhu] exec patch failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Fallback: try the simpler approach (may not work if get_quant
        # is closure-captured, but worth trying)
        try:
            if hasattr(fm, 'get_quant'):
                fm.get_quant = aiter.get_torch_quant
                print(f"[zhubenzhu] Fallback: patched fm.get_quant",
                      file=sys.stderr)
        except Exception as e2:
            print(f"[zhubenzhu] Fallback also failed: {e2}",
                  file=sys.stderr)

    # === Proven optimizations from submission_optimized_v2 ===

    # 1. use_nt=False for ALL shapes
    fm.use_nt = lambda token, topk, expert: False

    # 2. block_m tuning for E<=64
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128  # d=2048 large batch: default=128 is better
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 3. Inject CK kernels for E<=64 d<2048
    try:
        orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
    except AttributeError:
        orig_get_2stage = fm.get_2stage_cfgs

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

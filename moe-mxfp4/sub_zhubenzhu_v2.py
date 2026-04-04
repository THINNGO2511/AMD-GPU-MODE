#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE -- zhubenzhu quant fix v2 (Discord Mar 30).

Root cause: aiter's HIP quant kernel rounds scale differently from pytorch.
When scale is in [1.5, 1.75)*2^E, HIP rounds UP to 2 while python rounds DOWN to 1.
This causes subtle accuracy differences in the fused Triton quant path.

Fix (from zhubenzhu):
1. Patch get_quant to return pytorch quant function (get_torch_quant)
2. Set token_num_quant_moe_sort_switch = -1 to NEVER use fused Triton quant
   (token_num > -1 is always true -> always takes the quant_func path)

v2 fix: get_torch_quant returns per_1x32_f4_quant which only accepts
(x, scale=None, quant_dtype=..., shuffle=False). But fused_moe_2stages
calls quant_func with extra kwargs (e.g. num_rows) that only get_hip_quant
accepts. We wrap the torch quant function to filter out unsupported kwargs.

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


def _make_quant_wrapper():
    """Create a wrapper around get_torch_quant that filters kwargs.

    get_torch_quant(QuantType.per_1x32) returns per_1x32_f4_quant with sig:
        (x, scale=None, quant_dtype=torch.float4_e2m1fn_x2, shuffle=False)

    But fused_moe_2stages calls quant_func with extra kwargs like num_rows
    that only get_hip_quant's returned function accepts.

    We return a get_quant replacement that wraps the torch quant function
    to accept and discard any extra kwargs.
    """
    _torch_quant_cache = {}
    # Accepted kwargs for per_1x32_f4_quant
    _accepted_kwargs = {'scale', 'quant_dtype', 'shuffle'}

    def wrapped_get_quant(quant_type):
        if quant_type not in _torch_quant_cache:
            torch_fn = aiter.get_torch_quant(quant_type)

            # Probe the actual signature to be safe
            try:
                sig = inspect.signature(torch_fn)
                actual_params = set(sig.parameters.keys()) - {'x', 'self'}
                accepted = actual_params if actual_params else _accepted_kwargs
            except (ValueError, TypeError):
                accepted = _accepted_kwargs

            def quant_wrapper(x, **kwargs):
                filtered = {k: v for k, v in kwargs.items() if k in accepted}
                return torch_fn(x, **filtered)

            _torch_quant_cache[quant_type] = quant_wrapper
            print(f"[zhubenzhu_v2] Wrapped torch quant for {quant_type}, "
                  f"accepted kwargs: {accepted}", file=sys.stderr)

        return _torch_quant_cache[quant_type]

    return wrapped_get_quant


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
        # (token_num > -1 is always true -> never uses fused Triton quant)
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
            print(f"[zhubenzhu_v2] Patched threshold to -1", file=sys.stderr)
        else:
            print(f"[zhubenzhu_v2] WARNING: threshold var not found in source",
                  file=sys.stderr)

        # Build namespace with ALL globals from fused_moe module
        # but override get_quant with our wrapped version
        ns = dict(fm.__dict__)

        # Change 2: Replace get_quant with wrapped pytorch quant
        # The wrapper accepts any kwargs and filters to only the ones
        # that per_1x32_f4_quant actually supports
        try:
            ns['get_quant'] = _make_quant_wrapper()
            print(f"[zhubenzhu_v2] Replaced get_quant with wrapped get_torch_quant",
                  file=sys.stderr)
        except Exception as e:
            print(f"[zhubenzhu_v2] WARNING: Failed to create quant wrapper: {e}",
                  file=sys.stderr)
            # Fallback: try raw get_torch_quant (may fail on extra kwargs)
            try:
                ns['get_quant'] = aiter.get_torch_quant
                print(f"[zhubenzhu_v2] Fallback: using raw get_torch_quant",
                      file=sys.stderr)
            except AttributeError:
                print(f"[zhubenzhu_v2] WARNING: aiter.get_torch_quant not found",
                      file=sys.stderr)

        # Also add common imports that might be needed
        ns['torch'] = torch
        ns['aiter'] = aiter
        ns['functools'] = functools
        ns['os'] = __import__('os')
        ns['__builtins__'] = __builtins__

        # Compile and execute the patched function
        code = compile(src, '<zhubenzhu_v2_patch>', 'exec')
        exec(code, ns)

        # Replace the function in the module
        if 'fused_moe_2stages' in ns:
            fm.fused_moe_2stages = ns['fused_moe_2stages']
            print(f"[zhubenzhu_v2] Successfully patched fused_moe_2stages",
                  file=sys.stderr)
        else:
            print(f"[zhubenzhu_v2] WARNING: fused_moe_2stages not found in exec namespace",
                  file=sys.stderr)
            # Try alternative names
            for key in ns:
                if 'fused_moe_2stages' in key or '_fused_moe_ck2stages' in key:
                    fm.fused_moe_2stages = ns[key]
                    print(f"[zhubenzhu_v2] Found and patched via key: {key}",
                          file=sys.stderr)
                    break

    except Exception as e:
        print(f"[zhubenzhu_v2] exec patch failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Fallback: try the simpler approach (may not work if get_quant
        # is closure-captured, but worth trying)
        try:
            if hasattr(fm, 'get_quant'):
                fm.get_quant = _make_quant_wrapper()
                print(f"[zhubenzhu_v2] Fallback: patched fm.get_quant with wrapper",
                      file=sys.stderr)
        except Exception as e2:
            print(f"[zhubenzhu_v2] Fallback also failed: {e2}",
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

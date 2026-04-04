#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE -- zhubenzhu quant fix v3 (robust for benchmark mode).

Root cause: aiter's HIP quant kernel rounds scale differently from pytorch.
Fix: Force pytorch quant path by patching fused_moe_2stages.

v2 FAILED benchmark mode because exec() used `ns = dict(fm.__dict__)` which
creates a DISCONNECTED COPY of the module namespace. The exec'd function's
__globals__ pointed to this copy, so:
  - Module state changes (cfg_2stages, etc.) made by C++ code or other module
    functions were invisible to the patched function
  - State changes made by the patched function were invisible to the module
  - On 2nd+ calls, stale/missing state caused crashes or wrong behavior

v3 fix: Use `fm.__dict__` directly (NOT a copy) as the exec namespace.
This means the patched function shares the REAL module globals. All state
mutations are bidirectionally visible. We only override the specific names
we need (get_quant) and the rest stays live.

Additionally, we also replace fused_dynamic_mxfp4_quant_moe_sort in the
module to cover the token_num <= threshold path (fused Triton quant).
This is a belt-and-suspenders approach: even if the exec patch somehow
fails, the fused quant function itself is replaced with pytorch quant.

Combined with all proven optimizations from submission_optimized_v2.
"""
import torch
import functools
import inspect
import textwrap
import re
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
    """Create a get_quant replacement that returns pytorch quant with kwargs filtering.

    get_torch_quant(QuantType.per_1x32) returns per_1x32_f4_quant with sig:
        (x, scale=None, quant_dtype=torch.float4_e2m1fn_x2, shuffle=False)

    But fused_moe_2stages calls quant_func with extra kwargs like num_rows
    that only get_hip_quant's function accepts. We filter to accepted params.
    """
    _cache = {}

    def wrapped_get_quant(quant_type):
        if quant_type not in _cache:
            torch_fn = aiter.get_torch_quant(quant_type)

            # Probe the actual signature to determine accepted kwargs
            try:
                sig = inspect.signature(torch_fn)
                accepted = set(sig.parameters.keys()) - {'x', 'self'}
            except (ValueError, TypeError):
                accepted = {'scale', 'quant_dtype', 'shuffle'}

            def quant_wrapper(x, **kwargs):
                filtered = {k: v for k, v in kwargs.items() if k in accepted}
                return torch_fn(x, **filtered)

            _cache[quant_type] = quant_wrapper

        return _cache[quant_type]

    return wrapped_get_quant


def _make_fused_quant_replacement():
    """Replace fused_dynamic_mxfp4_quant_moe_sort with pytorch quant + original sort.

    This covers the token_num <= token_num_quant_moe_sort_switch path which
    normally calls the fused Triton kernel. We replace it with:
    1. pytorch quant (accurate)
    2. the original moe_mxfp4_sort for the sorting step

    The fused kernel signature:
        fused_dynamic_mxfp4_quant_moe_sort(x, sorted_ids, num_valid_ids,
            token_num, topk, block_size=32, scaling_mode="even")
        -> (x_fp4_sorted, blockscale_sorted)
    """
    # Get the original fused function for fallback
    _orig_fused = getattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort', None)

    # Get pytorch quant
    try:
        _torch_quant_fn = aiter.get_torch_quant(QuantType.per_1x32)
    except Exception:
        _torch_quant_fn = None

    # Get moe_mxfp4_sort for the sort step
    _moe_sort = getattr(fm, 'moe_mxfp4_sort', None)

    if _torch_quant_fn is None or _moe_sort is None:
        print(f"[v3] Cannot build fused quant replacement: "
              f"torch_quant={_torch_quant_fn is not None}, "
              f"moe_sort={_moe_sort is not None}", file=sys.stderr)
        return None

    def replacement(x, sorted_ids, num_valid_ids, token_num, topk,
                    block_size=32, scaling_mode="even"):
        # Step 1: pytorch quant (accurate, no HIP rounding bug)
        x_fp4, x_scale = _torch_quant_fn(x)
        # Step 2: sort the quantized data using original sort function
        return _moe_sort(x_fp4, x_scale, sorted_ids, num_valid_ids,
                         token_num, topk, block_size)

    return replacement


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === APPROACH 1: Replace fused_dynamic_mxfp4_quant_moe_sort ===
    # This covers the token_num <= threshold path (fused Triton quant)
    fused_replacement = _make_fused_quant_replacement()
    if fused_replacement is not None:
        fm.fused_dynamic_mxfp4_quant_moe_sort = fused_replacement
        print(f"[v3] Replaced fused_dynamic_mxfp4_quant_moe_sort", file=sys.stderr)

    # === APPROACH 2: Patch fused_moe_2stages via exec with SHARED globals ===
    # This covers the token_num > threshold path (separate quant via get_quant)
    #
    # KEY FIX vs v2: Use fm.__dict__ directly, NOT dict(fm.__dict__).
    # This ensures the exec'd function shares the REAL module namespace,
    # so all state changes (cfg_2stages, etc.) are bidirectionally visible.
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        src = textwrap.dedent(src)

        # Change 1: Set threshold to -1 so ALL token counts use quant_func path
        # (token_num > -1 is always true -> never uses fused quant)
        if 'token_num_quant_moe_sort_switch' in src:
            src = re.sub(
                r'token_num_quant_moe_sort_switch\s*=\s*\d+',
                'token_num_quant_moe_sort_switch = -1',
                src)
            print(f"[v3] Patched threshold to -1", file=sys.stderr)
        else:
            print(f"[v3] WARNING: threshold var not found in source",
                  file=sys.stderr)

        # Change 2: Override get_quant in the REAL module dict
        # This will be seen by the exec'd function since it uses fm.__dict__
        wrapped_get_quant = _make_quant_wrapper()
        fm.__dict__['get_quant'] = wrapped_get_quant
        print(f"[v3] Replaced get_quant in module dict", file=sys.stderr)

        # Ensure necessary builtins are available in the module dict
        # (exec needs __builtins__ to be present)
        if '__builtins__' not in fm.__dict__:
            fm.__dict__['__builtins__'] = __builtins__

        # Compile and execute the patched function INTO the real module dict
        code = compile(src, '<zhubenzhu_v3_patch>', 'exec')
        exec(code, fm.__dict__)

        if 'fused_moe_2stages' in fm.__dict__:
            print(f"[v3] Successfully patched fused_moe_2stages "
                  f"(globals is fm.__dict__: "
                  f"{fm.fused_moe_2stages.__globals__ is fm.__dict__})",
                  file=sys.stderr)
        else:
            print(f"[v3] WARNING: fused_moe_2stages not found after exec",
                  file=sys.stderr)

    except Exception as e:
        print(f"[v3] exec patch failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

        # Fallback: at least try to patch get_quant at module level.
        # Even if fused_moe_2stages resolves get_quant from module globals
        # (not a closure), this might still work.
        try:
            fm.get_quant = _make_quant_wrapper()
            print(f"[v3] Fallback: patched fm.get_quant", file=sys.stderr)
        except Exception as e2:
            print(f"[v3] Fallback also failed: {e2}", file=sys.stderr)

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
                return 128
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

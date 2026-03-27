#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Bypass C++ fused_moe_ wrapper. Call sorting + fused_moe_2stages directly.
Profiling showed 28% python/C++ dispatch overhead.

Flow: fused_moe(py) → fused_moe_(C++) → sorting(py) → fused_moe_2stages(py)
Our flow: sorting(py) → fused_moe_2stages(py)  [skip C++ round-trip]

Also: pre-compute block_size_M, cache output tensor, explicit block_size_M.
"""
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes as aiter_dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_direct_works = None  # None = untested, True/False after first attempt
_out_cache = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Patch use_nt
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs
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
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    # Probe moe_sorting return type
    print(f"[DIRECT] fm.moe_sorting type: {type(fm.moe_sorting)}", file=sys.stderr)
    print(f"[DIRECT] fm.fused_moe_2stages type: {type(fm.fused_moe_2stages)}", file=sys.stderr)

    # Probe _moe_sorting_impl
    if hasattr(fm, '_moe_sorting_impl'):
        import inspect
        try:
            src = inspect.getsource(fm._moe_sorting_impl)
            lines = src.split('\n')
            print(f"[DIRECT] _moe_sorting_impl ({len(lines)} lines):", file=sys.stderr)
            for i, line in enumerate(lines[:60]):
                print(f"  {i+1}: {line}", file=sys.stderr)
        except Exception as e:
            print(f"[DIRECT] Cannot get _moe_sorting_impl source: {e}", file=sys.stderr)


def _try_direct(hidden_states, w1, w2, topk_weights, topk_ids,
                w1_scale, w2_scale, hidden_pad, intermediate_pad):
    """Try calling sorting + fused_moe_2stages directly."""
    global _direct_works

    token_num, model_dim = hidden_states.shape
    E = topk_ids.max().item() + 1
    topk = topk_ids.shape[1]
    inter_dim = w1.shape[1]  # gate+up dim

    # Get block_size_M
    block_m = fm.get_block_size_M(token_num, topk, E, inter_dim)

    # Call sorting
    sort_result = fm.moe_sorting(
        topk_ids, topk_weights, E, model_dim,
        moebuf_dtype=torch.bfloat16, block_size=block_m,
    )

    # Probe return type on first call
    if _direct_works is None:
        print(f"[DIRECT] sort result type: {type(sort_result)}", file=sys.stderr)
        if isinstance(sort_result, (tuple, list)):
            print(f"[DIRECT] sort result len: {len(sort_result)}", file=sys.stderr)
            for i, r in enumerate(sort_result):
                if isinstance(r, torch.Tensor):
                    print(f"[DIRECT]   [{i}] Tensor {r.dtype} {r.shape}", file=sys.stderr)
                else:
                    print(f"[DIRECT]   [{i}] {type(r).__name__} = {r}", file=sys.stderr)

    # Unpack: expect (sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, ...)
    if not isinstance(sort_result, (tuple, list)) or len(sort_result) < 4:
        raise ValueError(f"Unexpected sort result type: {type(sort_result)} len={len(sort_result) if hasattr(sort_result, '__len__') else '?'}")

    sorted_ids = sort_result[0]
    sorted_weights = sort_result[1]
    sorted_expert_ids = sort_result[2]
    num_valid_ids = sort_result[3]

    # Allocate output
    out_key = (token_num, model_dim)
    if out_key not in _out_cache:
        _out_cache[out_key] = torch.zeros(
            (token_num, model_dim), dtype=torch.bfloat16, device='cuda'
        )
    moe_out = _out_cache[out_key]
    moe_out.zero_()

    # Call fused_moe_2stages directly
    fm.fused_moe_2stages(
        hidden_states, w1, w2,
        topk,
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
        moe_out,
        False,  # isG1U1
        block_m,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        q_dtype_a=aiter_dtypes.fp4x2,
        q_dtype_w=aiter_dtypes.fp4x2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=None,
        a2_scale=None,
        num_local_tokens=None,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
        bias1=None,
        bias2=None,
    )

    return moe_out


def custom_kernel(data: input_t) -> output_t:
    global _direct_works, _call_count
    _patch()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Try direct path
    if _direct_works is not False:
        try:
            result = _try_direct(
                hidden_states,
                gate_up_weight_shuffled, down_weight_shuffled,
                topk_weights, topk_ids,
                gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
                hidden_pad, intermediate_pad,
            )
            if _direct_works is None:
                _direct_works = True
                print(f"[DIRECT] Direct path WORKS!", file=sys.stderr)
            return result
        except Exception as e:
            if _direct_works is None:
                _direct_works = False
                print(f"[DIRECT] Direct path FAILED: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)

    # Fallback to fused_moe
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

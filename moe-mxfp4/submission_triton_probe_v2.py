#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Triton MoE Probe: Import + dump Triton MXFP4 kernels, then use CK best.
Gathers info about available Triton MoE functions and their configs.
"""
import torch
import sys
import inspect
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Try importing Triton MoE MXFP4 kernels
    modules = [
        "aiter.ops.triton.moe.moe_op_mxfp4_silu_fused",
        "aiter.ops.triton.moe.moe_op_mxfp4",
        "aiter.ops.triton.moe.moe_op_gemm_a4w4",
        "aiter.ops.triton.moe.moe_op_e2e",
    ]
    for mod_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[''])
            funcs = [f for f in dir(mod) if not f.startswith('_') and callable(getattr(mod, f, None))]
            print(f"\n=== {mod_name} ===", file=sys.stderr)
            print(f"  functions: {funcs}", file=sys.stderr)
            for fn_name in funcs[:3]:
                fn = getattr(mod, fn_name)
                try:
                    sig = inspect.signature(fn)
                    print(f"  {fn_name}{sig}", file=sys.stderr)
                except:
                    print(f"  {fn_name}: <no signature>", file=sys.stderr)
        except Exception as e:
            print(f"\n=== {mod_name}: IMPORT FAILED: {e} ===", file=sys.stderr)

    # 2. Check if fused_moe_mxfp4_silu can be used as stage1 replacement
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
        src = inspect.getsource(fused_moe_mxfp4_silu)
        lines = src.split('\n')
        print(f"\n=== fused_moe_mxfp4_silu source ({len(lines)} lines) ===", file=sys.stderr)
        # Just show the function def and first 40 lines
        for i, line in enumerate(lines[:40]):
            print(f"  {i}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"\n=== fused_moe_mxfp4_silu source: {e} ===", file=sys.stderr)

    # 3. List all .py files under ops/triton/moe/
    try:
        import os
        moe_dir = "/home/runner/aiter/aiter/ops/triton/moe"
        if os.path.isdir(moe_dir):
            files = sorted(os.listdir(moe_dir))
            print(f"\n=== {moe_dir} ({len(files)} files) ===", file=sys.stderr)
            for f in files:
                print(f"  {f}", file=sys.stderr)
    except Exception as e:
        print(f"\n=== moe dir: {e} ===", file=sys.stderr)

    sys.stderr.flush()


def _patch():
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

_patched = False

def custom_kernel(data: input_t) -> output_t:
    global _patched
    _probe()
    if not _patched:
        _patch()
        _patched = True

    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

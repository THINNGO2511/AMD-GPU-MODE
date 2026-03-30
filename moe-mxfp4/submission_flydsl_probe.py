#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Pure probe: dump _flydsl_stage2_wrapper source + kernel discovery.
Uses best_kernels patches for correctness. Probe outputs to stderr.
"""
import sys
import os
import torch
import functools
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe():
    # 1. Full _flydsl_stage2_wrapper source
    try:
        src = inspect.getsource(fm._flydsl_stage2_wrapper)
        print(f"[FLYDSL] _flydsl_stage2_wrapper:", file=sys.stderr)
        for i, line in enumerate(src.split('\n')):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] source err: {e}", file=sys.stderr)

    # 2. FlyDSL-related lines in get_2stage_cfgs
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\n[FLYDSL] get_2stage_cfgs flydsl refs:", file=sys.stderr)
        for i, line in enumerate(lines):
            if 'flydsl' in line.lower() or 'fly' in line.lower():
                for j in range(max(0, i-2), min(len(lines), i+5)):
                    print(f"  L{j+1}: {lines[j]}", file=sys.stderr)
                print(f"  ---", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] get_2stage err: {e}", file=sys.stderr)

    # 3. Search for flydsl kernel binaries
    try:
        base = '/home/runner/aiter/hsa/gfx950/'
        found = []
        for root, dirs, files in os.walk(base):
            for f in files:
                if 'flydsl' in f.lower() or 'fly_dsl' in f.lower():
                    found.append(os.path.join(root, f))
        print(f"\n[FLYDSL] Kernel binaries: {len(found)}", file=sys.stderr)
        for f in found[:20]:
            print(f"  {f}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] binary search err: {e}", file=sys.stderr)

    # 4. Full dsv3 CSV
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        print(f"\n[FLYDSL] Full dsv3 CSV:", file=sys.stderr)
        with open(csv_path) as f:
            for i, line in enumerate(f):
                print(f"  CSV[{i}]: {line.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] CSV err: {e}", file=sys.stderr)

    # 5. Check FlyDSL Python module
    try:
        import importlib, pkgutil
        moe_pkg = importlib.import_module('aiter.ops.triton.moe')
        subs = [name for _, name, _ in pkgutil.iter_modules(moe_pkg.__path__)]
        print(f"\n[FLYDSL] triton.moe submodules: {subs}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] submod err: {e}", file=sys.stderr)

    # 6. Check if flydsl_moe_stage2 or similar exists in aiter
    try:
        for attr in dir(aiter):
            if 'fly' in attr.lower():
                print(f"[FLYDSL] aiter.{attr}: {type(getattr(aiter, attr)).__name__}", file=sys.stderr)
    except Exception as e:
        print(f"[FLYDSL] aiter attr err: {e}", file=sys.stderr)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe()

    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    def new_bsm(t, k, e, d):
        if e <= 64:
            est_m = t * k // e
            return 32 if est_m < 50 else 64
        return orig_bsm(t, k, e, d)
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
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None
    print(f"\n[FLYDSL] Patches applied", file=sys.stderr)


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
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

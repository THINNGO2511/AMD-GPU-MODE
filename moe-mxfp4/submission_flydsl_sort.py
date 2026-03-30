#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — FlyDSL stage2 + sort kernel optimization.
Profiling: stage1=41%, quant_sort=28%, stage2=23%, sort=8%.

Two optimizations:
1. FlyDSL stage2: _flydsl_stage2_wrapper exists in fm, may be 8-20% faster for stage2
2. Monkey-patch fused_dynamic_mxfp4_quant_moe_sort to remove tl.constexpr recompilation
   (PR #2414 findings: 52% sort speedup, but runner has OLD kernel)

Also: probe is_flydsl_available() and try injecting FlyDSL for stage2.
"""
import sys
import os
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
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === Probe FlyDSL availability ===
    flydsl_available = False
    if hasattr(fm, 'is_flydsl_available'):
        flydsl_available = fm.is_flydsl_available()
        print(f"[MOE] FlyDSL available: {flydsl_available}", file=sys.stderr)
    if hasattr(fm, '_flydsl_stage2_wrapper'):
        print(f"[MOE] _flydsl_stage2_wrapper exists: {type(fm._flydsl_stage2_wrapper)}", file=sys.stderr)

    # === Probe sort kernel source for PR #2414 check ===
    try:
        import inspect
        src = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
        # Check if the Triton kernel has tl.int64 (PR #2414 marker)
        mod = inspect.getmodule(fm.fused_dynamic_mxfp4_quant_moe_sort)
        if mod and hasattr(mod, '_fused_dynamic_mxfp4_quant_moe_sort_kernel'):
            kernel = mod._fused_dynamic_mxfp4_quant_moe_sort_kernel
            kernel_src = inspect.getsource(kernel.fn) if hasattr(kernel, 'fn') else ""
            has_2414 = 'tl.int64' in kernel_src
            print(f"[MOE] Sort kernel has PR #2414: {has_2414}", file=sys.stderr)
            if not has_2414:
                print(f"[MOE] OLD sort kernel — constexpr strides cause recompilation", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] Sort kernel probe: {e}", file=sys.stderr)

    # === Probe FlyDSL stage2 kernel names ===
    try:
        flydsl_dir = '/home/runner/aiter/hsa/gfx950/fmoe_2stages/'
        if os.path.isdir(flydsl_dir):
            files = sorted(os.listdir(flydsl_dir))
            flydsl_files = [f for f in files if 'flydsl' in f.lower() or 'FlyDSL' in f]
            fp4_stage2 = [f for f in files if 'gemm2' in f and 'FP4X2' in f]
            print(f"[MOE] fmoe_2stages dir: {len(files)} files, {len(flydsl_files)} flydsl, {len(fp4_stage2)} fp4 stage2", file=sys.stderr)
            # Print a few stage2 kernel names for reference
            for f in fp4_stage2[:5]:
                print(f"[MOE]   stage2: {f}", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] Dir probe: {e}", file=sys.stderr)

    # === Check dsv3_fp4_tuned_fmoe.csv for FlyDSL entries ===
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path) as f:
            lines = f.readlines()
        flydsl_lines = [l for l in lines if 'flydsl' in l.lower() or 'FlyDSL' in l.lower()]
        print(f"[MOE] dsv3 CSV: {len(lines)} total, {len(flydsl_lines)} FlyDSL entries", file=sys.stderr)
        for l in flydsl_lines[:3]:
            print(f"[MOE]   {l.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] CSV probe: {e}", file=sys.stderr)

    # === Standard best_kernels patches ===
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # === Patch get_2stage_cfgs: inject CK kernels + try FlyDSL stage2 ===
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

        # For E<=64 with d<2048: inject CK kernels
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    # Try FlyDSL for stage2 if available
                    if flydsl_available and hasattr(fm, '_flydsl_stage2_wrapper'):
                        stage2_fn = functools.partial(fm._flydsl_stage2_wrapper,
                            activation=activation,
                            quant_type=q_type, use_non_temporal_load=False)
                        print(f"[MOE] Using FlyDSL stage2 for E={expert} d={inter_dim}", file=sys.stderr)
                    else:
                        stage2_fn = functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False)
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        stage2_fn,
                        32, 0, False)
            except Exception as e:
                print(f"[MOE] inject err: {e}", file=sys.stderr)

        # For ALL configs: try replacing stage2 with FlyDSL if available
        if (flydsl_available and hasattr(fm, '_flydsl_stage2_wrapper')
                and not result.run_1stage and q_type == QuantType.per_1x32):
            try:
                # Check if result already uses FlyDSL
                if hasattr(result.stage2, 'func') and result.stage2.func is not fm._flydsl_stage2_wrapper:
                    stage2_fn = functools.partial(fm._flydsl_stage2_wrapper,
                        activation=activation,
                        quant_type=q_type, use_non_temporal_load=False)
                    return fm.MOEMetadata(
                        result.stage1, stage2_fn,
                        result.block_m, result.ksplit, result.run_1stage)
            except Exception as e:
                pass

        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    print(f"[MOE] All patches applied", file=sys.stderr)


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

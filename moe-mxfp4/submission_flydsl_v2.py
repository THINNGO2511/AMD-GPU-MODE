#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — FlyDSL stage2 with correct kernel name.
Probe found: FlyDSL available, PR #2414 sort already deployed.
FlyDSL kernel name format: flydsl_moe2_afp4_wfp4_bf16_t{block_m}x{N}x{K}_reduce
CSV only has entries for token=16384, but _flydsl_stage2_wrapper may accept
other sizes. Need to probe what kernelName it expects.

This submission: probes _flydsl_stage2_wrapper signature, reads its source,
and tries to construct the right kernel name for our benchmark sizes.
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

# FlyDSL kernel name from CSV for E=257 d=256
FLYDSL_STAGE2_64 = "flydsl_moe2_afp4_wfp4_bf16_t64x256x256_reduce"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === Probe _flydsl_stage2_wrapper source ===
    try:
        src = inspect.getsource(fm._flydsl_stage2_wrapper)
        lines = src.split('\n')
        print(f"[MOE] _flydsl_stage2_wrapper ({len(lines)} lines):", file=sys.stderr)
        for i, line in enumerate(lines[:60]):
            print(f"  {i+1}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] Cannot get _flydsl_stage2_wrapper source: {e}", file=sys.stderr)

    # === List available FlyDSL kernel binaries ===
    try:
        for d in ['/home/runner/aiter/hsa/gfx950/fmoe_2stages/',
                  '/home/runner/aiter/hsa/gfx950/flydsl/',
                  '/home/runner/aiter/hsa/gfx950/']:
            if os.path.isdir(d):
                files = os.listdir(d)
                flydsl = [f for f in files if 'flydsl' in f.lower()]
                if flydsl:
                    print(f"[MOE] FlyDSL in {d}: {len(flydsl)} files", file=sys.stderr)
                    for f in sorted(flydsl)[:10]:
                        print(f"[MOE]   {f}", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] FlyDSL dir probe: {e}", file=sys.stderr)

    # === Read full dsv3 CSV to understand FlyDSL configs ===
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path) as f:
            for i, line in enumerate(f):
                print(f"[MOE] CSV[{i}]: {line.strip()}", file=sys.stderr)
    except Exception as e:
        print(f"[MOE] CSV read: {e}", file=sys.stderr)

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

    # === Patch get_2stage_cfgs: CK injection for E<=64 (no FlyDSL for now) ===
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
    print(f"[MOE] Patches applied (CK injection only, probing FlyDSL)", file=sys.stderr)


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

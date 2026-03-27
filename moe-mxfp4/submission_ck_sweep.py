#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — List ALL CK kernel .co files and try different stage1 tile sizes for E=33 d=512.
The 186 files in fmoe_2stages/ may have kernel variants we haven't tested.
Our current E=33 injection uses 256x32 and 64x32 — there might be better tiles.
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

# Current best kernels
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === PROBE: List ALL .co files in fmoe_2stages ===
    try:
        fmoe_dir = '/home/runner/aiter/hsa/gfx950/fmoe_2stages/'
        files = sorted(os.listdir(fmoe_dir))
        print(f"[CK] fmoe_2stages: {len(files)} files", file=sys.stderr)

        # Categorize by type
        stage1_fp4 = [f for f in files if 'gemm1' in f and 'FP4X2' in f and 'silu' in f.lower()]
        stage2_fp4 = [f for f in files if 'gemm2' in f and 'FP4X2' in f]
        stage1_other = [f for f in files if 'gemm1' in f and f not in stage1_fp4]
        stage2_other = [f for f in files if 'gemm2' in f and f not in stage2_fp4]

        print(f"[CK] Stage1 FP4+Silu: {len(stage1_fp4)} files", file=sys.stderr)
        for f in stage1_fp4:
            # Extract tile size from name
            name = f.replace('.co', '')
            print(f"[CK]   S1: {name}", file=sys.stderr)

        print(f"\n[CK] Stage2 FP4: {len(stage2_fp4)} files", file=sys.stderr)
        for f in stage2_fp4:
            name = f.replace('.co', '')
            print(f"[CK]   S2: {name}", file=sys.stderr)

        print(f"\n[CK] Stage1 other: {len(stage1_other)} files", file=sys.stderr)
        for f in stage1_other[:10]:
            print(f"[CK]   {f.replace('.co', '')}", file=sys.stderr)

        print(f"\n[CK] Stage2 other: {len(stage2_other)} files", file=sys.stderr)
        for f in stage2_other[:10]:
            print(f"[CK]   {f.replace('.co', '')}", file=sys.stderr)
    except Exception as e:
        print(f"[CK] Dir probe error: {e}", file=sys.stderr)

    # === PROBE: Check what kernel names the CK module can load ===
    try:
        # List all kernel names from the CSV that we could use
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path) as f:
            lines = f.readlines()
        # Extract unique stage1 and stage2 kernel names
        s1_names = set()
        s2_names = set()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) > 20:
                kn1 = parts[16] if len(parts) > 16 else ''
                kn2 = parts[19] if len(parts) > 19 else ''
                if kn1: s1_names.add(kn1)
                if kn2: s2_names.add(kn2)
        print(f"\n[CK] Unique stage1 names in CSV: {len(s1_names)}", file=sys.stderr)
        for n in sorted(s1_names):
            print(f"[CK]   {n}", file=sys.stderr)
        print(f"\n[CK] Unique stage2 names in CSV: {len(s2_names)}", file=sys.stderr)
        for n in sorted(s2_names):
            print(f"[CK]   {n}", file=sys.stderr)
    except Exception as e:
        print(f"[CK] CSV parse error: {e}", file=sys.stderr)

    # === PROBE: Check general tuned_fmoe.csv for E=33 entries ===
    try:
        csv_path = '/home/runner/aiter/aiter/configs/tuned_fmoe.csv'
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"\n[CK] tuned_fmoe.csv: {len(lines)} lines", file=sys.stderr)
        for line in lines[:3]:
            print(f"[CK]   {line.strip()[:200]}", file=sys.stderr)
        # Find entries with d=512 or E=33
        for line in lines:
            if ',33,' in line or ',512,' in line:
                print(f"[CK]   E33/d512: {line.strip()[:200]}", file=sys.stderr)
    except Exception as e:
        print(f"[CK] tuned CSV error: {e}", file=sys.stderr)

    # === Standard patches ===
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
    print(f"\n[CK] Patches applied", file=sys.stderr)


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

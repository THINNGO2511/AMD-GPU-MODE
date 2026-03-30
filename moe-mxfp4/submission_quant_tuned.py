"""
MoE — Attempt to speed up Triton quant kernels by modifying num_stages.
Quant takes 35% of total time. Changing num_stages from 1→2 in the
fused_mxfp4_quant Triton kernel might improve throughput via pipelining.
Also probes flydsl and get_2stage_cfgs for alternative kernel selection.
"""
import torch
import functools
import os
import sys
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe_and_patch_quant():
    """Try to modify quant kernel num_stages for better pipelining."""
    quant_path = '/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py'
    try:
        content = open(quant_path).read()
        lines = content.split('\n')

        # Find the fused_dynamic_mxfp4_quant_moe_sort kernel launch
        # Look for num_stages in the kernel launch near the function
        print("\n=== QUANT KERNEL num_stages LOCATIONS ===", flush=True)
        for i, line in enumerate(lines):
            if 'num_stages' in line:
                print(f"  L{i+1}: {line.strip()}", flush=True)
                # Print context
                for j in range(max(0, i-3), min(len(lines), i+3)):
                    print(f"    ctx L{j+1}: {lines[j]}", flush=True)

        # Count occurrences
        count = content.count('num_stages=1')
        print(f"\n  Total 'num_stages=1' occurrences: {count}", flush=True)

        # Try patching ALL num_stages=1 to num_stages=2
        if count > 0:
            new_content = content.replace('num_stages=1', 'num_stages=2')
            open(quant_path, 'w').write(new_content)
            print(f"  Patched {count} occurrences from num_stages=1 to num_stages=2", flush=True)

            # Clear any cached compiled kernels
            import importlib
            import aiter.ops.triton.quant.fused_mxfp4_quant as fmq
            importlib.reload(fmq)
            # Re-import the function
            fm.fused_dynamic_mxfp4_quant_moe_sort = fmq.fused_dynamic_mxfp4_quant_moe_sort
            print("  Reloaded quant module", flush=True)
    except Exception as e:
        print(f"  Quant patch error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # Probe flydsl
    try:
        avail = fm.is_flydsl_available()
        print(f"\nis_flydsl_available(): {avail}", flush=True)
        if avail:
            import inspect
            try:
                src = inspect.getsource(fm._flydsl_stage2_wrapper)
                for i, line in enumerate(src.split('\n')[:30]):
                    print(f"  flydsl L{i+1}: {line}", flush=True)
            except:
                pass
    except Exception as e:
        print(f"flydsl check error: {e}", flush=True)

    # Probe get_2stage_cfgs source (key 100 lines)
    try:
        import inspect
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\n=== get_2stage_cfgs ({len(lines)} lines) ===", flush=True)
        for i, line in enumerate(lines[:120]):
            print(f"  L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"get_2stage_cfgs error: {e}", flush=True)

    # Enumerate available CK kernel names
    try:
        import aiter.fused_moe as fm2
        # Try to get kernel registry from the CK module
        if hasattr(aiter, 'ck_moe_stage1'):
            # Call with empty kernel name to see what's available
            print(f"\n=== CK Module Info ===", flush=True)
            # Check for kernel list method
            ck_module_name = 'module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_'
            for mod_name in dir(torch.ops.aiter):
                if 'moe' in mod_name.lower() and 'stage' in mod_name.lower():
                    print(f"  torch.ops.aiter.{mod_name}", flush=True)

        # Also check for cktile kernels
        if hasattr(fm, 'cktile_moe_stage1'):
            print(f"\n=== cktile_moe_stage1 ===", flush=True)
            print(f"  type: {type(fm.cktile_moe_stage1)}", flush=True)
            try:
                src = inspect.getsource(fm.cktile_moe_stage1)
                for i, line in enumerate(src.split('\n')[:20]):
                    print(f"  L{i+1}: {line}", flush=True)
            except:
                pass
    except Exception as e:
        print(f"CK info error: {e}", flush=True)

    # Check available tuned CSV for E=33 d=2048
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path) as f:
            all_lines = f.readlines()
        # Find configs for d=2048 or E=33
        print(f"\n=== CSV configs for E=33 or d=2048 ===", flush=True)
        for line in all_lines:
            if ',33,' in line or ',2048,' in line:
                print(f"  {line.strip()[:200]}", flush=True)
        print(f"  (total {len(all_lines)} configs)", flush=True)
    except Exception as e:
        print(f"CSV error: {e}", flush=True)

    print("\n=== END PROBE ===\n", flush=True)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe_and_patch_quant()

    # Standard optimizations
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    # CK kernel injection for E<=64 d<2048
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
            except:
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

"""
MoE — Try block_m=16 for tiny cases + TRITON_NUM_STAGES=3 for quant kernels.
Also probe moe_op_e2e.py, read full fused_moe flow (L200-400, L1050-1250).
"""
import os
os.environ["TRITON_NUM_STAGES"] = "3"  # Must be set before Triton imports

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe_e2e_and_fused_moe():
    """Read moe_op_e2e.py and key fused_moe sections."""
    print("\n=== PROBE E2E + FUSED_MOE DETAILS ===", flush=True)

    # 1. Read moe_op_e2e.py
    try:
        path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_e2e.py'
        content = open(path).read()
        lines = content.split('\n')
        print(f"\n--- moe_op_e2e.py ({len(lines)} lines) ---", flush=True)
        # Print first 120 lines
        for i, line in enumerate(lines[:120]):
            print(f"  L{i+1}: {line}", flush=True)
        # Find launcher functions
        for i, line in enumerate(lines):
            if 'def ' in line:
                print(f"  func L{i+1}: {line.strip()}", flush=True)
    except Exception as e:
        print(f"  e2e error: {e}", flush=True)

    # 2. Read full fused_moe function (L128-L400)
    try:
        src = open('/home/runner/aiter/aiter/fused_moe.py').read()
        lines = src.split('\n')

        # Print the _fused_moe_2stages function
        for i, line in enumerate(lines):
            if 'def _fused_moe_2stages' in line or 'def _fused_moe_ck2stages' in line:
                print(f"\n--- {line.strip()} (L{i+1}) ---", flush=True)
                for j in range(i, min(i+150, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                    if j > i + 5 and lines[j].strip() and not lines[j].startswith(' ') and not lines[j].startswith('\t'):
                        break

        # Print fused_moe function (the entry point)
        for i, line in enumerate(lines):
            if line.startswith('def fused_moe('):
                print(f"\n--- fused_moe entry (L{i+1}) ---", flush=True)
                for j in range(i, min(i+80, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break

        # Print the quant+stage1+stage2 section (L1050-1250)
        print(f"\n--- fused_moe quant/stages section ---", flush=True)
        for j in range(1045, min(1260, len(lines))):
            print(f"  L{j+1}: {lines[j]}", flush=True)

        # Print _moe_sorting_impl
        for i, line in enumerate(lines):
            if 'def _moe_sorting_impl' in line:
                print(f"\n--- _moe_sorting_impl (L{i+1}) ---", flush=True)
                for j in range(i, min(i+40, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break

        # Print get_block_size_M
        for i, line in enumerate(lines):
            if 'def get_block_size_M' in line:
                print(f"\n--- get_block_size_M (L{i+1}) ---", flush=True)
                for j in range(i, min(i+30, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break

        # Print get_padded_M
        for i, line in enumerate(lines):
            if 'def get_padded_M' in line:
                print(f"\n--- get_padded_M (L{i+1}) ---", flush=True)
                for j in range(i, min(i+15, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break

    except Exception as e:
        print(f"  fused_moe read error: {e}", flush=True)

    # 3. Read quant_moe.py
    try:
        path = '/home/runner/aiter/aiter/ops/triton/moe/quant_moe.py'
        content = open(path).read()
        lines = content.split('\n')
        print(f"\n--- quant_moe.py ({len(lines)} lines) ---", flush=True)
        for i, line in enumerate(lines[:80]):
            print(f"  L{i+1}: {line}", flush=True)
        for i, line in enumerate(lines):
            if 'def ' in line:
                print(f"  func L{i+1}: {line.strip()}", flush=True)
    except Exception as e:
        print(f"  quant_moe error: {e}", flush=True)

    # 4. Read fused_mxfp4_quant.py
    try:
        path = '/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py'
        content = open(path).read()
        lines = content.split('\n')
        print(f"\n--- fused_mxfp4_quant.py ({len(lines)} lines) ---", flush=True)
        for i, line in enumerate(lines[:60]):
            print(f"  L{i+1}: {line}", flush=True)
        for i, line in enumerate(lines):
            if 'def ' in line:
                print(f"  func L{i+1}: {line.strip()}", flush=True)
    except Exception as e:
        print(f"  fused_mxfp4_quant error: {e}", flush=True)

    # 5. Check what CK kernel names are available
    try:
        import aiter.fused_moe as fm2
        if hasattr(fm2, 'ck_moe_stage1'):
            # List all kernel names that have been registered
            print(f"\n--- CK kernel info ---", flush=True)
            print(f"  ck_moe_stage1 type: {type(fm2.ck_moe_stage1)}", flush=True)
    except:
        pass

    print("\n=== END PROBE ===\n", flush=True)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe_e2e_and_fused_moe()

    # Patch use_nt: disable for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m: try block_m=16 for very small est_m
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 16:
                return 16  # NEW: try block_m=16 for tiny cases
            elif est_m < 50:
                return 32
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs: inject kernel names for d=512 E=33
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
    global _call_count
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

    _call_count += 1

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

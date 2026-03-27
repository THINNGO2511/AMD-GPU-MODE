"""
MoE ksplit bypass probe: Test if CK kernel supports bf16 A (skipping quant).
In fused_moe_2stages, when ksplit>1 and is_shuffled=True, quant is skipped
and bf16 activations are passed directly to stage1.
This probe tests if the CK kernel can handle this for our shapes.
"""
import os
os.environ["AITER_USE_OPUS_MOE_SORTING"] = "1"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_sort_cache = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patched_sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype,
                  block_size, expert_mask, num_local_tokens, dispatch_policy, use_opus):
    device = topk_ids.device
    M, topk = topk_ids.shape
    key = (M, topk, num_experts, block_size, model_dim)
    if key not in _sort_cache:
        max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
        max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
        _sort_cache[key] = (
            torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=moebuf_dtype, device=device),
        )
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = _sort_cache[key]
    fwd_fn = aiter.moe_sorting_fwd
    fwd_fn(topk_ids, topk_weights, sorted_ids, sorted_weights, sorted_expert_ids,
           num_valid_ids, moe_buf, num_experts, int(block_size),
           expert_mask, num_local_tokens, dispatch_policy)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Probe: read fused_moe lines 200-280 to understand q_dtype_a setup
    try:
        lines = open('/home/runner/aiter/aiter/fused_moe.py').readlines()
        print("\n=== fused_moe L200-280 ===", flush=True)
        for i in range(199, min(280, len(lines))):
            print(f"L{i+1}: {lines[i].rstrip()}", flush=True)

        # Find fused_moe_2stages
        for i, line in enumerate(lines):
            if 'def fused_moe_2stages' in line:
                print(f"\n=== fused_moe_2stages (L{i+1}) ===", flush=True)
                for j in range(i, min(i+30, len(lines))):
                    print(f"L{j+1}: {lines[j].rstrip()}", flush=True)
                break

        # Find get_2stage_cfgs to understand ksplit path
        for i, line in enumerate(lines):
            if 'def get_2stage_cfgs' in line:
                print(f"\n=== get_2stage_cfgs (L{i+1}) ===", flush=True)
                for j in range(i, min(i+100, len(lines))):
                    print(f"L{j+1}: {lines[j].rstrip()}", flush=True)
                break

        # Find MOEMetadata class
        for i, line in enumerate(lines):
            if 'class MOEMetadata' in line or 'MOEMetadata' in line and 'namedtuple' in line:
                print(f"\n=== MOEMetadata (L{i+1}) ===", flush=True)
                for j in range(i, min(i+10, len(lines))):
                    print(f"L{j+1}: {lines[j].rstrip()}", flush=True)
                break

        # Check if ksplit CK kernels exist
        print("\n=== Available CK kernel names with ksplit ===", flush=True)
        for i, line in enumerate(lines):
            if 'ksplit' in line.lower() and 'kernel' in line.lower():
                print(f"L{i+1}: {line.rstrip()}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # Check what CK kernel configs support ksplit
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path) as f:
            lines_csv = f.readlines()
        # Find any configs with ksplit > 0
        print("\n=== CSV configs with ksplit > 0 ===", flush=True)
        header = lines_csv[0].strip()
        print(f"Header: {header}", flush=True)
        for line in lines_csv[1:]:
            parts = line.strip().split(',')
            if len(parts) > 14:
                ksplit = parts[14].strip()
                if ksplit and ksplit != '0':
                    print(f"  ksplit={ksplit}: {line.strip()[:200]}", flush=True)
        print("(end of ksplit scan)", flush=True)
    except Exception as e:
        print(f"CSV error: {e}", flush=True)

    # Check tuned_fmoe.csv too
    try:
        csv2 = '/home/runner/aiter/aiter/configs/tuned_fmoe.csv'
        with open(csv2) as f:
            lines_csv2 = f.readlines()
        print(f"\n=== tuned_fmoe.csv ({len(lines_csv2)} lines, first 5) ===", flush=True)
        for line in lines_csv2[:5]:
            print(f"  {line.strip()}", flush=True)
        # Find ksplit > 0
        for line in lines_csv2[1:]:
            parts = line.strip().split(',')
            if len(parts) > 14:
                ksplit = parts[14].strip()
                if ksplit and ksplit != '0':
                    print(f"  ksplit={ksplit}: {line.strip()[:200]}", flush=True)
    except Exception as e:
        print(f"tuned CSV error: {e}", flush=True)

    # Standard patches
    fm._moe_sorting_impl = _patched_sort
    fm.use_nt = lambda t, k, e: False
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    # CK kernel injection
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
        # Log the result
        print(f"[META] token={token} E={expert} d={inter_dim} ksplit={result.ksplit} "
              f"block_m={result.block_m} run_1stage={result.run_1stage} "
              f"q_dtype_a={q_dtype_a} is_shuffled={is_shuffled}", flush=True)

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

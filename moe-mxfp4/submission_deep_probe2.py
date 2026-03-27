"""
MoE Deep Probe v2: Read full source of:
1. fused_moe.py lines 1050-1250 (the 2-stage MXFP4 implementation)
2. moe_op_e2e.py (end-to-end MoE Triton)
3. quant_moe.py (MoE-specialized quantization)
4. Full fused_moe_mxfp4 launcher
5. _moe_sorting_impl (lines 37-78)
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True
    import os

    # 1. fused_moe.py lines 1040-1260 (2-stage MXFP4 path)
    print("\n" + "="*60, flush=True)
    print("=== fused_moe.py L1040-1260 (2-stage MXFP4 impl) ===", flush=True)
    try:
        lines = open('/home/runner/aiter/aiter/fused_moe.py').readlines()
        for i in range(1039, min(1260, len(lines))):
            print(f"L{i+1}: {lines[i].rstrip()}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 1b. Also print _moe_sorting_impl (lines 37-78)
    print("\n" + "="*60, flush=True)
    print("=== _moe_sorting_impl (L37-78) ===", flush=True)
    try:
        for i in range(36, min(78, len(lines))):
            print(f"L{i+1}: {lines[i].rstrip()}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 1c. Also print lines 280-320 (where fused_moe calls sorting)
    print("\n" + "="*60, flush=True)
    print("=== fused_moe main body L280-340 ===", flush=True)
    try:
        for i in range(279, min(340, len(lines))):
            print(f"L{i+1}: {lines[i].rstrip()}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 2. moe_op_e2e.py (full file)
    print("\n" + "="*60, flush=True)
    print("=== moe_op_e2e.py (end-to-end MoE) ===", flush=True)
    try:
        e2e_path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_e2e.py'
        content = open(e2e_path).read()
        for i, line in enumerate(content.split('\n')[:200]):
            print(f"L{i+1}: {line}", flush=True)
        total_lines = len(content.split('\n'))
        if total_lines > 200:
            print(f"... ({total_lines} total lines, showing first 200)", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 3. quant_moe.py (full file)
    print("\n" + "="*60, flush=True)
    print("=== quant_moe.py ===", flush=True)
    try:
        qm_path = '/home/runner/aiter/aiter/ops/triton/moe/quant_moe.py'
        content = open(qm_path).read()
        for i, line in enumerate(content.split('\n')[:150]):
            print(f"L{i+1}: {line}", flush=True)
        total_lines = len(content.split('\n'))
        if total_lines > 150:
            print(f"... ({total_lines} total lines)", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 4. fused_moe_mxfp4 full launcher (after line 80)
    print("\n" + "="*60, flush=True)
    print("=== fused_moe_mxfp4 launcher (L80+) ===", flush=True)
    try:
        fm_path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4.py'
        content = open(fm_path).read()
        for i, line in enumerate(content.split('\n')[79:]):
            print(f"L{i+80}: {line}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 5. fused_moe_mxfp4_silu full launcher (after line 80)
    print("\n" + "="*60, flush=True)
    print("=== fused_moe_mxfp4_silu launcher (L80+) ===", flush=True)
    try:
        fm_path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py'
        content = open(fm_path).read()
        for i, line in enumerate(content.split('\n')[79:]):
            print(f"L{i+80}: {line}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 6. Check _triton_kernels directory for moe kernels
    print("\n" + "="*60, flush=True)
    print("=== _triton_kernels/moe/ contents ===", flush=True)
    try:
        tk_path = '/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/'
        if os.path.isdir(tk_path):
            files = os.listdir(tk_path)
            print(f"Files: {files}", flush=True)
    except Exception as e:
        print(f"Error: {e}", flush=True)

    # 7. Check fused_mxfp4_quant.py (the quant kernel)
    print("\n" + "="*60, flush=True)
    print("=== fused_mxfp4_quant.py (fused_dynamic_mxfp4_quant_moe_sort) ===", flush=True)
    try:
        fq_path = '/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py'
        content = open(fq_path).read()
        # Find the launcher function
        lines_q = content.split('\n')
        for i, line in enumerate(lines_q):
            if 'def fused_dynamic_mxfp4_quant_moe_sort' in line:
                print(f"\n--- fused_dynamic_mxfp4_quant_moe_sort (L{i+1}) ---", flush=True)
                for j in range(i, min(i+80, len(lines_q))):
                    print(f"L{j+1}: {lines_q[j]}", flush=True)
                break
    except Exception as e:
        print(f"Error: {e}", flush=True)

    print("\n=== END PROBE ===\n", flush=True)


def custom_kernel(data: input_t) -> output_t:
    _probe()
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
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

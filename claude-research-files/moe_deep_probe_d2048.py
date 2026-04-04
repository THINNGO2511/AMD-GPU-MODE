#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""MoE deep probe: dump d=2048 kernel configs, available variants, quant threshold."""
import torch
import functools
import inspect
import os
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. What does get_2stage_cfgs return for d=2048?
    print("=== get_2stage_cfgs for d=2048 shapes ===", flush=True)
    try:
        orig = fm.get_2stage_cfgs.__wrapped__
    except AttributeError:
        orig = fm.get_2stage_cfgs

    for token in [16, 128, 512]:
        try:
            result = orig(256, 7168, 2048, 33, 9,
                         torch.bfloat16,
                         torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2,
                         QuantType.per_1x32, False, ActivationType.Silu, False,
                         0, 0, True)
            s1_kn = 'NONE'
            s2_kn = 'NONE'
            if hasattr(result.stage1, 'keywords'):
                s1_kn = result.stage1.keywords.get('kernelName', 'NONE')
            if hasattr(result.stage2, 'keywords'):
                s2_kn = result.stage2.keywords.get('kernelName', 'NONE')
            print(f"  E=33 token={token}: bm={result.block_m} sk={result.ksplit} 1stage={result.run_1stage}", flush=True)
            print(f"    S1: {s1_kn}", flush=True)
            print(f"    S2: {s2_kn}", flush=True)
        except Exception as e:
            print(f"  E=33 token={token}: ERROR {e}", flush=True)

    # 2. Also check E=257 d=256 for comparison
    print("\n=== get_2stage_cfgs for E=257 d=256 ===", flush=True)
    for token in [16, 128, 512]:
        try:
            result = orig(256, 7168, 256, 257, 9,
                         torch.bfloat16,
                         torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2,
                         QuantType.per_1x32, False, ActivationType.Silu, False,
                         0, 0, True)
            s1_kn = 'NONE'
            s2_kn = 'NONE'
            if hasattr(result.stage1, 'keywords'):
                s1_kn = result.stage1.keywords.get('kernelName', 'NONE')
            if hasattr(result.stage2, 'keywords'):
                s2_kn = result.stage2.keywords.get('kernelName', 'NONE')
            print(f"  E=257 token={token}: bm={result.block_m} sk={result.ksplit} 1stage={result.run_1stage}", flush=True)
            print(f"    S1: {s1_kn}", flush=True)
            print(f"    S2: {s2_kn}", flush=True)
        except Exception as e:
            print(f"  E=257 token={token}: ERROR {e}", flush=True)

    # 3. token_num_quant_moe_sort_switch
    print("\n=== fused_moe_2stages source inspection ===", flush=True)
    try:
        from aiter.fused_moe import fused_moe_2stages
        src = inspect.getsource(fused_moe_2stages)
        lines = src.split('\n')[:80]
        for i, line in enumerate(lines):
            print(f"  {i}: {line}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 4. List all kernel variant names from module
    print("\n=== Available CK kernel functions ===", flush=True)
    try:
        import aiter.jit
        mod_path = "/home/runner/aiter/aiter/jit/"
        if os.path.exists(mod_path):
            for f in sorted(os.listdir(mod_path)):
                if 'moe' in f.lower() and f.endswith('.so'):
                    print(f"  JIT module: {f}", flush=True)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)

    # 5. CSV configs for our shapes
    print("\n=== tuned_fmoe.csv for E=33 ===", flush=True)
    csv_path = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            header = f.readline().strip()
            print(f"  Header: {header}", flush=True)
            for line in f:
                # E=33 or nexperts=33
                if ',33,' in line or 'nexperts=33' in line.lower():
                    print(f"  {line.strip()}", flush=True)
    else:
        print("  CSV not found", flush=True)

    # Also check dsv3 CSV
    dsv3_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
    if os.path.exists(dsv3_path):
        print(f"\n=== dsv3 CSV for E=33 ===", flush=True)
        with open(dsv3_path) as f:
            header = f.readline().strip()
            print(f"  Header: {header}", flush=True)
            count = 0
            for line in f:
                if ',33,' in line:
                    print(f"  {line.strip()}", flush=True)
                    count += 1
            print(f"  Total E=33 entries: {count}", flush=True)


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

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe available kernel names for d=2048 shapes.
Lists all stage1/stage2 kernel names and tests which ones work for d=2048.
"""
import torch
import functools
import os
import glob
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

    # 1. List all CK kernel .co files
    co_dir = "/home/runner/aiter/hsa/gfx950/"
    print(f"\n=== CK Kernel .co files in {co_dir} ===", flush=True)
    for root, dirs, files in os.walk(co_dir):
        for f in sorted(files):
            if "moe" in f.lower() and f.endswith(".co"):
                print(f"  {os.path.join(root, f)}", flush=True)

    # 2. List fmoe_2stages directory
    fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    if os.path.exists(fmoe_dir):
        print(f"\n=== fmoe_2stages directory ===", flush=True)
        files = sorted(os.listdir(fmoe_dir))
        print(f"  Total: {len(files)} files", flush=True)
        # Show all unique tile sizes
        sizes = set()
        for f in files:
            parts = f.split("_")
            for p in parts:
                if "x" in p and any(c.isdigit() for c in p):
                    sizes.add(p)
            print(f"  {f}", flush=True)

    # 3. Probe what get_2stage_cfgs returns for d=2048 shapes
    print(f"\n=== Default configs for d=2048 shapes ===", flush=True)
    try:
        orig = fm.get_2stage_cfgs.__wrapped__
        for token in [16, 128, 512]:
            for expert in [33]:
                result = orig(256, 7168, 2048, expert, 9,
                             torch.bfloat16,
                             torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2,
                             QuantType.per_1x32, False, ActivationType.Silu, False,
                             0, 0, True)
                s1_kn = result.stage1.keywords.get('kernelName', 'NONE') if hasattr(result.stage1, 'keywords') else 'no keywords'
                s2_kn = result.stage2.keywords.get('kernelName', 'NONE') if hasattr(result.stage2, 'keywords') else 'no keywords'
                print(f"  E={expert} token={token}: bm={result.block_m} sk={result.ksplit} "
                      f"1stage={result.run_1stage}", flush=True)
                print(f"    S1: {s1_kn}", flush=True)
                print(f"    S2: {s2_kn}", flush=True)
    except Exception as e:
        print(f"  Error probing: {e}", flush=True)

    # 4. Search for 256x128 kernel variants
    print(f"\n=== 256x128 kernel search ===", flush=True)
    for f in sorted(os.listdir(fmoe_dir)) if os.path.exists(fmoe_dir) else []:
        if "256x128" in f or "128x128" in f:
            print(f"  FOUND: {f}", flush=True)

    # 5. Read tuned CSV for d=2048 entries
    csv_paths = [
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
    ]
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"\n=== {csv_path} (d=2048 entries) ===", flush=True)
            with open(csv_path) as f:
                header = f.readline().strip()
                print(f"  Header: {header}", flush=True)
                for line in f:
                    if "2048" in line:
                        print(f"  {line.strip()}", flush=True)


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

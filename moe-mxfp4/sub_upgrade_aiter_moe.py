#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Download latest aiter configs from GitHub.
The runner's tuned_fmoe.csv is old. Latest has E=33 topk=9 entries.
Also download latest fused_moe.py to check for new optimizations.
"""
import os
import subprocess
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

URLS = {
    "tuned_fmoe": "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/configs/tuned_fmoe.csv",
    "dsv3_fmoe": "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
    "fused_moe": "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/fused_moe.py",
}


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    for name, url in URLS.items():
        try:
            out = f"/tmp/aiter_latest_{name}"
            r = subprocess.run(
                ["wget", "-q", "-O", out, url],
                capture_output=True, text=True, timeout=30
            )
            if os.path.exists(out):
                size = os.path.getsize(out)
                print(f"Downloaded {name}: {size} bytes")

                with open(out) as f:
                    content = f.read()

                if name == "tuned_fmoe":
                    lines = content.strip().split('\n')
                    print(f"  Total lines: {len(lines)}")
                    # Find E=33 entries
                    e33 = [l for l in lines if ',33,' in l and ',9,' in l]
                    print(f"  E=33 topk=9 entries: {len(e33)}")
                    for l in e33[:10]:
                        print(f"    {l[:150]}")
                    # Find cu_num=256 entries
                    cu256 = [l for l in lines if l.startswith('256,')]
                    print(f"  cu_num=256 entries: {len(cu256)}")

                elif name == "dsv3_fmoe":
                    lines = content.strip().split('\n')
                    print(f"  Total lines: {len(lines)}")
                    e33 = [l for l in lines if ',33,' in l]
                    print(f"  E=33 entries: {len(e33)}")
                    for l in e33[:5]:
                        print(f"    {l[:150]}")

                elif name == "fused_moe":
                    # Check for new features
                    for keyword in ['qseqlen', 'flydsl', 'bf16_fp8_bound',
                                    'token_num_quant_moe_sort_switch',
                                    'get_hip_quant', 'get_torch_quant']:
                        count = content.count(keyword)
                        if count:
                            print(f"  '{keyword}': {count} occurrences")
                    # Show the quant import line
                    for i, line in enumerate(content.split('\n')[:30]):
                        if 'import' in line and ('quant' in line.lower() or 'get_' in line):
                            print(f"  L{i}: {line.rstrip()}")

        except Exception as e:
            print(f"  {name} error: {e}")

    # Compare runner's CSV
    runner_csv = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
    if os.path.exists(runner_csv):
        with open(runner_csv) as f:
            old = f.read()
        old_lines = old.strip().split('\n')
        print(f"\nRunner tuned_fmoe.csv: {len(old_lines)} lines")
        e33_old = [l for l in old_lines if ',33,' in l and ',9,' in l]
        print(f"  E=33 topk=9 entries: {len(e33_old)}")

    runner_dsv3 = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
    if os.path.exists(runner_dsv3):
        with open(runner_dsv3) as f:
            old = f.read()
        old_lines = old.strip().split('\n')
        print(f"Runner dsv3_fmoe.csv: {len(old_lines)} lines")


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

    # use_nt=False (proven)
    fm.use_nt = lambda token, topk, expert: False

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

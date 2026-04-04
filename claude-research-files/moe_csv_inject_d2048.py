#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Inject custom CSV entries for E=33 d=2048 with block_m=64.
Write to tuned_fmoe.csv BEFORE importing fused_moe.
"""
import os

# Write custom CSV entries for E=33 d=2048 with block_m=64
# The CSV is read on first fused_moe import, so write BEFORE importing
csv_path = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
header = "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,err1,us2,kernelName2,err2,us,run_1stage,tflops,bw"

# Read existing CSV
existing = ""
if os.path.exists(csv_path):
    with open(csv_path) as f:
        existing = f.read()

# Only add if not already present
if ",33,9," not in existing:
    new_entries = []
    # E=33 d=2048 shapes with block_m=64 (our best finding)
    for token in [16, 128, 512]:
        entry = f"256,{token},7168,2048,33,9,silu,bf16,fp4,fp4,per_1x32,0,0,64,0,0,,0,0,,0,0,0,0,0"
        new_entries.append(entry)

    # E=33 d=512 shapes with block_m=32 (proven by CK injection)
    for token in [16, 128, 512]:
        entry = f"256,{token},7168,512,33,9,silu,bf16,fp4,fp4,per_1x32,0,0,32,0,0,,0,0,,0,0,0,0,0"
        new_entries.append(entry)

    with open(csv_path, 'a') as f:
        for entry in new_entries:
            f.write(entry + '\n')
    print(f"Injected {len(new_entries)} CSV entries", flush=True)
else:
    print("E=33 entries already exist", flush=True)

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

# Verify CSV was read
print(f"CSV exists: {os.path.exists(csv_path)}", flush=True)
try:
    with open(csv_path) as f:
        lines = f.readlines()
    e33_lines = [l for l in lines if ',33,' in l]
    print(f"E=33 CSV entries: {len(e33_lines)}", flush=True)
    for l in e33_lines:
        print(f"  {l.strip()}", flush=True)
except Exception as e:
    print(f"CSV read error: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
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

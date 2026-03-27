#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Monkey-patch internal functions for better tuning.
Probe get_block_size_M, use_nt, get_ksplit defaults and try overrides.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import inspect

_probed = False


def custom_kernel(data: input_t) -> output_t:
    global _probed

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    if not _probed:
        _probed = True
        # Probe internal functions
        for fname in ['get_block_size_M', 'use_nt', 'get_ksplit', 'get_2stage_cfgs']:
            fn = getattr(fm, fname, None)
            if fn:
                try:
                    sig = inspect.signature(fn)
                    print(f"[FN] {fname}{sig}")
                    # Try to read source
                    src = inspect.getsource(fn)
                    # Print first 10 lines
                    lines = src.split('\n')[:15]
                    for line in lines:
                        print(f"  {line}")
                except Exception as e:
                    print(f"[FN] {fname}: error {e}")

        # Read tuned config header
        try:
            with open("/home/runner/aiter/aiter/configs/tuned_fmoe.csv") as f:
                lines = f.readlines()
            print(f"\n[CFG] tuned_fmoe.csv header: {lines[0].strip()}")
            for line in lines[1:5]:
                print(f"  {line.strip()}")
        except Exception as e:
            print(f"[CFG] error: {e}")

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

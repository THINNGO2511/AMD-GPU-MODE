#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try calling fused_moe_2stages directly + read untuned config + list kernel files.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import inspect, os, glob

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
        # Read untuned config
        path = "/home/runner/aiter/aiter/configs/untuned_fmoe.csv"
        with open(path) as f:
            print(f"[UNTUNED] content:")
            print(f.read())

        # List MoE 2-stages kernel files
        kdir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
        if os.path.exists(kdir):
            files = os.listdir(kdir)
            print(f"\n[KERNELS] fmoe_2stages: {len(files)} files")
            # Show fp4/mxfp4 related kernels
            for f in sorted(files)[:20]:
                print(f"  {f}")
            # Search for per1x32 or Fp4 kernels
            fp4_kernels = [f for f in files if 'Fp4' in f or 'fp4' in f or '1x32' in f]
            print(f"\n[FP4] FP4/per1x32 kernels: {len(fp4_kernels)}")
            for f in sorted(fp4_kernels)[:15]:
                print(f"  {f}")

        # Read fused_moe_2stages signature
        try:
            sig = inspect.signature(fm.fused_moe_2stages)
            print(f"\n[FN] fused_moe_2stages{sig}")
        except:
            try:
                src = inspect.getsource(fm.fused_moe_2stages)
                lines = src.split('\n')[:5]
                for l in lines:
                    print(f"  {l}")
            except Exception as e:
                print(f"[FN] fused_moe_2stages: {e}")

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

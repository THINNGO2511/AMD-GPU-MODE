#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Read untuned config + try AITER_USE_NT=0 + probe kernel names.
"""
import os
os.environ["AITER_USE_NT"] = "0"

import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

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
        # Read untuned config (14 lines — our MXFP4 fallback)
        try:
            with open("/home/runner/aiter/aiter/configs/untuned_fmoe.csv") as f:
                content = f.read()
            print(f"[UNTUNED]\n{content}")
        except Exception as e:
            print(f"[UNTUNED] error: {e}")

        # Probe what block_size_M the auto-selection picks for our sizes
        try:
            for bs, e, d in [(16, 257, 256), (128, 257, 256), (512, 257, 256),
                              (16, 33, 512), (128, 33, 512), (512, 33, 512),
                              (512, 33, 2048)]:
                inter = 2 * d  # gate+up dimension
                bm = fm.get_block_size_M(bs, 9, e, inter)
                nt = fm.use_nt(bs, 9, e)
                print(f"[AUTO] bs={bs} E={e} d={d}: block_m={bm}, use_nt={nt}")
        except Exception as ex:
            print(f"[AUTO] error: {ex}")

        # List MoE kernel files
        import glob
        moe_kernels = glob.glob("/home/runner/aiter/hsa/gfx950/*moe*") + \
                      glob.glob("/home/runner/aiter/hsa/gfx950/fmoe*")
        print(f"\n[KERNELS] MoE kernel dirs: {moe_kernels[:10]}")

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

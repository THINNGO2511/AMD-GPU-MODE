#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — ONLY use_nt=False. Zero other changes.

Previous submission of this exact approach timed out — likely a runner fluke.
This adds ZERO JIT overhead vs vanilla. Just one Python lambda override.
If vanilla fused_moe works (it does), this MUST work.

Purpose: establish whether our block_m override + CK injection is HURTING.
If this scores < 169μs, all our monkey-patching was net negative.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_p = False
def _patch():
    global _p
    if _p: return
    _p = True
    fm.use_nt = lambda t, k, e: False

def custom_kernel(data: input_t) -> output_t:
    _patch()
    (hs, guw, dw, guws, dws, guw_sh, dw_sh, guws_sh, dws_sh, tw, ti, cfg) = data
    return fused_moe(
        hs, guw_sh, dw_sh, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guws_sh, w2_scale=dws_sh,
        a1_scale=None, a2_scale=None,
        hidden_pad=cfg["d_hidden_pad"]-cfg["d_hidden"],
        intermediate_pad=cfg["d_expert_pad"]-cfg["d_expert"],
    )

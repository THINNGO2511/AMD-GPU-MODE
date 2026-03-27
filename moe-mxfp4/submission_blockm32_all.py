#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE submission: Force block_m=32 for ALL shapes.
Rationale: DSv3 CSV uses block_m=32 for ALL E=257 shapes.
Test if the same applies to E=33.

RISK: Previous test showed block_m=32 for d=2048 was 43% WORSE.
This submission is exploratory — benchmarking only.

Analysis of block_m effects:
- Smaller block_m (16/32) = more tiles = better CU occupancy on MI355X (304 CUs)
- Smaller block_m = less register pressure per tile
- But smaller block_m = more kernel launches or more tiles to schedule
- For E=33 with 9 experts per token: est_m = bs*9/33
  - bs=16:  est_m=4   → default block_m=32  (no change)
  - bs=128: est_m=35  → default block_m=64  (override to 32)
  - bs=512: est_m=140 → default block_m=128 (d=2048) or 64 (d=512) (override to 32)

Competitor analysis:
- ry2009 (151.6μs): "force32_128" = block_m=32 for small, 128 for large
- Ryan Mathieu (151μs): "16128_blockmwide" = block_m=16 and 128, wide block_m
- "blockmwide" means WIDER/LARGER block_m, not smaller
- "16128" = block_m=16 for tiny est_m, block_m=128 for large est_m
"""

import os
import torch
from typing import Dict, Tuple

# Environment setup
os.environ["CU_NUM"] = "256"
os.environ["AITER_USE_NT"] = "0"

# Monkey-patch block_m BEFORE importing fused_moe internals
import aiter.fused_moe as fm

# Save original for reference
_original_get_block_size_M = fm.get_block_size_M

# Force block_m=32 for everything
fm.get_block_size_M = lambda t, k, e, d: 32

# Also force use_nt = False
fm.use_nt = lambda t, k, e: False

from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

input_t = Tuple[torch.Tensor, ...]
output_t = torch.Tensor


def custom_kernel(data: input_t) -> output_t:
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data

    num_tokens = hidden_states.shape[0]
    d_expert = config["d_expert"]
    n_routed = config["n_routed_experts"]
    n_shared = config["n_shared_experts"]
    topk = config["total_top_k"]
    E = n_routed + n_shared

    out = fused_moe(
        hidden_states,
        w1_qw,   # pre-shuffled weights
        w2_qw,   # pre-shuffled weights
        topk_weights,
        topk_ids,
        w1_scale=w1_qs,  # pre-shuffled scales
        w2_scale=w2_qs,  # pre-shuffled scales
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
    )
    return out

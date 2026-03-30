#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Metadata Inject — Monkey-patch get_2stage_cfgs to return optimal MOEMetadata
for E=33 shapes that have ZERO tuned CSV entries on the runner.

Key insight: E=257 shapes have tuned CSV entries with specific kernel names → fast.
E=33 shapes fall to heuristics with empty kernelName="" → auto-select → slow.
This patch injects the same proven kernel names from DSv3 CSV for E=33 shapes.

Approach:
1. BEFORE first fused_moe call, unwrap the @lru_cache on get_2stage_cfgs
2. Replace with our wrapper that returns hand-crafted MOEMetadata for E=33
3. Falls through to original for E=257 (already has CSV matches)
4. Also patches use_nt, sepqsort, and block_m

This avoids the CSV format issues that plagued custom CSV approaches.
"""
import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

import functools
import torch
from typing import Dict, Tuple

# Import BEFORE any fused_moe call to ensure clean LRU cache
import aiter
import aiter.fused_moe as fm
from aiter.fused_moe import (
    fused_moe, MOEMetadata, ck_moe_stage1,
    get_block_size_M, get_inter_dim, get_padded_M,
)
from aiter import ActivationType, QuantType, dtypes

input_t = Tuple[torch.Tensor, ...]
output_t = torch.Tensor

# ============================================================================
# PROVEN KERNEL NAMES (from dsv3_fp4_tuned_fmoe.csv, confirmed on MI355X runner)
# ============================================================================
# Stage1: two tile sizes depending on estimated tokens per expert
S1_SMALL = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_MEDIUM = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
# Stage2: single kernel works well for all decode sizes
S2_DEFAULT = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Note: The E=257 CSV uses S1_SMALL for token<=16 and S1_MEDIUM for token=128,
# then back to S1_SMALL for token>=512. But all use block_m=32, ksplit=0.
# For E=33: est_m_per_expert is much higher, so different tile sizes may be optimal.

# ============================================================================
# MONKEY-PATCH get_2stage_cfgs BEFORE first call
# ============================================================================

# Step 1: Unwrap the @lru_cache to get the raw function
_orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

def _make_e33_metadata(token, inter_dim, dtype, q_type, activation):
    """
    Create optimal MOEMetadata for E=33 MXFP4 shapes.
    
    E=33 with topk=9: est_m_per_expert = token * 9 / 33
      token=16:  est_m ≈ 4   → very small tiles, block_m=32
      token=128: est_m ≈ 35  → medium tiles, block_m=32
      token=512: est_m ≈ 140 → larger tiles, block_m=32
    
    DSv3 CSV uses block_m=32 for ALL decode sizes (token <= 2048).
    This is DIFFERENT from heuristic which picks 64 or 128 for larger tokens.
    block_m=32 with specific kernel names is proven faster.
    """
    # Select stage1 kernel based on token count
    # E=257 CSV pattern: SMALL for token<=16, MEDIUM for token=128, SMALL for token>=512
    # For E=33 we try the same pattern
    if token <= 16:
        s1_kernel = S1_SMALL
    elif token <= 128:
        s1_kernel = S1_MEDIUM
    else:
        # For large token counts, SMALL kernel often better due to less overhead
        s1_kernel = S1_SMALL
    
    stage1 = functools.partial(
        ck_moe_stage1,
        kernelName=s1_kernel,
        activation=activation,
        quant_type=q_type,
        dtype=dtype,
        splitk=0,
        use_non_temporal_load=False,
    )
    
    stage2 = functools.partial(
        aiter.ck_moe_stage2_fwd,
        kernelName=S2_DEFAULT,
        activation=activation,
        quant_type=q_type,
        use_non_temporal_load=False,
    )
    
    return MOEMetadata(
        stage1=stage1,
        stage2=stage2,
        block_m=32,       # DSv3 CSV uses 32 for ALL decode sizes
        ksplit=0,          # No split-k (ksplit>1 triggers slow cktile path for Silu)
        run_1stage=False,  # Always 2-stage for MXFP4
        has_bias=False,
        use_non_temporal_load=False,  # Confirmed better
    )

# Step 2: Create wrapper with fresh @lru_cache
@functools.lru_cache(maxsize=2048)
def _patched_get_2stage_cfgs(
    token, model_dim, inter_dim, expert, topk,
    dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
    activation, doweight_stage1,
    hidden_pad, intermediate_pad, is_shuffled=True,
):
    """
    Wrapper for get_2stage_cfgs that injects optimal MOEMetadata for E=33 shapes.
    Falls through to original for E=257 (already has CSV matches).
    """
    # Only intercept E=33 shapes with MXFP4 (per_1x32) + Silu
    if (expert <= 64 
        and q_type == QuantType.per_1x32 
        and activation == ActivationType.Silu
        and q_dtype_a == dtypes.fp4x2
        and q_dtype_w == dtypes.fp4x2
        and not doweight_stage1
    ):
        return _make_e33_metadata(token, inter_dim, dtype, q_type, activation)
    
    # E=257 and all other shapes: use original (has CSV matches)
    return _orig_get_2stage(
        token, model_dim, inter_dim, expert, topk,
        dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1,
        activation, doweight_stage1,
        hidden_pad, intermediate_pad, is_shuffled,
    )

# Step 3: Clear the original cache and replace the function
fm.get_2stage_cfgs.cache_clear()
fm.get_2stage_cfgs = _patched_get_2stage_cfgs

# Also reset the global CSV dict so it reloads fresh
fm.cfg_2stages = None

# ============================================================================
# ADDITIONAL PATCHES
# ============================================================================

# Force use_nt=False for ALL shapes (confirmed better)
fm.use_nt = lambda t, k, e: False

# Force separate quant+sort (competitor "sepqsort" technique)
# This uses separate per_1x32_f4_quant_hip + moe_mxfp4_sort instead of
# the fused fused_dynamic_mxfp4_quant_moe_sort kernel
# The fused kernel has overhead from combined quant+sort logic
# NOTE: We set the switch threshold to 0 so ALL sizes use the fused path
#       (which is actually faster for small M due to fewer kernel launches)
# Actually, token_num_quant_moe_sort_switch=1024 means token<=1024 uses fused,
# token>1024 uses separate. For our shapes (16/128/512), all use fused already.
# Setting to 0 would force separate for all — worth testing but risky.
# Keep default for safety.

# ============================================================================
# ENTRY POINT
# ============================================================================

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
        gate_up_weight_shuffled,
        down_weight_shuffled,
        topk_weights,
        topk_ids,
        expert_mask=None,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad,
        intermediate_pad=intermediate_pad,
    )

import os
# =============================================================================
# Environment setup BEFORE any aiter imports
# =============================================================================
os.environ["CU_NUM"] = "256"        # Force MI355X config lookup match
os.environ["AITER_USE_NT"] = "0"    # Disable non-temporal loads globally

import torch
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

# =============================================================================
# MONKEY-PATCHES — applied at import time, before any kernel calls
# =============================================================================

# 1. Force use_nt=False for ALL shapes (confirmed better across all sizes)
fm.use_nt = lambda t, k, e: False

# 2. Force separate quant+sort ("sepqsort")
#    Competitors ry2009 and Ryan Mathieu both use this. The fused
#    fused_dynamic_mxfp4_quant_moe_sort kernel has overhead from combined
#    quant+sort logic. Two simpler kernels (quant then sort) are faster.
try:
    fm.token_num_quant_moe_sort_switch = 0
except AttributeError:
    pass

# 3. Per-shape block_m via get_block_size_M override
#    ry2009: "force32_128" = block_m=32 for small, 128 for large
#    Ryan Mathieu: "16128" = block_m=16 for E=257 small, 128 for large
_orig_get_block_size_M = fm.get_block_size_M

def _patched_get_block_size_M(token, topk, expert, inter_dim):
    est_m = (token * topk) / expert
    if expert > 64:
        # E=257: very sparse (0.56 tokens/expert at bs=16, topk=9)
        if est_m < 8:
            return 16      # Ryan Mathieu "16128" for small E=257
        elif est_m < 50:
            return 32
        else:
            return 128     # Ryan Mathieu "16128" for large E=257
    else:
        # E=33: moderate density
        if est_m < 16:
            return 32      # ry2009 "force32" for small E=33
        elif est_m < 100:
            return 64      # Default heuristic
        else:
            return 128     # ry2009 "128" for large E=33

fm.get_block_size_M = _patched_get_block_size_M

# 4. Per-shape ksplit via global variable + patched get_ksplit
#    get_ksplit() likely takes NO args (just reads AITER_KSPLIT env var).
#    We set a global _current_ksplit before each fused_moe call,
#    and our patched get_ksplit reads it.
_current_ksplit = 0  # default: let aiter decide

try:
    _orig_get_ksplit = fm.get_ksplit

    def _patched_get_ksplit(*args, **kwargs):
        return _current_ksplit

    fm.get_ksplit = _patched_get_ksplit
except AttributeError:
    pass


# =============================================================================
# Entry point
# =============================================================================
def custom_kernel(data):
    global _current_ksplit

    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data

    M = hidden_states.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1]
    topk = topk_ids.shape[1]

    # Padding info from config dict
    hidden_pad = config.get("hidden_pad", 0)
    intermediate_pad = config.get("intermediate_pad", 0)

    # Set per-shape ksplit BEFORE calling fused_moe
    # Bortlesboat (150.9μs): ksplit=2 for E=257
    # Ryan Mathieu (151μs): ksplit=2 for E=257 and E=33
    # CAUTION: ksplit=2 for E=33 d=2048 triggers cktile → 2x SLOWER
    if E > 64:
        # E=257: ksplit=2 (confirmed by top competitors)
        _current_ksplit = 2
    elif inter_dim <= 512:
        # E=33, d=512: ksplit=2 (Ryan Mathieu "33k2")
        _current_ksplit = 2
    else:
        # E=33, d=2048: ksplit=0 (default, avoid cktile 2x slowdown)
        _current_ksplit = 0

    # Call fused_moe with shuffled quantized weights
    # Monkey-patches active:
    #   - use_nt=False (via fm.use_nt lambda)
    #   - per-shape block_m (via fm.get_block_size_M)
    #   - per-shape ksplit (via fm.get_ksplit + _current_ksplit)
    #   - separate quant+sort (via fm.token_num_quant_moe_sort_switch=0)
    out = fused_moe(
        hidden_states,
        w1_qw, w2_qw,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1_qs, w2_scale=w2_qs,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

    return out

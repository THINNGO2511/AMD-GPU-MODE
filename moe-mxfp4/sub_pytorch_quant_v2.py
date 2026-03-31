#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — PyTorch quant replacement v2 (fully decoded zhubenzhu hint).

The exact mechanism:
- fused_moe.py uses `get_hip_quant` which returns `per_1x32_f4_quant_hip`
  → triggers CK JIT compilation (98-105s per module → timeout)
- Line 12 (commented) shows alternative: `get_torch_quant` → pure PyTorch, NO JIT
- Monkey-patch at module level to replace get_hip_quant → get_torch_quant

This does NOT change CK GEMM kernels → no JIT timeout.
Quant is 28% of kernel time → significant if pytorch is comparable speed.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === CORE FIX: Replace HIP quant with PyTorch quant ===
    # This avoids the CK JIT compilation that causes 130s timeout
    try:
        # Get the pytorch quant function
        torch_quant_getter = aiter.get_torch_quant

        # Replace get_hip_quant with get_torch_quant at the aiter module level
        # fused_moe.py imports: `from aiter import get_hip_quant as get_quant`
        # We need to replace it where it's imported
        if hasattr(fm, 'get_quant'):
            fm.get_quant = torch_quant_getter
        # Also try patching at aiter level
        if hasattr(aiter, 'get_hip_quant'):
            original_hip = aiter.get_hip_quant
            aiter.get_hip_quant = torch_quant_getter
    except Exception as e:
        print(f"Quant patch failed: {e}")

    # === use_nt=False for ALL shapes (proven 2-3% improvement) ===
    fm.use_nt = lambda token, topk, expert: False

    # === block_m tuning for E<=64 ===
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm


def custom_kernel(data: input_t) -> output_t:
    _patch()
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

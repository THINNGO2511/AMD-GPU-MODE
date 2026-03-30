import os
os.environ['CU_NUM'] = '256'
os.environ['AITER_USE_NT'] = '0'

from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType
import aiter.fused_moe as fm

# Disable NT store
fm.use_nt = lambda t, k, e: False

# Separate quant+sort
fm.token_num_quant_moe_sort_switch = 0


def custom_kernel(data):
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data

    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']

    num_experts = topk_ids.max().item() + 1
    num_tokens = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    est_m = (num_tokens * topk) / num_experts

    # ksplit=2 for large expert counts only (not E=33 which triggers slow cktile)
    ksplit = 2 if num_experts >= 257 else 1

    # block_m override
    if est_m < 50:
        block_m = 32
    else:
        block_m = 128

    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)


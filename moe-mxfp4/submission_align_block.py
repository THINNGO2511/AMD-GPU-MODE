#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Use moe_align_block_size for proper per-block expert IDs.
moe_align_block_size(topk_ids, num_experts, block_size,
    sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad) -> None
All output tensors must be pre-allocated.
"""
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
import functools

_probed = False
_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

def _patch():
    global _patched
    if _patched: return
    _patched = True
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
    try: fm._USE_OPUS_MOE_SORTING = True
    except: pass
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
        if expert <= 64 and q_type == QuantType.per_1x32 and not r.run_1stage and inter_dim < 2048:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(functools.partial(fm.ck_moe_stage1, kernelName=kn1, activation=activation, quant_type=q_type, dtype=dtype, splitk=0, use_non_temporal_load=False), functools.partial(aiter.ck_moe_stage2_fwd, kernelName=STAGE2_32, activation=activation, quant_type=q_type, use_non_temporal_load=False), 32, 0, False)
            except: pass
        return r
    fm.get_2stage_cfgs = new
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _probed
    _patch()
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _probed:
        _probed = True
        M = hidden_states.shape[0]
        E = gate_up_weight.shape[0]
        topk = topk_ids.shape[1]

        try:
            BLOCK_M = 32
            max_num_tokens_padded = M * topk + E * BLOCK_M
            # Pre-allocate output tensors
            sorted_token_ids = torch.empty(max_num_tokens_padded, dtype=torch.int32, device='cuda')
            expert_ids = torch.empty(max_num_tokens_padded // BLOCK_M, dtype=torch.int32, device='cuda')
            token_nums = torch.empty(E, dtype=torch.int32, device='cuda')
            num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device='cuda')

            # Call moe_align_block_size with pre-allocated tensors
            aiter.moe_align_block_size(
                topk_ids, E, BLOCK_M,
                sorted_token_ids, expert_ids, token_nums, num_tokens_post_pad)

            ntp = num_tokens_post_pad.item()
            num_blocks = ntp // BLOCK_M

            print(f"[ALIGN] sorted_token_ids: {sorted_token_ids.shape}")
            print(f"[ALIGN] expert_ids: {expert_ids.shape}")
            print(f"[ALIGN] token_nums[:10]: {token_nums[:10].tolist()}")
            print(f"[ALIGN] num_tokens_post_pad: {ntp}")
            print(f"[ALIGN] num_blocks: {num_blocks}")
            print(f"[ALIGN] expert_ids[:10]: {expert_ids[:10].tolist()}")
            print(f"[ALIGN] expert_ids range (first {num_blocks}): [{expert_ids[:num_blocks].min()}, {expert_ids[:num_blocks].max()}]")
            print(f"[ALIGN] sorted_ids[:10]: {sorted_token_ids[:10].tolist()}")
            print(f"[ALIGN] sorted_ids max (first {ntp}): {sorted_token_ids[:ntp].max()}")
            print(f"[ALIGN] M*topk={M*topk}")
        except Exception as e:
            import traceback
            print(f"[ALIGN] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

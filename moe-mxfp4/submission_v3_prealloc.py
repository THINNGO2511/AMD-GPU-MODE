#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE v3 — Aggressive optimization:
1. use_nt=False for ALL cases
2. CK kernel injection for E<=64 d<2048
3. Opus sorting
4. Pre-allocate sorting buffers (monkey-patch _moe_sorting_impl)
5. Force ksplit=0 to avoid slow cktile path
6. Smarter block_m selection
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

# Best CK kernel names
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Pre-allocated buffer cache
_buf_cache = {}


def _get_or_alloc(key, shape, dtype, device):
    """Get pre-allocated buffer or create one."""
    if key not in _buf_cache or _buf_cache[key].shape != shape:
        _buf_cache[key] = torch.empty(shape, dtype=dtype, device=device)
    return _buf_cache[key]


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Force use_nt=False for ALL cases
    fm.use_nt = lambda t, k, e: False

    # 2. Block_m selection with E<=64 optimization
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # 3. Force ksplit=0 to avoid slow cktile path for E<=64
    try:
        orig_ksplit = fm.get_ksplit
        def new_ksplit(token, topk, expert, inter_dim, model_dim):
            if expert <= 64:
                return 0  # Never use ksplit for small expert counts
            return orig_ksplit(token, topk, expert, inter_dim, model_dim)
        fm.get_ksplit = new_ksplit
    except Exception:
        pass

    # 4. Opus sorting
    try:
        fm._USE_OPUS_MOE_SORTING = True
    except Exception:
        pass

    # 5. Pre-allocate sorting buffers
    orig_sorting_impl = fm._moe_sorting_impl

    def fast_moe_sorting_impl(topk_ids, topk_weights, num_experts, model_dim,
                               moebuf_dtype, block_size, expert_mask,
                               num_local_tokens, dispatch_policy, use_opus):
        device = topk_ids.device
        M, topk = topk_ids.shape
        max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)

        cache_key = (M, topk, num_experts, model_dim, block_size, str(moebuf_dtype))
        if cache_key not in _buf_cache:
            _buf_cache[cache_key] = {
                'sorted_ids': torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
                'sorted_weights': torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
                'sorted_expert_ids': torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
                'num_valid_ids': torch.empty(2, dtype=dtypes.i32, device=device),
                'moe_buf': torch.empty((M, model_dim), dtype=moebuf_dtype, device=device),
            }
        bufs = _buf_cache[cache_key]
        sorted_ids = bufs['sorted_ids']
        sorted_weights = bufs['sorted_weights']
        sorted_expert_ids = bufs['sorted_expert_ids']
        num_valid_ids = bufs['num_valid_ids']
        moe_buf = bufs['moe_buf']

        fwd_fn = aiter.moe_sorting_opus_fwd if use_opus else aiter.moe_sorting_fwd
        fwd_fn(
            topk_ids, topk_weights,
            sorted_ids, sorted_weights, sorted_expert_ids,
            num_valid_ids, moe_buf,
            num_experts, block_size,
            expert_mask, num_local_tokens, dispatch_policy,
        )
        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf

    fm._moe_sorting_impl = fast_moe_sorting_impl

    # 6. Inject CK kernels for E<=64 d<2048
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)

        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


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

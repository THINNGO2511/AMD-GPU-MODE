#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Sort Cache: Cache moe_sorting results across GEMM stages.
Sort depends only on topk_ids/topk_weights — identical between stage1 and stage2.
Saves one sort kernel launch per fused_moe call (for token_num > 1024).
Base: optimized_v2 (169us) + sort caching.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_sort_cache = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. use_nt=False for ALL shapes (proven)
    fm.use_nt = lambda token, topk, expert: False

    # 2. Sort caching monkey-patch
    # moe_sorting called 2x per fused_moe (once per GEMM stage)
    # Sort order depends only on routing — cache indices, allocate fresh output buffer
    _orig_sort = None
    _sort_name = None
    for name in ['moe_sorting', '_moe_sorting_impl']:
        if hasattr(fm, name) and callable(getattr(fm, name)):
            _orig_sort = getattr(fm, name)
            _sort_name = name
            break

    if _orig_sort is not None:
        orig_sort = _orig_sort  # capture in closure

        def cached_sort(*args, **kwargs):
            topk_ids = args[0]
            cache_key = topk_ids.data_ptr()
            if cache_key in _sort_cache:
                cached_indices, buf_row_count = _sort_cache[cache_key]
                # Extract model_dim and dtype for fresh output buffer
                model_dim = args[3] if len(args) > 3 else kwargs.get('model_dim', 256)
                dtype = args[4] if len(args) > 4 else kwargs.get('dtype', torch.bfloat16)
                new_buf = torch.empty(buf_row_count, model_dim, dtype=dtype,
                                      device=topk_ids.device)
                return (*cached_indices, new_buf)
            result = orig_sort(*args, **kwargs)
            # Cache first 4 returns (sort indices) + output buffer row count
            _sort_cache[cache_key] = (result[:-1], result[-1].shape[0])
            return result

        setattr(fm, _sort_name, cached_sort)

    # 3. block_m tuning for E<=64 (proven)
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

    # 4. CK kernel injection for E<=64 d<2048 (proven)
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
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _sort_cache
    _sort_cache = {}  # Clear for new routing each call
    _patch()

    (hidden_states, gate_up_weight, down_weight,
     gate_up_weight_scale, down_weight_scale,
     gate_up_weight_shuffled, down_weight_shuffled,
     gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
     topk_weights, topk_ids, config) = data

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

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Direct pipeline: bypass fused_moe Python overhead.
Optimizations:
1. Pre-allocate sort buffers (reuse across calls)
2. Enable opus sorting
3. Direct pipeline (skip fused_moe dispatch overhead)
4. use_nt=False globally
5. CK kernel injection for E<=64 d<2048
6. Probe moe_sorting_opus_fwd availability
"""
import torch
import functools
import sys
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_dynamic_mxfp4_quant_moe_sort
import aiter.fused_moe as fm

_initialized = False
_call_count = 0

# CK kernel names
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# Cache for pre-allocated buffers and config
_sort_cache = {}
_cfg_cache = {}
_has_opus = None


def _init():
    """One-time initialization."""
    global _initialized, _has_opus
    if _initialized:
        return
    _initialized = True

    # Check if opus sorting is available
    try:
        _has_opus = hasattr(aiter, 'moe_sorting_opus_fwd')
        if _has_opus:
            fm._USE_OPUS_MOE_SORTING = True
            print(f"[INIT] Opus sorting: ENABLED")
        else:
            print(f"[INIT] Opus sorting: NOT AVAILABLE")
    except Exception as e:
        _has_opus = False
        print(f"[INIT] Opus sorting check failed: {e}")

    # Patch use_nt to False for ALL cases (not just E<=64)
    fm.use_nt = lambda t, k, e: False

    # Patch block_m: 32 for small batches, 64 for large (E<=64 only)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    # Patch get_2stage_cfgs for CK kernel injection
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

    print(f"[INIT] use_nt=False globally, CK injection for E<=64 d<2048")
    sys.stdout.flush()


def _get_sort_bufs(M, topk, num_experts, model_dim, block_size, device):
    """Get pre-allocated sort buffers, creating if needed."""
    key = (M, topk, num_experts, model_dim, block_size)
    if key not in _sort_cache:
        max_num_tokens_padded = int(M * topk + num_experts * block_size - topk)
        max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
        _sort_cache[key] = {
            'sorted_ids': torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device),
            'sorted_weights': torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device),
            'sorted_expert_ids': torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device),
            'num_valid_ids': torch.empty(2, dtype=dtypes.i32, device=device),
            'moe_buf': torch.empty((M, model_dim), dtype=torch.bfloat16, device=device),
        }
    return _sort_cache[key]


def _direct_sort(topk_ids, topk_weights, num_experts, model_dim, block_size, device):
    """Sort with pre-allocated buffers."""
    M, topk = topk_ids.shape
    bufs = _get_sort_bufs(M, topk, num_experts, model_dim, block_size, device)

    fwd_fn = aiter.moe_sorting_opus_fwd if _has_opus else aiter.moe_sorting_fwd
    fwd_fn(
        topk_ids, topk_weights,
        bufs['sorted_ids'], bufs['sorted_weights'],
        bufs['sorted_expert_ids'], bufs['num_valid_ids'],
        bufs['moe_buf'],
        num_experts, int(block_size),
        None, None, 0,
    )
    return (bufs['sorted_ids'], bufs['sorted_weights'],
            bufs['sorted_expert_ids'], bufs['num_valid_ids'], bufs['moe_buf'])


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _init()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    _, model_dim, inter_dim = fm.get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)

    _call_count += 1

    # Use direct pipeline for hot path
    try:
        # Get block_m and config
        padded_M = fm.get_padded_M(M)
        block_m = fm.get_block_size_M(padded_M, topk, E, inter_dim)

        # Step 1: Sort (with pre-allocated buffers)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = \
            _direct_sort(topk_ids, topk_weights, E, model_dim, block_m, hidden_states.device)

        # Step 2: Input quantization
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=block_m)

        # Step 3: Get stage configs (cached)
        isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
        q_dtype_a = torch.float4_e2m1fn_x2
        q_dtype_w = torch.float4_e2m1fn_x2
        cfg = fm.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
            isG1U1, ActivationType.Silu, False,
            hidden_pad, intermediate_pad, True)

        if cfg.run_1stage:
            # 1-stage path (unlikely for our cases)
            return fused_moe(
                hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                topk_weights, topk_ids, expert_mask=None,
                activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

        # Step 4: Stage 1 GEMM + SiLU
        a2 = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=hidden_states.device)
        w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)

        cfg.stage1(
            a1, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
            block_m=block_m if cfg.block_m == 0 else cfg.block_m,
            a1_scale=a1_scale, w1_scale=w1_scale_v)

        # Step 5: Inter-stage quantization
        a2_flat = a2.view(-1, inter_dim)
        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m)
        a2_q = a2_q.view(M, topk, -1)

        # Step 6: Stage 2 GEMM + weighted accumulation
        w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)

        cfg.stage2(
            a2_q, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
            block_m=block_m if cfg.block_m == 0 else cfg.block_m,
            a2_scale=a2_scale, w2_scale=w2_scale_v,
            sorted_weights=sorted_weights)

        # Trim output
        d_hidden = config["d_hidden"]
        if moe_buf.shape[1] > d_hidden:
            return moe_buf[:, :d_hidden].contiguous()
        return moe_buf

    except Exception as e:
        import traceback
        if _call_count <= 7:
            print(f"[DIRECT ERR] M={M} E={E} d={inter_dim}: {e}")
            traceback.print_exc()
            sys.stdout.flush()

        # Fallback to standard fused_moe
        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

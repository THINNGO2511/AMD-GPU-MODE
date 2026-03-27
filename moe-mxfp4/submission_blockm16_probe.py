#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Direct pipeline with block_m=16 for sparse cases + source dump.
block_m=16 for E=257 reduces padding waste by ~50%.
Also dumps fused_moe source to find hidden optimization hooks.
"""
import torch
import functools
import sys
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_dynamic_mxfp4_quant_moe_sort
import aiter.fused_moe as fm

_initialized = False
_call_count = 0

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

_sort_cache = {}


def _init():
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Enable opus sorting
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # use_nt=False globally
    fm.use_nt = lambda t, k, e: False

    # block_m: 16 for very sparse (est_m < 10), 32 for small, 64 for large
    orig_bsm = fm.get_block_size_M
    def new_bsm(t, k, e, d):
        if e > 64:
            est_m = t * k // e
            if est_m < 10:
                return 16
            return 32
        else:
            est_m = t * k // e
            if est_m < 10:
                return 16
            elif est_m < 50:
                return 32
            else:
                return 64
    fm.get_block_size_M = new_bsm

    # CK kernel injection for E<=64 d<2048
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

    # Dump fused_moe source
    try:
        src = inspect.getsource(fused_moe)
        print(f"[DUMP] fused_moe source ({len(src)} chars):")
        for line in src.split('\n'):
            print(f"  {line}")
    except Exception as e:
        print(f"[DUMP] fused_moe source error: {e}")

    # List available CK kernel binaries
    import os
    try:
        moe_dir = "/home/runner/aiter/hsa/gfx950/"
        if os.path.isdir(moe_dir):
            subdirs = os.listdir(moe_dir)
            print(f"\n[DUMP] gfx950 subdirs: {subdirs}")
            for sd in subdirs:
                sd_path = os.path.join(moe_dir, sd)
                if os.path.isdir(sd_path):
                    files = os.listdir(sd_path)[:20]
                    print(f"  {sd}/: {len(os.listdir(sd_path))} files, first 20: {files}")
    except Exception as e:
        print(f"[DUMP] kernel dir error: {e}")

    # Read tuned CSV configs
    try:
        csv_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"\n[DUMP] dsv3_fp4_tuned_fmoe.csv ({len(lines)} lines):")
        for line in lines:
            print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"[DUMP] CSV error: {e}")

    # Check tuned_fmoe.csv too
    try:
        csv_path2 = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
        with open(csv_path2) as f:
            lines2 = f.readlines()
        print(f"\n[DUMP] tuned_fmoe.csv ({len(lines2)} lines):")
        for line in lines2[:30]:
            print(f"  {line.rstrip()}")
    except Exception as e:
        print(f"[DUMP] tuned_fmoe CSV error: {e}")

    sys.stdout.flush()


def _get_sort_bufs(M, topk, num_experts, model_dim, block_size, device):
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

    try:
        padded_M = fm.get_padded_M(M)
        block_m = fm.get_block_size_M(padded_M, topk, E, inter_dim)

        # Pre-allocated sort
        bufs = _get_sort_bufs(M, topk, E, model_dim, block_m, hidden_states.device)
        fwd_fn = aiter.moe_sorting_opus_fwd if fm._USE_OPUS_MOE_SORTING else aiter.moe_sorting_fwd
        fwd_fn(topk_ids, topk_weights,
               bufs['sorted_ids'], bufs['sorted_weights'],
               bufs['sorted_expert_ids'], bufs['num_valid_ids'],
               bufs['moe_buf'], E, int(block_m), None, None, 0)

        sorted_ids = bufs['sorted_ids']
        sorted_weights = bufs['sorted_weights']
        sorted_expert_ids = bufs['sorted_expert_ids']
        num_valid_ids = bufs['num_valid_ids']
        moe_buf = bufs['moe_buf']

        # Input quant
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=block_m)

        # Get stage config
        isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
        cfg = fm.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            torch.bfloat16, torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2,
            QuantType.per_1x32, isG1U1, ActivationType.Silu, False,
            hidden_pad, intermediate_pad, True)

        if cfg.run_1stage:
            return fused_moe(
                hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                topk_weights, topk_ids, expert_mask=None,
                activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

        # Stage 1
        a2 = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=hidden_states.device)
        w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        bm = block_m if cfg.block_m == 0 else cfg.block_m
        cfg.stage1(a1, gate_up_weight_shuffled, down_weight_shuffled,
                   sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
                   block_m=bm, a1_scale=a1_scale, w1_scale=w1_scale_v)

        # Inter-stage quant
        a2_flat = a2.view(-1, inter_dim)
        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m)
        a2_q = a2_q.view(M, topk, -1)

        # Stage 2
        w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        cfg.stage2(a2_q, gate_up_weight_shuffled, down_weight_shuffled,
                   sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
                   block_m=bm, a2_scale=a2_scale, w2_scale=w2_scale_v,
                   sorted_weights=sorted_weights)

        d_hidden = config["d_hidden"]
        if moe_buf.shape[1] > d_hidden:
            return moe_buf[:, :d_hidden].contiguous()
        return moe_buf

    except Exception as e:
        if _call_count <= 7:
            import traceback
            print(f"[ERR] M={M} E={E} d={inter_dim} block_m={block_m if 'block_m' in dir() else '?'}: {e}")
            traceback.print_exc()
            sys.stdout.flush()

        return fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

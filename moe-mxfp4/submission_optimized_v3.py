#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Optimized v3: All optimizations + info gathering.
1. Direct pipeline (bypass fused_moe Python overhead ~15-25us)
2. Pre-allocated sort buffers
3. Opus sorting
4. use_nt=False globally (not just E<=64)
5. CK kernel injection for E<=64 d<2048
6. Pre-allocated a2 buffer
7. Dump CSV configs + fused_moe source for analysis
"""
import torch
import functools
import sys
import os
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
_a2_cache = {}
_direct_failed = set()


def _dump_info():
    """Dump CSV configs and source for analysis."""
    import inspect

    # 1. DSv3 tuned CSV
    for csv_path in [
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    ]:
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            print(f"\n[CSV] {csv_path} ({len(lines)} lines):")
            for line in lines[:50]:
                print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"[CSV] {csv_path}: {e}")

    # 2. fused_moe_2stages source (key function)
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        lines = src.split('\n')
        print(f"\n[SRC] fused_moe_2stages ({len(lines)} lines):")
        for i, line in enumerate(lines[:120]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"[SRC] fused_moe_2stages error: {e}")

    # 3. get_2stage_cfgs source (kernel matching logic)
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\n[SRC] get_2stage_cfgs ({len(lines)} lines):")
        for i, line in enumerate(lines[:150]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"[SRC] get_2stage_cfgs error: {e}")

    # 4. CK kernel binary directory
    try:
        fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe"
        if os.path.isdir(fmoe_dir):
            for root, dirs, files in os.walk(fmoe_dir):
                for f in files[:30]:
                    print(f"[BIN] {os.path.join(root, f)}")
    except Exception as e:
        print(f"[BIN] error: {e}")

    sys.stdout.flush()


def _init():
    global _initialized
    if _initialized:
        return
    _initialized = True

    # Opus sorting
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True

    # use_nt=False globally
    fm.use_nt = lambda t, k, e: False

    # block_m: safe defaults for fallback path
    orig_bsm = fm.get_block_size_M
    def new_bsm(t, k, e, d):
        est_m = t * k // e
        if e <= 64:
            return 32 if est_m < 50 else 64
        return 32
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

    # Dump info on first init
    try:
        _dump_info()
    except Exception:
        pass


def _get_sort_bufs(M, topk, E, model_dim, block_m, device):
    key = (M, topk, E, model_dim, block_m)
    if key not in _sort_cache:
        max_padded = int(M * topk + E * block_m - topk)
        max_blocks = int((max_padded + block_m - 1) // block_m)
        _sort_cache[key] = (
            torch.empty(max_padded, dtype=dtypes.i32, device=device),
            torch.empty(max_padded, dtype=dtypes.fp32, device=device),
            torch.empty(max_blocks, dtype=dtypes.i32, device=device),
            torch.empty(2, dtype=dtypes.i32, device=device),
            torch.empty((M, model_dim), dtype=torch.bfloat16, device=device),
        )
    return _sort_cache[key]


def _run_direct(hidden_states, w1_sh, w2_sh, w1_scale_sh, w2_scale_sh,
                topk_weights, topk_ids, config):
    M = hidden_states.shape[0]
    E = w1_sh.shape[0]
    topk = topk_ids.shape[1]
    _, model_dim, inter_dim = fm.get_inter_dim(w1_sh.shape, w2_sh.shape)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    d_hidden = config["d_hidden"]

    padded_M = fm.get_padded_M(M)
    block_m = fm.get_block_size_M(padded_M, topk, E, inter_dim)

    # Sort
    bufs = _get_sort_bufs(M, topk, E, model_dim, block_m, hidden_states.device)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = bufs

    fwd_fn = aiter.moe_sorting_opus_fwd if fm._USE_OPUS_MOE_SORTING else aiter.moe_sorting_fwd
    fwd_fn(topk_ids, topk_weights, sorted_ids, sorted_weights,
           sorted_expert_ids, num_valid_ids, moe_buf,
           E, int(block_m), None, None, 0)

    # Input quant
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m)

    # Config
    isG1U1 = inter_dim != w1_sh.shape[1]
    cfg = fm.get_2stage_cfgs(
        padded_M, model_dim, inter_dim, E, topk,
        torch.bfloat16, torch.float4_e2m1fn_x2, torch.float4_e2m1fn_x2,
        QuantType.per_1x32, isG1U1, ActivationType.Silu, False,
        hidden_pad, intermediate_pad, True)

    if cfg.run_1stage:
        raise RuntimeError("1-stage")

    # Stage 1
    a2_key = (M, topk, inter_dim)
    if a2_key not in _a2_cache:
        _a2_cache[a2_key] = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device=hidden_states.device)
    a2 = _a2_cache[a2_key]

    w1s = w1_scale_sh.view(dtypes.fp8_e8m0)
    bm = block_m if cfg.block_m == 0 else cfg.block_m
    cfg.stage1(a1, w1_sh, w2_sh, sorted_ids, sorted_expert_ids, num_valid_ids,
               a2, topk, block_m=bm, a1_scale=a1_scale, w1_scale=w1s)

    # Inter-stage quant
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m)
    a2_q = a2_q.view(M, topk, -1)

    # Stage 2
    w2s = w2_scale_sh.view(dtypes.fp8_e8m0)
    cfg.stage2(a2_q, w1_sh, w2_sh, sorted_ids, sorted_expert_ids, num_valid_ids,
               moe_buf, topk, block_m=bm, a2_scale=a2_scale, w2_scale=w2s,
               sorted_weights=sorted_weights)

    if moe_buf.shape[1] > d_hidden:
        return moe_buf[:, :d_hidden].contiguous()
    return moe_buf


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _init()

    (
        hidden_states, _, _, _, _,
        w1_sh, w2_sh, w1_scale_sh, w2_scale_sh,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    M = hidden_states.shape[0]
    E = w1_sh.shape[0]
    _, _, inter_dim = fm.get_inter_dim(w1_sh.shape, w2_sh.shape)
    _call_count += 1

    cfg_key = (M, E, inter_dim)
    if cfg_key not in _direct_failed:
        try:
            return _run_direct(hidden_states, w1_sh, w2_sh, w1_scale_sh, w2_scale_sh,
                               topk_weights, topk_ids, config)
        except Exception as e:
            _direct_failed.add(cfg_key)
            if _call_count <= 10:
                import traceback
                print(f"[DIRECT ERR] M={M} E={E} d={inter_dim}: {e}")
                traceback.print_exc()
                sys.stdout.flush()

    return fused_moe(
        hidden_states, w1_sh, w2_sh, topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1_scale_sh, w2_scale=w2_scale_sh,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Profile each phase of the 2-stage pipeline.
Measure: sorting, input quant, stage1, inter-stage quant, stage2.
"""
import torch
import functools
import time
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_profiled = False
_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
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
            except:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _profile_phases(data):
    """Run MoE with per-phase timing."""
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    _, model_dim, inter_dim = fm.get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Phase timings
    torch.cuda.synchronize()

    # 1. Sorting
    t0 = time.perf_counter()
    block_m = fm.get_block_size_M(fm.get_padded_M(M), topk, E, inter_dim)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
        topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m, None, None, 0)
    torch.cuda.synchronize()
    t_sort = time.perf_counter() - t0

    # 2. Input quant
    t0 = time.perf_counter()
    from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m)
    torch.cuda.synchronize()
    t_quant1 = time.perf_counter() - t0

    # 3. Stage 1
    a2 = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device='cuda')
    w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    t0 = time.perf_counter()
    fm.ck_moe_stage1(
        a1, gate_up_weight_shuffled, down_weight_shuffled,
        sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
        kernelName=STAGE1_64, block_m=block_m,
        a1_scale=a1_scale, w1_scale=w1_scale_v,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        splitk=0, use_non_temporal_load=False)
    torch.cuda.synchronize()
    t_stage1 = time.perf_counter() - t0

    # 4. Inter-stage quant
    a2_flat = a2.view(-1, inter_dim)
    t0 = time.perf_counter()
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m)
    a2_q = a2_q.view(M, topk, -1)
    torch.cuda.synchronize()
    t_quant2 = time.perf_counter() - t0

    # 5. Stage 2
    w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    t0 = time.perf_counter()
    aiter.ck_moe_stage2_fwd(
        a2_q, gate_up_weight_shuffled, down_weight_shuffled,
        sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
        kernelName=STAGE2_32, block_m=block_m,
        a2_scale=a2_scale, w2_scale=w2_scale_v,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        sorted_weights=sorted_weights, use_non_temporal_load=False)
    torch.cuda.synchronize()
    t_stage2 = time.perf_counter() - t0

    total = t_sort + t_quant1 + t_stage1 + t_quant2 + t_stage2
    print(f"[PROF] M={M} E={E} d={inter_dim}: "
          f"sort={t_sort*1e3:.1f}ms quant1={t_quant1*1e3:.1f}ms "
          f"stage1={t_stage1*1e3:.1f}ms quant2={t_quant2*1e3:.1f}ms "
          f"stage2={t_stage2*1e3:.1f}ms total={total*1e3:.1f}ms")


def custom_kernel(data: input_t) -> output_t:
    global _profiled
    _patch()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    if not _profiled:
        _profiled = True
        try:
            # Warmup
            hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
            intermediate_pad = config["d_expert_pad"] - config["d_expert"]
            fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                      topk_weights, topk_ids, expert_mask=None,
                      activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                      doweight_stage1=False,
                      w1_scale=gate_up_weight_scale_shuffled,
                      w2_scale=down_weight_scale_shuffled,
                      hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
            torch.cuda.synchronize()
            # Profile
            _profile_phases(data)
        except Exception as e:
            import traceback
            print(f"[PROF] error: {e}")
            traceback.print_exc()

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

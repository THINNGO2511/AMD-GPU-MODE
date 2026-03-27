#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — GPU-accurate profiling of each pipeline phase using CUDA events.
Profile ALL calls to find where time is spent per benchmark case.
"""
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
from aiter.utility import fp4_utils

_patched = False
_call_count = 0

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


def _profile_manual(data):
    """Manually run each phase with CUDA event timing."""
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
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    block_m = fm.get_block_size_M(fm.get_padded_M(M), topk, E, inter_dim)

    events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
    torch.cuda.synchronize()

    events[0].record()
    # 1. Sorting
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
        topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m, None, None, 0)
    events[1].record()

    # 2. Input quant
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=1, block_size=block_m)
    events[2].record()

    # 3. Stage 1
    a2 = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device='cuda')
    w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)

    est_m = fm.get_padded_M(M) * topk // E
    kn1 = STAGE1_256 if (E <= 64 and inter_dim < 2048 and est_m >= 100) else (STAGE1_64 if E <= 64 and inter_dim < 2048 else "")

    fm.ck_moe_stage1(
        a1, gate_up_weight_shuffled, down_weight_shuffled,
        sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
        kernelName=kn1, block_m=block_m,
        a1_scale=a1_scale, w1_scale=w1_scale_v,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        splitk=0, use_non_temporal_load=False)
    events[3].record()

    # 4. Inter-stage quant
    a2_flat = a2.view(-1, inter_dim)
    a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=M, topk=topk, block_size=block_m)
    a2_q = a2_q.view(M, topk, -1)
    events[4].record()

    # 5. Stage 2
    w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
    kn2 = STAGE2_32 if E <= 64 and inter_dim < 2048 else ""
    aiter.ck_moe_stage2_fwd(
        a2_q, gate_up_weight_shuffled, down_weight_shuffled,
        sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
        kernelName=kn2, block_m=block_m,
        a2_scale=a2_scale, w2_scale=w2_scale_v,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        sorted_weights=sorted_weights, use_non_temporal_load=False)
    events[5].record()

    # Total
    events[6].record()
    torch.cuda.synchronize()

    t_sort = events[0].elapsed_time(events[1])
    t_q1 = events[1].elapsed_time(events[2])
    t_s1 = events[2].elapsed_time(events[3])
    t_q2 = events[3].elapsed_time(events[4])
    t_s2 = events[4].elapsed_time(events[5])
    t_total = events[0].elapsed_time(events[5])

    print(f"[GPU] M={M} E={E} d={inter_dim}: "
          f"sort={t_sort*1000:.0f}us q1={t_q1*1000:.0f}us "
          f"s1={t_s1*1000:.0f}us q2={t_q2*1000:.0f}us "
          f"s2={t_s2*1000:.0f}us | total={t_total*1000:.0f}us")


def custom_kernel(data: input_t) -> output_t:
    global _call_count
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

    _call_count += 1
    # Profile on 2nd call (after warmup) for each size
    if _call_count <= 7:
        try:
            # Warmup first
            fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                      topk_weights, topk_ids, expert_mask=None,
                      activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                      doweight_stage1=False,
                      w1_scale=gate_up_weight_scale_shuffled,
                      w2_scale=down_weight_scale_shuffled,
                      hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
            torch.cuda.synchronize()
            _profile_manual(data)
        except Exception as e:
            import traceback
            print(f"[PROF ERR] {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

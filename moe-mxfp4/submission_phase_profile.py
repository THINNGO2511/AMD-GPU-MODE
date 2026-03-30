#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — GPU-accurate per-phase profiling using cuda events.
Profiles all 7 test cases: sort, quant1, stage1, quant2, stage2.
Also probes moe_sorting internals and sort kernel info.
"""
import torch
import functools
import sys
import os
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

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


def _gpu_profile(data):
    """GPU-accurate per-phase timing with cuda events."""
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

    from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort

    N_RUNS = 5
    times = {'sort': [], 'quant1': [], 'stage1': [], 'quant2': [], 'stage2': [], 'total': []}

    for run in range(N_RUNS):
        events = {}
        for name in ['start', 'after_sort', 'after_q1', 'after_s1', 'after_q2', 'after_s2']:
            events[name] = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()

        # Total fused_moe timing
        t_start = torch.cuda.Event(enable_timing=True)
        t_end = torch.cuda.Event(enable_timing=True)
        t_start.record()
        _ = fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
        t_end.record()
        torch.cuda.synchronize()
        times['total'].append(t_start.elapsed_time(t_end))

        # Per-phase timing
        events['start'].record()

        # 1. Sorting
        block_m = fm.get_block_size_M(fm.get_padded_M(M), topk, E, inter_dim)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
            topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m, None, None, 0)
        events['after_sort'].record()

        # 2. Input quant
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=block_m)
        events['after_q1'].record()

        # 3. Stage 1
        a2 = torch.empty((M, topk, inter_dim), dtype=torch.bfloat16, device='cuda')
        w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        fm.ck_moe_stage1(
            a1, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
            kernelName=STAGE1_64, block_m=block_m,
            a1_scale=a1_scale, w1_scale=w1_scale_v,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            splitk=0, use_non_temporal_load=False)
        events['after_s1'].record()

        # 4. Inter-stage quant
        a2_flat = a2.view(-1, inter_dim)
        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2_flat, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m)
        a2_q = a2_q.view(M, topk, -1)
        events['after_q2'].record()

        # 5. Stage 2
        w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        aiter.ck_moe_stage2_fwd(
            a2_q, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
            kernelName=STAGE2_32, block_m=block_m,
            a2_scale=a2_scale, w2_scale=w2_scale_v,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            sorted_weights=sorted_weights, use_non_temporal_load=False)
        events['after_s2'].record()

        torch.cuda.synchronize()

        times['sort'].append(events['start'].elapsed_time(events['after_sort']))
        times['quant1'].append(events['after_sort'].elapsed_time(events['after_q1']))
        times['stage1'].append(events['after_q1'].elapsed_time(events['after_s1']))
        times['quant2'].append(events['after_s1'].elapsed_time(events['after_q2']))
        times['stage2'].append(events['after_q2'].elapsed_time(events['after_s2']))

    # Report median
    def med(lst):
        s = sorted(lst)
        return s[len(s)//2]

    sort_ms = med(times['sort'])
    q1_ms = med(times['quant1'])
    s1_ms = med(times['stage1'])
    q2_ms = med(times['quant2'])
    s2_ms = med(times['stage2'])
    total_ms = med(times['total'])
    phases_ms = sort_ms + q1_ms + s1_ms + q2_ms + s2_ms

    print(f"[PROF] M={M} E={E} d={inter_dim} topk={topk} block_m={block_m}")
    print(f"  sort:   {sort_ms:.3f}ms ({sort_ms/phases_ms*100:.1f}%)")
    print(f"  quant1: {q1_ms:.3f}ms ({q1_ms/phases_ms*100:.1f}%)")
    print(f"  stage1: {s1_ms:.3f}ms ({s1_ms/phases_ms*100:.1f}%)")
    print(f"  quant2: {q2_ms:.3f}ms ({q2_ms/phases_ms*100:.1f}%)")
    print(f"  stage2: {s2_ms:.3f}ms ({s2_ms/phases_ms*100:.1f}%)")
    print(f"  phases: {phases_ms:.3f}ms  fused_moe: {total_ms:.3f}ms  overhead: {phases_ms-total_ms:.3f}ms")
    sys.stdout.flush()


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

    # Profile first call of each config (first 7 calls = 7 test cases)
    if _call_count <= 7:
        M = hidden_states.shape[0]
        E = gate_up_weight_shuffled.shape[0]
        _, _, inter_dim = fm.get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)

        # Warmup this config
        _ = fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
        torch.cuda.synchronize()

        try:
            _gpu_profile(data)
        except Exception as e:
            import traceback
            print(f"[PROF] error: {e}")
            traceback.print_exc()

    # Also probe sort kernel on first call
    if _call_count == 1:
        try:
            # Probe what moe_sorting uses
            print(f"\n[PROBE] moe_sorting function: {fm.moe_sorting}")
            print(f"[PROBE] dir(fm) sort-related: {[x for x in dir(fm) if 'sort' in x.lower()]}")
            print(f"[PROBE] _USE_OPUS_MOE_SORTING: {getattr(fm, '_USE_OPUS_MOE_SORTING', 'N/A')}")

            # Check for Triton sort kernels
            try:
                from aiter.ops.triton.moe import moe_align_block_size as mabs
                print(f"[PROBE] moe_align_block_size: {mabs}")
                if hasattr(mabs, '__file__'):
                    print(f"[PROBE] mabs file: {mabs.__file__}")
            except Exception as e:
                print(f"[PROBE] moe_align_block_size import failed: {e}")

            # Check for sort kernel source
            import inspect
            try:
                src = inspect.getsource(fm.moe_sorting)
                # Print first 50 lines
                lines = src.split('\n')[:50]
                print(f"[PROBE] moe_sorting source ({len(src)} chars):")
                for line in lines:
                    print(f"  {line}")
            except:
                print(f"[PROBE] Cannot get moe_sorting source")

            # Check fused_dynamic_mxfp4_quant_moe_sort
            try:
                from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
                src2 = inspect.getsource(fused_dynamic_mxfp4_quant_moe_sort)
                lines2 = src2.split('\n')[:30]
                print(f"[PROBE] fused_dynamic_mxfp4_quant_moe_sort source:")
                for line in lines2:
                    print(f"  {line}")
            except:
                print(f"[PROBE] Cannot get quant source")

            # Probe Triton MoE kernel availability
            try:
                from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu
                print(f"[PROBE] fused_moe_mxfp4_silu available: {fused_moe_mxfp4_silu}")
                src3 = inspect.getsource(fused_moe_mxfp4_silu)
                lines3 = src3.split('\n')[:40]
                print(f"[PROBE] fused_moe_mxfp4_silu source:")
                for line in lines3:
                    print(f"  {line}")
            except Exception as e:
                print(f"[PROBE] fused_moe_mxfp4_silu import: {e}")

            try:
                from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4
                print(f"[PROBE] fused_moe_mxfp4 available: {fused_moe_mxfp4}")
            except Exception as e:
                print(f"[PROBE] fused_moe_mxfp4 import: {e}")

            # Check num_stages in Triton compile context
            print(f"[PROBE] TRITON_NUM_STAGES env: {os.environ.get('TRITON_NUM_STAGES', 'not set')}")

            # Check available env vars
            triton_envs = {k: v for k, v in os.environ.items() if 'TRITON' in k.upper()}
            print(f"[PROBE] Triton env vars: {triton_envs}")

        except Exception as e:
            import traceback
            print(f"[PROBE] error: {e}")
            traceback.print_exc()

    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

#!POPCORN gpu MI355X
"""
MoE Profiling Submission
========================
Designed for --mode profile to get detailed kernel-level profiling data.
Also includes in-submission torch.profiler for extra detail and fallback timing.

Usage:
  popcorn submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode profile sub_profile.py --no-tui

The eval.py profile mode (line 307-329) wraps custom_kernel() in
torch.profiler and returns a base64-encoded table of top 20 kernels by
CUDA time. This submission ALSO runs its own profiler on the first few
calls so we get printed output in stdout regardless.

Three profiling layers:
  1. eval.py --mode profile: wraps custom_kernel in torch.profiler (automatic)
  2. In-submission torch.profiler: runs on warmup calls, prints table + saves trace
  3. Fallback CUDA event timing: if torch.profiler fails, uses manual events
"""
import torch
import os
import sys
import time
import functools
import traceback
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

# ─── State ───────────────────────────────────────────────────────────────────
_patched = False
_call_count = 0
_profiled_shapes = set()

# ─── CK Kernel names (proven on MI355X) ─────────────────────────────────────
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

PROFILE_DIR = "/tmp/profiler_data"


def _patch():
    """Apply standard MoE optimizations (use_nt=False, block_m, kernel injection)."""
    global _patched
    if _patched:
        return
    _patched = True

    # Disable non-temporal loads for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda token, topk, expert: False if expert <= 64 else orig_use_nt(token, topk, expert)

    # Tuned block_m: 32 for small est_m, 64 for large
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Inject optimized CK kernels for E=33, d=512
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


def _run_torch_profiler(data, hidden_pad, intermediate_pad):
    """Run torch.profiler around fused_moe and print/save results."""
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
    d_expert = config.get("d_expert", 0)

    shape_key = f"M{M}_E{E}_d{d_expert}"

    try:
        from torch.profiler import profile, ProfilerActivity

        # Warmup call (outside profiler)
        _ = fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
        torch.cuda.synchronize()

        # Profiled call
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            _ = fused_moe(
                hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                topk_weights, topk_ids, expert_mask=None,
                activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
            )
            torch.cuda.synchronize()

        # Print table
        table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30)
        print(f"\n{'='*80}")
        print(f"TORCH PROFILER: {shape_key}")
        print(f"{'='*80}")
        print(table)

        # Also print grouped by input shape
        table_shapes = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=20)
        print(f"\n--- Grouped by input shape ({shape_key}) ---")
        print(table_shapes)

        # Try to save chrome trace
        try:
            os.makedirs(PROFILE_DIR, exist_ok=True)
            trace_path = os.path.join(PROFILE_DIR, f"moe_trace_{shape_key}.json")
            prof.export_chrome_trace(trace_path)
            print(f"[TRACE] Saved to {trace_path}")
        except Exception as e:
            print(f"[TRACE] Could not save trace: {e}")

        return True

    except Exception as e:
        print(f"[PROF] torch.profiler failed: {e}")
        traceback.print_exc()
        return False


def _run_cuda_event_timing(data, hidden_pad, intermediate_pad):
    """Fallback: time the entire fused_moe call with CUDA events."""
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    d_expert = config.get("d_expert", 0)

    # Warmup
    _ = fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )
    torch.cuda.synchronize()

    # Timed runs (5 iterations)
    times_us = []
    for i in range(5):
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()

        start_ev.record()
        _ = fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
        end_ev.record()
        torch.cuda.synchronize()
        times_us.append(start_ev.elapsed_time(end_ev) * 1000)

    avg = sum(times_us) / len(times_us)
    mn = min(times_us)
    mx = max(times_us)
    print(f"[CUDA_EVENT] M={M} E={E} d={d_expert}: "
          f"avg={avg:.1f}us min={mn:.1f}us max={mx:.1f}us "
          f"runs={[f'{t:.1f}' for t in times_us]}")

    # Also time with wall clock
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    for _ in range(10):
        _ = fused_moe(
            hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
            topk_weights, topk_ids, expert_mask=None,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=gate_up_weight_scale_shuffled,
            w2_scale=down_weight_scale_shuffled,
            hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
        )
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    wall_us = (t1 - t0) / 10 / 1000
    print(f"[WALL] M={M} E={E} d={d_expert}: avg={wall_us:.1f}us over 10 calls")


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

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    d_expert = config.get("d_expert", 0)
    shape_key = (M, E, d_expert)

    _call_count += 1

    # Profile each unique shape once (on first encounter)
    # Check if eval.py is already running a profiler around us (--mode profile).
    # If so, skip our internal profiler to avoid nesting issues -- eval.py will
    # capture the data. We still print shape/config info.
    already_profiled_externally = False
    try:
        from torch.profiler import profile as _prof_cls
        # torch.autograd.profiler.profile has _is_active when running
        if hasattr(torch.autograd.profiler, '_is_profiler_enabled'):
            already_profiled_externally = torch.autograd.profiler._is_profiler_enabled
        elif torch.autograd.profiler_util is not None:
            pass  # no reliable check, proceed with internal profiling
    except Exception:
        pass

    if shape_key not in _profiled_shapes:
        _profiled_shapes.add(shape_key)
        print(f"\n{'#'*80}")
        print(f"# PROFILING shape: M={M}, E={E}, d_expert={d_expert}, "
              f"topk={topk_ids.shape[1]}, config={config}")
        print(f"{'#'*80}")

        # Print tensor info
        print(f"  hidden_states: {hidden_states.shape} {hidden_states.dtype}")
        print(f"  gate_up_weight_shuffled: {gate_up_weight_shuffled.shape} {gate_up_weight_shuffled.dtype}")
        print(f"  down_weight_shuffled: {down_weight_shuffled.shape} {down_weight_shuffled.dtype}")
        print(f"  gate_up_weight_scale_shuffled: {gate_up_weight_scale_shuffled.shape} {gate_up_weight_scale_shuffled.dtype}")
        print(f"  down_weight_scale_shuffled: {down_weight_scale_shuffled.shape} {down_weight_scale_shuffled.dtype}")
        print(f"  topk_weights: {topk_weights.shape}, topk_ids: {topk_ids.shape}")
        print(f"  hidden_pad={hidden_pad}, intermediate_pad={intermediate_pad}")

        # Print what aiter will choose
        try:
            _, model_dim, inter_dim = fm.get_inter_dim(
                gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
            padded_M = fm.get_padded_M(M)
            use_nt_val = fm.use_nt(padded_M, topk_ids.shape[1], E)
            block_m_val = fm.get_block_size_M(padded_M, topk_ids.shape[1], E, inter_dim)
            est_m = padded_M * topk_ids.shape[1] // E
            print(f"  [AITER] model_dim={model_dim}, inter_dim={inter_dim}, "
                  f"padded_M={padded_M}, est_m={est_m}")
            print(f"  [AITER] use_nt={use_nt_val}, block_m={block_m_val}")
        except Exception as e:
            print(f"  [AITER INFO ERR] {e}")

        if already_profiled_externally:
            print("  [INFO] External profiler detected (eval.py --mode profile), "
                  "skipping internal profiler")
        else:
            # Try torch.profiler first, fallback to CUDA events
            ok = _run_torch_profiler(data, hidden_pad, intermediate_pad)
            if not ok:
                print("[FALLBACK] Using CUDA event timing")
                _run_cuda_event_timing(data, hidden_pad, intermediate_pad)

    # Actual computation for eval.py correctness/benchmark
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

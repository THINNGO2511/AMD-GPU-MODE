#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Test: Swiglu vs Silu activation type.
The Swiglu code path in fused_moe_2stages SKIPS first quant (a1_scale=None).
This could save ~14% (one of two fused_dynamic_mxfp4_quant_moe_sort calls).

Also probes: dispatch table, q_dtype_a mapping, timing difference.
"""
import torch
import functools
import time
import os

from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call = 0

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # use_nt=False for ALL shapes
    fm.use_nt = lambda token, topk, expert: False

    # block_m tuning for E<=64
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
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result

    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    # Probe: check if Swiglu is in the dispatch table
    try:
        # Look for the _apis dict or similar
        for attr in dir(fm):
            obj = getattr(fm, attr)
            if isinstance(obj, dict) and len(obj) > 5:
                for k, v in obj.items():
                    if isinstance(k, tuple) and len(k) >= 2:
                        print(f"[PROBE] DISPATCH: {k} -> {v}", flush=True)
                        break
    except Exception as e:
        print(f"[PROBE] dispatch probe: {e}", flush=True)

    # Check what activation types exist
    try:
        for attr in ['Silu', 'Swiglu', 'Gelu', 'SwiGLU', 'SWIGLU']:
            val = getattr(ActivationType, attr, None)
            if val is not None:
                print(f"[PROBE] ActivationType.{attr} = {val}", flush=True)
    except Exception as e:
        print(f"[PROBE] activation types: {e}", flush=True)

    # Check fused_moe signature for Swiglu support
    try:
        import inspect
        sig = inspect.signature(fused_moe)
        print(f"[PROBE] fused_moe sig: {sig}", flush=True)
    except Exception as e:
        print(f"[PROBE] sig: {e}", flush=True)

    # Try calling fused_moe with Swiglu to see if it's accepted
    # (We'll do this in the first real call to compare timing)


def custom_kernel(data: input_t) -> output_t:
    global _call
    _call += 1
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

    # On first call, try both Silu and Swiglu to compare
    if _call == 1:
        try:
            # Time Silu path
            torch.cuda.synchronize()
            t0 = time.time()
            out_silu = fused_moe(
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
            torch.cuda.synchronize()
            t1 = time.time()
            silu_time = (t1 - t0) * 1e6

            # Try Swiglu path
            try:
                torch.cuda.synchronize()
                t2 = time.time()
                out_swiglu = fused_moe(
                    hidden_states,
                    gate_up_weight_shuffled, down_weight_shuffled,
                    topk_weights, topk_ids,
                    expert_mask=None, activation=ActivationType.Swiglu,
                    quant_type=QuantType.per_1x32, doweight_stage1=False,
                    w1_scale=gate_up_weight_scale_shuffled,
                    w2_scale=down_weight_scale_shuffled,
                    a1_scale=None, a2_scale=None,
                    hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
                )
                torch.cuda.synchronize()
                t3 = time.time()
                swiglu_time = (t3 - t2) * 1e6

                # Compare accuracy
                diff = (out_silu - out_swiglu).abs()
                maxdiff = diff.max().item()
                reldiff = (diff / (out_silu.abs() + 1e-8)).max().item()
                print(f"[PROBE] Silu: {silu_time:.0f}μs, Swiglu: {swiglu_time:.0f}μs", flush=True)
                print(f"[PROBE] maxdiff: {maxdiff:.6f}, reldiff: {reldiff:.6f}", flush=True)
                print(f"[PROBE] speedup: {silu_time/swiglu_time:.2f}x", flush=True)
            except Exception as e:
                print(f"[PROBE] Swiglu FAILED: {str(e)[:300]}", flush=True)

            return out_silu

        except Exception as e:
            print(f"[PROBE] timing failed: {str(e)[:200]}", flush=True)

    # Use proven Silu path for benchmark
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

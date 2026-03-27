#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Try Triton MoE kernel for stage 1 (fused_moe_mxfp4_silu).
These use tl.dot_scaled which may be faster than CK kernels.
Also probe the kernel source to understand config options.
Stage 2 still uses CK.
"""
import torch
import functools
import sys
import os
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, moe_sorting, get_padded_M, get_inter_dim
from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
import aiter.fused_moe as fm

_patched = False
_probed = False

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


def _probe_triton_kernels():
    """Probe Triton MoE kernel source code to understand config options."""
    global _probed
    if _probed:
        return
    _probed = True

    try:
        from aiter.ops.triton.moe.moe_op_mxfp4_silu_fused import fused_moe_mxfp4_silu, _fused_moe_kernel_mxfp4_silu

        # Get full source of the wrapper function
        src = inspect.getsource(fused_moe_mxfp4_silu)
        print(f"[PROBE] fused_moe_mxfp4_silu full source ({len(src)} chars):")
        for line in src.split('\n'):
            print(f"  {line}")

        # Get kernel source
        try:
            ksrc = inspect.getsource(_fused_moe_kernel_mxfp4_silu)
            print(f"\n[PROBE] _fused_moe_kernel_mxfp4_silu source ({len(ksrc)} chars):")
            # Print first 100 lines
            for i, line in enumerate(ksrc.split('\n')[:100]):
                print(f"  {line}")
            if len(ksrc.split('\n')) > 100:
                print(f"  ... ({len(ksrc.split(chr(10)))} total lines)")
        except Exception as e:
            print(f"[PROBE] kernel source error: {e}")

    except Exception as e:
        import traceback
        print(f"[PROBE] error: {e}")
        traceback.print_exc()

    # Also probe stage 2 Triton kernel
    try:
        from aiter.ops.triton.moe.moe_op_mxfp4 import fused_moe_mxfp4, _fused_moe_kernel_mxfp4
        src2 = inspect.getsource(fused_moe_mxfp4)
        print(f"\n[PROBE] fused_moe_mxfp4 full source ({len(src2)} chars):")
        for line in src2.split('\n'):
            print(f"  {line}")
    except Exception as e:
        print(f"[PROBE] stage2 triton error: {e}")

    # Probe _moe_sorting_impl source
    try:
        src3 = inspect.getsource(fm._moe_sorting_impl)
        print(f"\n[PROBE] _moe_sorting_impl source ({len(src3)} chars):")
        for line in src3.split('\n')[:60]:
            print(f"  {line}")
    except Exception as e:
        print(f"[PROBE] sorting impl error: {e}")

    sys.stdout.flush()


def custom_kernel(data: input_t) -> output_t:
    _patch()
    _probe_triton_kernels()

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
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

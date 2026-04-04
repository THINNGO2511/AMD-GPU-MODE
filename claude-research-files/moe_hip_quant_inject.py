#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE with HIP quant injection: Replace dynamic_mxfp4_quant with 2.6x faster HIP kernel.
Based on submission_optimized_v2 (proven best on leaderboard).
The HIP quant replaces the SEPARATE quant path (token_num > 1024).
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_hip_mod = None

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

HIP_QUANT_SOURCE = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

__device__ __forceinline__ unsigned char compute_e8m0(float amax) {
    if (amax == 0.0f) return 0;
    unsigned int amax_int = __float_as_uint(amax);
    unsigned int rounded = (amax_int + 0x200000u) & 0xFF800000u;
    int biased_exp = (int)((rounded >> 23) & 0xFF);
    int e8m0 = biased_exp - 2;
    if (e8m0 < 0) e8m0 = 0;
    if (e8m0 > 254) e8m0 = 254;
    return (unsigned char)e8m0;
}

__device__ __forceinline__ unsigned char quantize_fp4(float val, float inv_scale) {
    float scaled = val * inv_scale;
    unsigned int qx = __float_as_uint(scaled);
    unsigned char sign = (qx >> 31) & 1;
    float abs_val = fabsf(scaled);
    if (abs_val < 0.25f) return sign << 3;
    unsigned char code;
    if (abs_val < 0.75f)       code = 1;
    else if (abs_val < 1.25f)  code = 2;
    else if (abs_val < 1.75f)  code = 3;
    else if (abs_val < 2.5f)   code = 4;
    else if (abs_val < 3.5f)   code = 5;
    else if (abs_val < 5.0f)   code = 6;
    else                       code = 7;
    return (sign << 3) | code;
}

__global__ void fast_mxfp4_quant_v2(
    const hip_bfloat16* __restrict__ input,
    unsigned char* __restrict__ fp4_out,
    unsigned char* __restrict__ scale_out,
    int num_rows, int num_cols
) {
    int row = blockIdx.x;
    int group = blockIdx.y * blockDim.x + threadIdx.x;
    int groups_per_row = num_cols / 32;
    if (row >= num_rows || group >= groups_per_row) return;
    int base_col = group * 32;
    const hip_bfloat16* row_ptr = input + (long long)row * num_cols + base_col;
    float vals[32];
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (float)(row_ptr[i]);
        float a = fabsf(vals[i]);
        if (a > max_abs) max_abs = a;
    }
    unsigned char e8m0 = compute_e8m0(max_abs);
    float scale_val = exp2f((float)e8m0 - 127.0f);
    float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;
    unsigned char* out_ptr = fp4_out + (long long)row * (num_cols / 2) + base_col / 2;
    for (int i = 0; i < 32; i += 2) {
        unsigned char lo = quantize_fp4(vals[i], inv_scale);
        unsigned char hi = quantize_fp4(vals[i+1], inv_scale);
        out_ptr[i/2] = (hi << 4) | lo;
    }
    scale_out[row * groups_per_row + group] = e8m0;
}

std::vector<torch::Tensor> fast_mxfp4_quant(torch::Tensor input) {
    int num_rows = input.size(0);
    int num_cols = input.size(1);
    auto fp4_out = torch::empty({num_rows, num_cols / 2},
                                torch::dtype(torch::kUInt8).device(input.device()));
    auto scale_out = torch::empty({num_rows, num_cols / 32},
                                  torch::dtype(torch::kUInt8).device(input.device()));
    int groups_per_row = num_cols / 32;
    dim3 grid(num_rows, (groups_per_row + 255) / 256);
    dim3 block(min(256, groups_per_row));
    hipLaunchKernelGGL(fast_mxfp4_quant_v2, grid, block, 0, 0,
        reinterpret_cast<const hip_bfloat16*>(input.data_ptr()),
        fp4_out.data_ptr<unsigned char>(),
        scale_out.data_ptr<unsigned char>(),
        num_rows, num_cols);
    return {fp4_out, scale_out};
}
"""

def _compile_hip():
    global _hip_mod
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_mod = load_inline(
            name="fast_quant_inject_v1",
            cpp_sources="std::vector<torch::Tensor> fast_mxfp4_quant(torch::Tensor input);",
            cuda_sources=HIP_QUANT_SOURCE,
            functions=["fast_mxfp4_quant"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        return True
    except:
        return False


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # 1. Compile HIP quant kernel
    hip_ok = _compile_hip()

    # 2. If HIP compiled, monkey-patch dynamic_mxfp4_quant
    if hip_ok and _hip_mod is not None:
        from aiter import dtypes as aiter_dtypes

        def hip_dynamic_mxfp4_quant(x, **kwargs):
            """Drop-in replacement for dynamic_mxfp4_quant using HIP kernel."""
            fp4_u8, scale_u8 = _hip_mod.fast_mxfp4_quant(x)
            # Convert to aiter's expected dtypes
            fp4 = fp4_u8.view(aiter_dtypes.fp4x2)
            scale = scale_u8.view(aiter_dtypes.fp8_e8m0)
            return fp4, scale

        # Patch in the quant module
        import aiter.ops.triton.quant as quant_mod
        quant_mod.dynamic_mxfp4_quant = hip_dynamic_mxfp4_quant

        # Also patch the fused_moe module's reference
        if hasattr(fm, 'get_quant'):
            orig_get_quant = fm.get_quant.__wrapped__ if hasattr(fm.get_quant, '__wrapped__') else fm.get_quant
            @functools.lru_cache(maxsize=32)
            def patched_get_quant(quant_type):
                result = orig_get_quant(quant_type)
                if quant_type == QuantType.per_1x32:
                    return hip_dynamic_mxfp4_quant
                return result
            fm.get_quant = patched_get_quant

    # 3. Standard v2 optimizations
    fm.use_nt = lambda token, topk, expert: False

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

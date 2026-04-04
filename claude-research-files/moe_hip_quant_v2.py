#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE HIP quant v2: Match aiter's exact E8M0 + FP4 quantization.
v1 was 73-87% accuracy. Need >98%.
Key: use bit manipulation matching _mxfp4_quant_op's method.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

_hip_mod = None
_tested = False

HIP_QUANT_SOURCE = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// Match aiter's _mxfp4_quant_op exactly:
// 1. Find amax of 32-element group
// 2. Scale = (amax_int + 0x200000) & 0xFF800000  (round up to next fp32 quantum)
// 3. E8M0 = exponent_of(scale) - 2  (= (scale_int >> 23) - 127 - 2 = (scale_int >> 23) - 129)
//    Wait: E8M0 represents 2^(e8m0 - 127). So if scale = 2^exp, then e8m0 = exp + 127.
//    But the formula is: e8m0 = floor(log2(rounded_amax)) - 2 + 127 = exponent - 2 + 127
//    Actually: the MXFP4 spec says scale = 2^(e8m0-127), max_val = 6 * scale
//    So we want: 6 * 2^(e8m0-127) >= amax
//    => e8m0 >= log2(amax/6) + 127
//    The aiter method rounds amax UP to nearest power of 2, then subtracts 2 from exponent.
//    rounded_amax = 2^ceil(log2(amax)), exponent = ceil(log2(amax))
//    e8m0 = exponent - 2 + 127 = ceil(log2(amax)) + 125
//    This means: 6 * 2^(e8m0-127) = 6 * 2^(ceil(log2(amax))-2) = 6/4 * 2^ceil(log2(amax)) >= 1.5*amax >= amax ✓

__device__ __forceinline__ unsigned char compute_e8m0(float amax) {
    if (amax == 0.0f) return 0;

    // Reinterpret amax as uint32
    unsigned int amax_int = __float_as_uint(amax);

    // Round up: add 0x200000 then mask mantissa bits
    // This rounds amax to the next power of 2 (or stays if already power of 2)
    unsigned int rounded = (amax_int + 0x200000u) & 0xFF800000u;

    // Extract exponent: bits 30:23, biased by 127
    int biased_exp = (int)((rounded >> 23) & 0xFF);

    // E8M0 = biased_exp - 2 (shift down by factor of 4)
    // because max FP4 value is 6.0, and 6 ≈ 2^2.58, so we need ~2 less exponent
    int e8m0 = biased_exp - 2;

    if (e8m0 < 0) e8m0 = 0;
    if (e8m0 > 254) e8m0 = 254;

    return (unsigned char)e8m0;
}

// FP4 E2M1 quantization using bit manipulation (RNE)
// Values: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} with sign
__device__ __forceinline__ unsigned char quantize_fp4(float val, float inv_scale) {
    float scaled = val * inv_scale;
    unsigned int qx = __float_as_uint(scaled);
    unsigned char sign = (qx >> 31) & 1;
    qx &= 0x7FFFFFFFu;  // abs

    // FP4 E2M1: bias=1, so exponent range is 0-3 (values 0.5 to 6.0)
    // Encoding: sign(1) | exp(2) | mant(1)
    // The bit manipulation: extract top 3 bits of significance

    // Clamp to max representable (6.0 in E2M1 = 0x40C00000)
    if (qx > 0x40C00000u) qx = 0x40C00000u;

    // If below minimum subnormal (0.5 = 0x3F000000), round to 0
    if (qx < 0x3E800000u) return sign << 3;  // underflow to 0

    // RNE rounding: add round bit + tie-break
    // For E2M1 with bias=1: we need to extract exponent and mantissa differently
    // Simplified: use lookup based on magnitude ranges
    float abs_val = __uint_as_float(qx);
    unsigned char code;
    if (abs_val < 0.75f)       code = 1;  // 0.5
    else if (abs_val < 1.25f)  code = 2;  // 1.0
    else if (abs_val < 1.75f)  code = 3;  // 1.5
    else if (abs_val < 2.5f)   code = 4;  // 2.0
    else if (abs_val < 3.5f)   code = 5;  // 3.0
    else if (abs_val < 5.0f)   code = 6;  // 4.0
    else                       code = 7;  // 6.0

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

    // Load and find amax
    float vals[32];
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (float)(row_ptr[i]);
        float a = fabsf(vals[i]);
        if (a > max_abs) max_abs = a;
    }

    // E8M0 scale using aiter's exact method
    unsigned char e8m0 = compute_e8m0(max_abs);

    // Compute scale and inverse
    float scale_val = exp2f((float)e8m0 - 127.0f);
    float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;

    // Quantize and pack
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

def _compile():
    global _hip_mod
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_mod = load_inline(
            name="fast_quant_v2",
            cpp_sources="std::vector<torch::Tensor> fast_mxfp4_quant(torch::Tensor input);",
            cuda_sources=HIP_QUANT_SOURCE,
            functions=["fast_mxfp4_quant"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("HIP QUANT v2 COMPILE SUCCESS!", flush=True)
        return True
    except Exception as e:
        print(f"HIP QUANT v2 COMPILE FAILED: {e}", flush=True)
        return False

def _test():
    if _hip_mod is None: return
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    for M, K in [(4, 128), (16, 512), (64, 2048), (512, 7168)]:
        torch.manual_seed(42)
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        ref_fp4, ref_scale = dynamic_mxfp4_quant(A)
        our_fp4, our_scale = _hip_mod.fast_mxfp4_quant(A)
        ref_u8 = ref_fp4.view(torch.uint8)
        ref_su8 = ref_scale.view(torch.uint8)
        fp4_match = (our_fp4 == ref_u8).float().mean().item()
        scale_match = (our_scale == ref_su8).float().mean().item()
        print(f"({M},{K}): fp4={fp4_match:.4f} scale={scale_match:.4f}", flush=True)
        if scale_match < 0.99 and M <= 16:
            for g in range(min(4, K//32)):
                print(f"  scale[0,{g}]: ours={our_scale[0,g].item()} ref={ref_su8[0,g].item()}", flush=True)

    # Timing
    print("\n=== Timing ===", flush=True)
    A_big = torch.randn(512, 7168, dtype=torch.bfloat16, device='cuda')
    for _ in range(3):
        _hip_mod.fast_mxfp4_quant(A_big)
        dynamic_mxfp4_quant(A_big)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N = 100
    start.record()
    for _ in range(N): _hip_mod.fast_mxfp4_quant(A_big)
    end.record(); torch.cuda.synchronize()
    hip_us = start.elapsed_time(end) * 1000 / N
    start.record()
    for _ in range(N): dynamic_mxfp4_quant(A_big)
    end.record(); torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) * 1000 / N
    print(f"HIP: {hip_us:.1f}μs, Triton: {triton_us:.1f}μs, Speedup: {triton_us/hip_us:.2f}x", flush=True)

import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

def custom_kernel(data: input_t) -> output_t:
    global _tested
    if not _tested:
        _tested = True
        ok = _compile()
        if ok: _test()
    (hidden_states, guw, dw, guws, dws, guwsh, dwsh, guwssh, dwssh,
     topk_weights, topk_ids, config) = data
    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, guwsh, dwsh, topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guwssh, w2_scale=dwssh, a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip)

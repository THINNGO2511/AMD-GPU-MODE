#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE with custom HIP bf16→FP4 quantization kernel.
Step 1: Compile HIP quant kernel + validate accuracy vs aiter reference.
Step 2: If accurate, monkey-patch into MoE pipeline.
Falls back to standard fused_moe for actual scoring.
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

// bf16 -> MXFP4 quantization kernel
// Each thread handles one group of 32 elements
// Produces: fp4x2 packed output + E8M0 scale per group

__device__ __forceinline__ unsigned char bf16_to_fp4(float val, float inv_scale) {
    float scaled = val * inv_scale;
    float abs_scaled = fabsf(scaled);
    unsigned char sign = (scaled < 0.0f) ? 0x8 : 0x0;

    // E2M1 FP4: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
    // Round to nearest with ties to even
    unsigned char code;
    if (abs_scaled < 0.25f)       code = 0x0;
    else if (abs_scaled < 0.75f)  code = 0x1;
    else if (abs_scaled < 1.25f)  code = 0x2;
    else if (abs_scaled < 1.75f)  code = 0x3;
    else if (abs_scaled < 2.5f)   code = 0x4;
    else if (abs_scaled < 3.5f)   code = 0x5;
    else if (abs_scaled < 5.0f)   code = 0x6;
    else                          code = 0x7;

    return sign | code;
}

__global__ void fast_mxfp4_quant_kernel(
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

    // Load 32 bf16 values and find max abs
    float vals[32];
    float max_abs = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (float)(row_ptr[i]);
        float abs_val = fabsf(vals[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }

    // E8M0 scale: we want 6.0 * 2^(e8m0 - 127) >= max_abs
    // Use aiter's method: round amax up, then floor(log2) - 2
    unsigned char e8m0;
    if (max_abs == 0.0f) {
        e8m0 = 0;
    } else {
        // Match aiter: (amax_int + 0x200000) & 0xFF800000 then floor(log2) - 2
        // Simpler equivalent: ceil(log2(max_abs / 6.0)) + 127
        int exp_val = (int)ceilf(log2f(max_abs / 6.0f)) + 127;
        e8m0 = (unsigned char)max(0, min(254, exp_val));
    }

    float scale_val = exp2f((float)e8m0 - 127.0f);
    float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;

    // Quantize and pack fp4x2
    unsigned char* out_ptr = fp4_out + (long long)row * (num_cols / 2) + base_col / 2;
    for (int i = 0; i < 32; i += 2) {
        unsigned char lo = bf16_to_fp4(vals[i], inv_scale);
        unsigned char hi = bf16_to_fp4(vals[i+1], inv_scale);
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

    hipLaunchKernelGGL(fast_mxfp4_quant_kernel, grid, block, 0, 0,
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
            name="fast_quant_v1",
            cpp_sources="std::vector<torch::Tensor> fast_mxfp4_quant(torch::Tensor input);",
            cuda_sources=HIP_QUANT_SOURCE,
            functions=["fast_mxfp4_quant"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("HIP QUANT COMPILE SUCCESS!", flush=True)
        return True
    except Exception as e:
        print(f"HIP QUANT COMPILE FAILED: {e}", flush=True)
        return False

def _test_accuracy():
    if _hip_mod is None: return

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    for shape in [(4, 128), (16, 512), (64, 2048), (512, 7168)]:
        M, K = shape
        torch.manual_seed(42)
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')

        ref_fp4, ref_scale = dynamic_mxfp4_quant(A)
        our_fp4, our_scale = _hip_mod.fast_mxfp4_quant(A)

        ref_fp4_u8 = ref_fp4.view(torch.uint8)
        ref_scale_u8 = ref_scale.view(torch.uint8)

        fp4_match = (our_fp4 == ref_fp4_u8).float().mean().item()
        scale_match = (our_scale == ref_scale_u8).float().mean().item()

        print(f"({M},{K}): fp4_match={fp4_match:.4f} scale_match={scale_match:.4f}", flush=True)

        if fp4_match < 0.99 and M <= 16:
            # Debug first mismatch
            diff_mask = our_fp4 != ref_fp4_u8
            if diff_mask.any():
                idx = diff_mask.nonzero()[0]
                r, c = idx[0].item(), idx[1].item()
                print(f"  First mismatch at [{r},{c}]: ours={our_fp4[r,c].item()} ref={ref_fp4_u8[r,c].item()}", flush=True)
                # Show the source values
                group = c * 2 // 32
                print(f"  Our scale[{r},{group}]={our_scale[r,group].item()} ref_scale={ref_scale_u8[r,group].item()}", flush=True)

    # Timing comparison
    print("\n=== Timing ===", flush=True)
    A_big = torch.randn(512, 7168, dtype=torch.bfloat16, device='cuda')
    # Warmup
    for _ in range(3):
        _hip_mod.fast_mxfp4_quant(A_big)
        dynamic_mxfp4_quant(A_big)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N = 100

    start.record()
    for _ in range(N):
        _hip_mod.fast_mxfp4_quant(A_big)
    end.record()
    torch.cuda.synchronize()
    hip_us = start.elapsed_time(end) * 1000 / N

    start.record()
    for _ in range(N):
        dynamic_mxfp4_quant(A_big)
    end.record()
    torch.cuda.synchronize()
    triton_us = start.elapsed_time(end) * 1000 / N

    print(f"HIP quant: {hip_us:.1f} μs", flush=True)
    print(f"Triton quant: {triton_us:.1f} μs", flush=True)
    print(f"Speedup: {triton_us/hip_us:.2f}x", flush=True)

# Standard MoE fallback
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

def custom_kernel(data: input_t) -> output_t:
    global _tested
    if not _tested:
        _tested = True
        ok = _compile()
        if ok:
            _test_accuracy()

    (hidden_states, guw, dw, guws, dws, guwsh, dwsh, guwssh, dwssh,
     topk_weights, topk_ids, config) = data
    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, guwsh, dwsh, topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guwssh, w2_scale=dwssh, a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip)

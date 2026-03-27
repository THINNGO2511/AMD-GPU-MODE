#!/usr/bin/env python3
"""
Minimal MLA MXFP4 HIP Attention Kernel
- 1 block per (batch, head) — NO split-K
- 64 threads per block (1 wavefront)
- Sequential loop over all KV tokens
- Online softmax for numerical stability
- Prioritizes correctness and compilation speed over performance
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// E2M1 lookup table for FP4 dequantization (16 entries)
// Index 0-7: positive values, 8-15: negative values
__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__global__ void mla_mxfp4_kernel(
    const float* __restrict__ Q,          // [total_q, 16, 576] but we index as [q_idx * 16 * 576 + head * 576 + d]
    const unsigned char* __restrict__ KV,  // [total_kv, 288] packed fp4x2
    const unsigned char* __restrict__ scales, // [total_kv, 24] e8m0 scales (first 18 used)
    const int* __restrict__ qo_indptr,     // [batch+1]
    const int* __restrict__ kv_indptr,     // [batch+1]
    float* __restrict__ Out,               // [total_q, 16, 512] output as float, converted to bf16 later
    int n_heads,
    float sm_scale
) {
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;  // 0..63

    // Determine ranges
    int q_start = qo_indptr[batch_idx];
    int kv_start = kv_indptr[batch_idx];
    int kv_end = kv_indptr[batch_idx + 1];
    int n_kv = kv_end - kv_start;

    if (n_kv == 0) return;

    // Shared memory for Q vector (576 floats) and KV dequantized (576 floats)
    __shared__ float s_Q[576];
    __shared__ float s_KV[576];
    __shared__ float s_reduce[64];

    // Load Q[576] into shared memory - 64 threads, 9 elements each
    for (int d = tid; d < 576; d += 64) {
        s_Q[d] = Q[(q_start * n_heads + head_idx) * 576 + d];
    }
    __syncthreads();

    // Online softmax state
    float max_score = -1e30f;
    float sum_exp = 0.0f;

    // V accumulator: each thread owns a subset of the 512 output dims
    // Thread tid owns elements tid, tid+64, tid+128, ..., tid+448 (8 elements)
    float v_acc[8];
    for (int i = 0; i < 8; i++) v_acc[i] = 0.0f;

    // Process each KV token sequentially
    for (int kv_idx = 0; kv_idx < n_kv; kv_idx++) {
        int kv_global = kv_start + kv_idx;

        // --- Step A: Dequantize KV into shared memory ---
        // Each thread dequantizes a portion: 576 elements, 64 threads -> 9 elements each
        // KV is packed: 288 bytes -> 576 fp4 values
        // Each byte: low nibble = even index, high nibble = odd index
        for (int d = tid; d < 288; d += 64) {
            unsigned char packed = KV[kv_global * 288 + d];
            int low_nib = packed & 0xF;
            int high_nib = (packed >> 4) & 0xF;

            // Which scale group? Each group covers 32 elements
            // Element 2*d is in group (2*d)/32, element 2*d+1 is in group (2*d+1)/32
            int elem0 = 2 * d;
            int elem1 = 2 * d + 1;
            int group0 = elem0 >> 5;  // /32
            int group1 = elem1 >> 5;

            // Only first 18 groups used (576/32=18)
            float scale0 = (group0 < 18) ? ldexpf(1.0f, (int)scales[kv_global * 24 + group0] - 127) : 0.0f;
            float scale1 = (group1 < 18) ? ldexpf(1.0f, (int)scales[kv_global * 24 + group1] - 127) : 0.0f;

            s_KV[elem0] = E2M1_LUT[low_nib] * scale0;
            s_KV[elem1] = E2M1_LUT[high_nib] * scale1;
        }
        __syncthreads();

        // --- Step B: Dot product Q . KV (576-dim) ---
        // Each thread computes partial sum over its assigned dimensions
        float partial = 0.0f;
        for (int d = tid; d < 576; d += 64) {
            partial += s_Q[d] * s_KV[d];
        }

        // Warp reduction within the single wavefront (64 threads)
        // Use shared memory reduction since we have 64 threads
        s_reduce[tid] = partial;
        __syncthreads();

        // Reduce 64 -> 1
        if (tid < 32) s_reduce[tid] += s_reduce[tid + 32];
        __syncthreads();
        if (tid < 16) s_reduce[tid] += s_reduce[tid + 16];
        __syncthreads();
        if (tid < 8) s_reduce[tid] += s_reduce[tid + 8];
        __syncthreads();
        if (tid < 4) s_reduce[tid] += s_reduce[tid + 4];
        __syncthreads();
        if (tid < 2) s_reduce[tid] += s_reduce[tid + 2];
        __syncthreads();
        if (tid < 1) s_reduce[0] += s_reduce[1];
        __syncthreads();

        float score = s_reduce[0] * sm_scale;

        // --- Step C: Online softmax update ---
        float old_max = max_score;
        float new_max = fmaxf(old_max, score);
        float exp_diff = expf(old_max - new_max);
        float exp_score = expf(score - new_max);

        // Rescale existing accumulators
        for (int i = 0; i < 8; i++) {
            v_acc[i] *= exp_diff;
        }
        sum_exp = sum_exp * exp_diff + exp_score;
        max_score = new_max;

        // --- Step D: Accumulate V contribution ---
        // V is the first 512 dims of the dequantized KV
        for (int i = 0; i < 8; i++) {
            int d = tid + i * 64;
            if (d < 512) {
                v_acc[i] += exp_score * s_KV[d];
            }
        }
        __syncthreads();
    }

    // --- Step E: Normalize and store ---
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    int out_base = (q_start * n_heads + head_idx) * 512;

    for (int i = 0; i < 8; i++) {
        int d = tid + i * 64;
        if (d < 512) {
            Out[out_base + d] = v_acc[i] * inv_sum;
        }
    }
}

torch::Tensor mla_fwd(
    torch::Tensor Q,
    torch::Tensor KV,
    torch::Tensor scales,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    int n_heads,
    float sm_scale
) {
    int batch_size = qo_indptr.size(0) - 1;
    int total_q = Q.size(0);

    // Output: [total_q, n_heads, 512] as float, will convert to bf16 after
    auto Out = torch::zeros({total_q, n_heads, 512}, Q.options().dtype(torch::kFloat32));

    dim3 grid(batch_size, n_heads);
    dim3 block(64);

    hipLaunchKernelGGL(
        mla_mxfp4_kernel,
        grid, block,
        0, 0,
        Q.data_ptr<float>(),
        KV.data_ptr<unsigned char>(),
        scales.data_ptr<unsigned char>(),
        qo_indptr.data_ptr<int>(),
        kv_indptr.data_ptr<int>(),
        Out.data_ptr<float>(),
        n_heads,
        sm_scale
    );

    // Convert to bf16
    return Out.to(torch::kBFloat16);
}
"""

CPP_SRC = """
torch::Tensor mla_fwd(
    torch::Tensor Q,
    torch::Tensor KV,
    torch::Tensor scales,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    int n_heads,
    float sm_scale
);
"""

mod = load_inline(
    name="mla_minimal_v2",
    cpp_sources=CPP_SRC,
    cuda_sources=HIP_SRC,
    functions=["mla_fwd"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
)


def custom_kernel(data):
    queries, kv_data, qo_indptr, kv_indptr, config = data
    fp4x2_buf, e8m0_scales = kv_data["mxfp4"]

    # View as uint8 and reshape
    fp4x2_buf = fp4x2_buf.view(torch.uint8).reshape(-1, 288)
    e8m0_scales = e8m0_scales.view(torch.uint8)

    batch_size = qo_indptr.shape[0] - 1
    sm_scale = config.get('sm_scale', 1.0 / (576 ** 0.5))

    # Q is [total_q, n_heads, 576] in bf16, convert to float for the kernel
    q_float = queries.float()

    return mod.mla_fwd(
        q_float,
        fp4x2_buf,
        e8m0_scales,
        qo_indptr.int(),
        kv_indptr.int(),
        16,
        sm_scale
    )

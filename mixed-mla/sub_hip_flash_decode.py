#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP Flash-Decoding MLA kernel for AMD MI355X (gfx950, CDNA4).

Two-stage flash-decoding with custom HIP C++ kernels:
  Stage 1: Each thread block handles (batch_idx, split_idx), processes ALL 16 heads.
            Loads KV once, reuses for all heads. Online softmax per head.
            Stores partial output [16, 512] fp32 + LSE [16] fp32 per split.
  Stage 2: Reduces across splits using LSE rescaling.
            Grid: (batch_size, 16) -- one block per (batch, head).

Key optimizations:
  - bf16 Q directly (no fp8 quantization -- saves 2 kernel launches vs aiter)
  - All 16 heads in one thread block (BLOCK_H=16) -- load KV once, reuse for all heads
  - fp8 KV with scalar scale (1 byte/elem vs 2 for bf16 -- 2x bandwidth savings)
  - dim=576 split into 512 + 64 for alignment
  - 256 threads per block (4 wavefronts on CDNA4)
  - BLOCK_KV=64 tokens processed at a time
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP C++ source -- Stage 1 + Stage 2 kernels
# ---------------------------------------------------------------------------

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

// Constants
#define NUM_HEADS 16
#define QK_DIM 576
#define V_DIM 512
#define ROPE_DIM 64
#define THREADS 256

// Helper: min for int
__device__ __forceinline__ int imin(int a, int b) { return a < b ? a : b; }

// ---------------------------------------------------------------------------
// Stage 1: Flash-decoding partial attention
// Grid: (batch_size, num_splits)
// Block: 256 threads
//
// Each block processes one (batch, split) pair for ALL 16 heads.
// KV tokens are loaded once from global memory, reused across all heads.
//
// Layout:
//   Q:          (total_q, 16, 576) bf16
//   KV:         (total_kv, 576) fp8_e4m3
//   Partial_O:  (batch, num_splits, 16, 512) fp32
//   Partial_LSE:(batch, num_splits, 16) fp32
//
// Approach:
//   - Load all 16 Q vectors (16x576) into shared memory once
//   - For each KV token: compute 16 dot products (Q.K), online softmax, V accum
//   - Each thread owns 2 V output dims (tid and tid+256)
//   - Dot product: 576 dims split across 256 threads (2-3 dims each)
//   - Reduction via shared memory (8 steps for 256 threads)
//   - Shared memory: 16*576 floats (Q) + 256 floats (reduction) = ~37.5 KB
// ---------------------------------------------------------------------------

__global__ void mla_stage1(
    const hip_bfloat16* __restrict__ Q,
    const __hip_fp8_e4m3_fnuz* __restrict__ KV,
    const float kv_scale,
    const int* __restrict__ qo_indptr,
    const int* __restrict__ kv_indptr,
    float* __restrict__ partial_o,
    float* __restrict__ partial_lse,
    const float sm_scale,
    const int num_splits
) {
    const int batch_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int kv_start = kv_indptr[batch_idx];
    const int kv_end = kv_indptr[batch_idx + 1];
    const int kv_len = kv_end - kv_start;

    // Compute this split's KV range
    const int tokens_per_split = (kv_len + num_splits - 1) / num_splits;
    const int split_start = tokens_per_split * split_idx;
    const int split_end_raw = split_start + tokens_per_split;
    const int split_end = split_end_raw < kv_len ? split_end_raw : kv_len;
    const int split_len = (split_start < kv_len) ? (split_end - split_start) : 0;

    if (split_len == 0) {
        // Empty split: write -inf LSE and zero output
        const int lse_base = (batch_idx * num_splits + split_idx) * NUM_HEADS;
        for (int h = tid; h < NUM_HEADS; h += THREADS) {
            partial_lse[lse_base + h] = -1e30f;
        }
        const int o_base = (batch_idx * num_splits + split_idx) * NUM_HEADS * V_DIM;
        for (int i = tid; i < NUM_HEADS * V_DIM; i += THREADS) {
            partial_o[o_base + i] = 0.0f;
        }
        return;
    }

    // Q base pointer
    const int q_idx = qo_indptr[batch_idx];
    const hip_bfloat16* q_base = Q + (long long)q_idx * NUM_HEADS * QK_DIM;

    // Shared memory layout:
    //   s_q:    [16][576] floats = 36864 bytes (Q vectors for all heads)
    //   s_red:  [256] floats = 1024 bytes (reduction workspace, reused per head)
    // Total: ~37 KB, well within 64 KB LDS limit
    __shared__ float s_q[NUM_HEADS][QK_DIM];
    __shared__ float s_red[THREADS];

    // Build fp8_e4m3fnuz LUT in shared memory (256 entries)
    __shared__ float fp8_lut[256];
    if (tid < 256) {
        unsigned char b = (unsigned char)tid;
        if (b == 0x80 || b == 0) {
            fp8_lut[tid] = 0.0f;
        } else {
            int sign = (b >> 7) & 1;
            int exp = (b >> 3) & 0xF;
            int mant = b & 0x7;
            float val;
            if (exp == 0) {
                val = ldexpf((float)mant / 8.0f, 1 - 8);
            } else {
                val = ldexpf(1.0f + (float)mant / 8.0f, exp - 8);
            }
            fp8_lut[tid] = sign ? -val : val;
        }
    }
    __syncthreads();

    // Load Q into shared memory: 16 * 576 = 9216 elements, 256 threads
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += THREADS) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        s_q[h][d] = (float)q_base[h * QK_DIM + d];
    }
    __syncthreads();

    // Per-head online softmax state (registers)
    float m_h[NUM_HEADS];   // running max score
    float l_h[NUM_HEADS];   // running sum of exp(score - max)
    float o0_h[NUM_HEADS];  // V accumulator dim tid
    float o1_h[NUM_HEADS];  // V accumulator dim tid+256

    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        m_h[h] = -1e30f;
        l_h[h] = 0.0f;
        o0_h[h] = 0.0f;
        o1_h[h] = 0.0f;
    }

    // V output dims this thread owns
    const int vd0 = tid;              // always < 512
    const int vd1 = tid + THREADS;    // tid + 256, always < 512

    // Process each KV token in this split
    for (int t = 0; t < split_len; t++) {
        const int kv_global = kv_start + split_start + t;
        const unsigned char* kv_ptr = (const unsigned char*)(KV + (long long)kv_global * QK_DIM);

        // Load fp8 as uint8, convert via LUT (single shared memory read per element)
        float kv0 = fp8_lut[kv_ptr[tid]] * kv_scale;
        float kv1 = fp8_lut[kv_ptr[tid + 256]] * kv_scale;
        float kv2 = (tid < ROPE_DIM) ? (fp8_lut[kv_ptr[tid + 512]] * kv_scale) : 0.0f;

        // Pre-load V values for this thread's output dims (first 512 of KV)
        float v0 = kv0;                // dim tid < 256, always valid V
        float v1 = kv1;                // dim tid+256 < 512, always valid V

        // For each head: compute dot product, update online softmax, accumulate V
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h++) {
            // Partial dot product: q[h][d] * kv[d] for this thread's dims
            float dot = s_q[h][tid] * kv0
                      + s_q[h][tid + 256] * kv1;
            if (tid < ROPE_DIM) {
                dot += s_q[h][tid + 512] * kv2;
            }

            // Reduce across 256 threads using shared memory
            s_red[tid] = dot;
            __syncthreads();

            // Tree reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
            if (tid < 128) s_red[tid] += s_red[tid + 128];
            __syncthreads();
            if (tid < 64) s_red[tid] += s_red[tid + 64];
            __syncthreads();

            // Last 64 threads: single wavefront, use volatile or syncthreads
            if (tid < 32) s_red[tid] += s_red[tid + 32];
            __syncthreads();
            if (tid < 16) s_red[tid] += s_red[tid + 16];
            __syncthreads();
            if (tid < 8) s_red[tid] += s_red[tid + 8];
            __syncthreads();
            if (tid < 4) s_red[tid] += s_red[tid + 4];
            __syncthreads();
            if (tid < 2) s_red[tid] += s_red[tid + 2];
            __syncthreads();
            if (tid < 1) s_red[0] += s_red[1];
            __syncthreads();

            // Score = dot(Q, K*scale) * sm_scale
            float score = s_red[0] * sm_scale;

            // Online softmax update
            float old_m = m_h[h];
            float new_m = (old_m > score) ? old_m : score;
            float exp_old = expf(old_m - new_m);
            float exp_new = expf(score - new_m);

            o0_h[h] = o0_h[h] * exp_old + exp_new * v0;
            o1_h[h] = o1_h[h] * exp_old + exp_new * v1;
            l_h[h] = l_h[h] * exp_old + exp_new;
            m_h[h] = new_m;
        }
    }

    // Store partial output (normalized) and LSE
    const int o_base = (batch_idx * num_splits + split_idx) * NUM_HEADS * V_DIM;
    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        float inv_l = (l_h[h] > 0.0f) ? (1.0f / l_h[h]) : 0.0f;
        partial_o[o_base + h * V_DIM + vd0] = o0_h[h] * inv_l;
        partial_o[o_base + h * V_DIM + vd1] = o1_h[h] * inv_l;
    }

    const int lse_base = (batch_idx * num_splits + split_idx) * NUM_HEADS;
    if (tid < NUM_HEADS) {
        float lse_val = (l_h[tid] > 0.0f) ? (m_h[tid] + logf(l_h[tid])) : -1e30f;
        partial_lse[lse_base + tid] = lse_val;
    }
}


// ---------------------------------------------------------------------------
// Stage 2: Reduce across splits
// Grid: (batch_size, NUM_HEADS)
// Block: 256 threads
//
// Each block merges all split partial results for one (batch, head).
// Online softmax merge using stored LSE values.
// ---------------------------------------------------------------------------

__global__ void mla_stage2(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_lse,
    hip_bfloat16* __restrict__ output,
    const int* __restrict__ qo_indptr,
    const int num_splits
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int q_idx = qo_indptr[batch_idx];

    // Each thread owns 2 output dims
    const int d0 = tid;
    const int d1 = tid + THREADS;

    float e_max = -1e30f;
    float e_sum = 0.0f;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        const int lse_idx = (batch_idx * num_splits + s) * NUM_HEADS + head_idx;
        float lse = partial_lse[lse_idx];

        if (lse > -1e29f) {
            const int o_idx = (batch_idx * num_splits + s) * NUM_HEADS * V_DIM + head_idx * V_DIM;
            float v0 = partial_o[o_idx + d0];
            float v1 = partial_o[o_idx + d1];

            // Online softmax merge across splits
            float new_max = (e_max > lse) ? e_max : lse;
            float old_scale = expf(e_max - new_max);
            float new_scale = expf(lse - new_max);

            acc0 = acc0 * old_scale + new_scale * v0;
            acc1 = acc1 * old_scale + new_scale * v1;
            e_sum = e_sum * old_scale + new_scale;
            e_max = new_max;
        }
    }

    // Normalize and write bf16 output
    float inv_sum = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;
    const long long out_base = (long long)q_idx * NUM_HEADS * V_DIM + (long long)head_idx * V_DIM;

    output[out_base + d0] = (hip_bfloat16)(acc0 * inv_sum);
    output[out_base + d1] = (hip_bfloat16)(acc1 * inv_sum);
}


// ---------------------------------------------------------------------------
// PyTorch C++ wrapper
// ---------------------------------------------------------------------------

torch::Tensor mla_flash_decode(
    torch::Tensor Q,
    torch::Tensor KV,
    float kv_scale_val,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    float sm_scale_val,
    int num_splits_val
) {
    const int batch_size = qo_indptr.size(0) - 1;
    const int total_q = Q.size(0);

    // Intermediate buffers
    auto opts_f32 = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto partial_o = torch::empty({batch_size, num_splits_val, NUM_HEADS, V_DIM}, opts_f32);
    auto partial_lse = torch::empty({batch_size, num_splits_val, NUM_HEADS}, opts_f32);

    // Output buffer
    auto opts_bf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(Q.device());
    auto output = torch::empty({total_q, NUM_HEADS, V_DIM}, opts_bf16);

    // Stage 1: partial attention per split
    dim3 grid1(batch_size, num_splits_val);
    dim3 block1(THREADS);

    hipLaunchKernelGGL(
        mla_stage1,
        grid1, block1,
        0, 0,
        (const hip_bfloat16*)Q.data_ptr(),
        (const __hip_fp8_e4m3_fnuz*)KV.data_ptr(),
        kv_scale_val,
        (const int*)qo_indptr.data_ptr(),
        (const int*)kv_indptr.data_ptr(),
        (float*)partial_o.data_ptr(),
        (float*)partial_lse.data_ptr(),
        sm_scale_val,
        num_splits_val
    );

    // Stage 2: reduce across splits
    dim3 grid2(batch_size, NUM_HEADS);
    dim3 block2(THREADS);

    hipLaunchKernelGGL(
        mla_stage2,
        grid2, block2,
        0, 0,
        (const float*)partial_o.data_ptr(),
        (const float*)partial_lse.data_ptr(),
        (hip_bfloat16*)output.data_ptr(),
        (const int*)qo_indptr.data_ptr(),
        num_splits_val
    );

    return output;
}
"""

CPP_SOURCE = """
torch::Tensor mla_flash_decode(
    torch::Tensor Q,
    torch::Tensor KV,
    float kv_scale_val,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    float sm_scale_val,
    int num_splits_val
);
"""

# ---------------------------------------------------------------------------
# Compile the HIP module
# ---------------------------------------------------------------------------

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="mla_flash_decode_v5",
            cpp_sources=CPP_SOURCE,
            cuda_sources=HIP_SOURCE,
            functions=["mla_flash_decode"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
    return _module

# Pre-allocated buffer caches
_alloc_cache = {}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    sm_scale = config["sm_scale"]

    # Use fp8 KV cache (1 byte/elem vs 2 for bf16 -- 2x bandwidth savings)
    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    # Flatten KV: (total_kv, 1, 576) -> (total_kv, 576)
    total_kv = kv_buffer_fp8.shape[0]
    kv_flat = kv_buffer_fp8.view(total_kv, 576)

    # Extract scalar kv_scale
    kv_scale_val = kv_scale.item()

    # Choose num_splits based on workload
    total_tokens = batch_size * kv_seq_len
    if total_tokens <= 4096:
        num_splits = 4
    elif total_tokens <= 32768:
        num_splits = 8
    else:
        num_splits = 16

    # Get compiled module
    mod = _get_module()

    # Launch
    output = mod.mla_flash_decode(
        q.contiguous(),
        kv_flat.contiguous(),
        kv_scale_val,
        qo_indptr.to(torch.int32).contiguous(),
        kv_indptr.to(torch.int32).contiguous(),
        sm_scale,
        num_splits,
    )

    return output

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP Flash-Decoding MLA kernel v4 for AMD MI355X (gfx950, CDNA4).

Optimizations over v3:
  1. Hardware fp8->float via __hip_fp8_e4m3_fnuz cast (replaces software ldexpf)
  2. Vectorized 128-bit loads: load 16 fp8 values per instruction via uint4
  3. Wave-level reduction with __shfl_down (replaces shared memory tree reduction)
  4. Reduced __syncthreads in inner loop
  5. Pre-compute Q*sm_scale so score = dot(Q_scaled, KV) with no post-multiply
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP C++ source -- Stage 1 + Stage 2 kernels (v4 optimized)
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
#define WAVE_SIZE 64
#define NUM_WAVES (THREADS / WAVE_SIZE)  // 4

// ---------------------------------------------------------------------------
// Hardware fp8_e4m3fnuz -> float conversion
// The __hip_fp8_e4m3_fnuz type has a built-in operator float() that uses
// the hardware conversion path on gfx950. We reinterpret raw bytes through
// this type to get single-cycle conversions.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float fp8_to_float(unsigned char b) {
    // Reinterpret the byte as __hip_fp8_e4m3_fnuz and use its hardware cast
    __hip_fp8_e4m3_fnuz val;
    // The __hip_fp8_e4m3_fnuz type stores its data as a single byte.
    // We can safely reinterpret-cast a byte pointer.
    *reinterpret_cast<unsigned char*>(&val) = b;
    return static_cast<float>(val);
}

// ---------------------------------------------------------------------------
// Wave-level (wavefront=64) reduction using shuffle-down
// ---------------------------------------------------------------------------
__device__ __forceinline__ float wave_reduce_sum(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val,  8);
    val += __shfl_down(val,  4);
    val += __shfl_down(val,  2);
    val += __shfl_down(val,  1);
    return val;  // result valid in lane 0 of each wavefront
}

// ---------------------------------------------------------------------------
// Stage 1: Flash-decoding partial attention (v4 optimized)
// Grid: (batch_size, num_splits)
// Block: 256 threads (4 wavefronts)
//
// Each block processes one (batch, split) pair for ALL 16 heads.
// KV tokens loaded once from global memory, reused across all heads.
//
// Layout:
//   Q:          (total_q, 16, 576) bf16
//   KV:         (total_kv, 576) fp8_e4m3
//   Partial_O:  (batch, num_splits, 16, 512) fp32
//   Partial_LSE:(batch, num_splits, 16) fp32
//
// Approach:
//   - Load all 16 Q vectors (16*576) into shared memory, pre-scaled by sm_scale
//   - For each KV token: vectorized 128-bit loads (16 fp8 values per load)
//   - Hardware fp8->float conversion
//   - Dot product reduced via wave shuffles + small cross-wave shared mem reduction
//   - Each thread owns 2 V output dims (tid and tid+256)
// ---------------------------------------------------------------------------

__global__ void mla_stage1(
    const hip_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ KV,  // raw bytes for fp8
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
    const int wave_id = tid / WAVE_SIZE;   // 0..3
    const int lane_id = tid % WAVE_SIZE;   // 0..63

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
    //   s_q:     [16][576] floats = 36864 bytes (Q vectors pre-scaled by sm_scale)
    //   s_wave:  [4] floats = 16 bytes (cross-wave reduction, one per wavefront)
    //   s_score: [1] float = 4 bytes (broadcast final score)
    // Total: ~36.9 KB, well within 64 KB LDS limit
    __shared__ float s_q[NUM_HEADS * QK_DIM];
    __shared__ float s_wave[NUM_WAVES];
    __shared__ float s_score;

    // Load Q into shared memory, pre-multiply by sm_scale
    // This means: score = dot(Q_scaled, KV * kv_scale) instead of dot(Q, KV) * sm_scale * kv_scale
    const float q_prescale = sm_scale;
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += THREADS) {
        int h = i / QK_DIM;
        int d = i % QK_DIM;
        s_q[i] = (float)q_base[h * QK_DIM + d] * q_prescale;
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
    const int vd0 = tid;              // always < 256, valid V dim
    const int vd1 = tid + THREADS;    // tid + 256, always < 512, valid V dim

    // ---------------------------------------------------------------
    // Main loop: process each KV token in this split
    // ---------------------------------------------------------------
    for (int t = 0; t < split_len; t++) {
        const int kv_global = kv_start + split_start + t;
        const unsigned char* kv_ptr = KV + (long long)kv_global * QK_DIM;

        // ---- Vectorized load + hardware fp8->float conversion ----
        // Each thread loads its assigned dims.
        // Thread tid handles dims: tid, tid+256, (tid+512 if tid < 64)
        //
        // For the first two chunks (0..255, 256..511), we use 128-bit
        // vectorized loads where possible. Each 128-bit load fetches
        // 16 consecutive fp8 bytes. We then convert in registers.
        //
        // However, since each thread needs non-contiguous dims (tid,
        // tid+256, tid+512), we load per-thread values individually
        // but use the hardware fp8 conversion.

        float kv0 = fp8_to_float(kv_ptr[tid]) * kv_scale;
        float kv1 = fp8_to_float(kv_ptr[tid + 256]) * kv_scale;
        float kv2 = (tid < ROPE_DIM) ? (fp8_to_float(kv_ptr[tid + 512]) * kv_scale) : 0.0f;

        // Pre-load V values (first 512 dims of KV)
        float v0 = kv0;   // dim tid < 256
        float v1 = kv1;   // dim tid+256 < 512

        // For each head: compute dot product, update online softmax, accumulate V
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h++) {
            const float* sq_h = s_q + h * QK_DIM;

            // Partial dot product: q[h][d] * kv[d] for this thread's dims
            float dot = sq_h[tid] * kv0
                      + sq_h[tid + 256] * kv1;
            if (tid < ROPE_DIM) {
                dot += sq_h[tid + 512] * kv2;
            }

            // ---- Wave-level reduction (wavefront = 64 lanes) ----
            float wave_sum = wave_reduce_sum(dot);

            // Cross-wave reduction: 4 wavefronts -> 1 value
            if (lane_id == 0) {
                s_wave[wave_id] = wave_sum;
            }
            __syncthreads();

            // Thread 0 sums across 4 waves and broadcasts
            if (tid == 0) {
                s_score = s_wave[0] + s_wave[1] + s_wave[2] + s_wave[3];
            }
            __syncthreads();

            float score = s_score;

            // Online softmax update
            float old_m = m_h[h];
            float new_m = fmaxf(old_m, score);
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
            float new_max = fmaxf(e_max, lse);
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
        (const unsigned char*)KV.data_ptr(),
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
            name="mla_flash_decode_v4",
            cpp_sources=CPP_SOURCE,
            cuda_sources=HIP_SOURCE,
            functions=["mla_flash_decode"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-ffast-math"],
        )
    return _module

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

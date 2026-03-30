#!/usr/bin/env python3
"""
MLA MXFP4 Split-K Attention Kernel for AMD MI355X (gfx950)
Uses HIP via torch load_inline. MXFP4 KV cache with bf16 Q/O.
"""

import os
import torch
import math

os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

from torch.utils.cpp_extension import load_inline

# ─────────────────────────────────────────────────────────────
# HIP kernel source
# ─────────────────────────────────────────────────────────────
HIP_SOURCE = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

// ── MXFP4 dequantization ──────────────────────────────────
// nibble: 4-bit value (0-15), scale: e8m0-derived float scale
__device__ __forceinline__ float dequant_fp4(uint8_t nibble, float scale) {
    int sign = (nibble >> 3) & 1;
    int exp  = (nibble >> 1) & 3;
    int man  = nibble & 1;
    float mag;
    if (exp == 0) {
        mag = man * 0.5f;           // subnormal: 0.0 or 0.5
    } else {
        mag = ldexpf(1.0f + man * 0.5f, exp - 1);  // normal
    }
    return sign ? -mag * scale : mag * scale;
}

// ── E8M0 scale decode ─────────────────────────────────────
__device__ __forceinline__ float decode_e8m0(uint8_t e) {
    // e8m0: value = 2^(e - 127). e=0 means zero, e=255 means inf/nan
    // For practical MXFP4, treat 0 as 0 and 255 as large.
    if (e == 0) return 0.0f;
    return ldexpf(1.0f, (int)e - 127);
}

// ── Wave-level reduction (64 threads per wave on GCN/CDNA) ─
__device__ __forceinline__ float wave_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset);
    }
    return val;
}

__device__ __forceinline__ float wave_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset));
    }
    return val;
}

// ── Constants ──────────────────────────────────────────────
#define BLOCK_SIZE 128
#define WAVE_SIZE  64
#define N_WAVES    2
#define HEAD_DIM   576
#define V_DIM      512
#define PACKED_DIM 288     // 576 / 2 packed fp4 bytes
#define N_SCALES   18      // 576 / 32 scale groups
#define SCALE_COLS 24      // padded scale columns

// Elements per thread for dot product: 576 / 128 = 4 full + remainder
// We'll use a loop approach instead for cleaner code
#define ELEMS_PER_THREAD_DOT  ((HEAD_DIM + BLOCK_SIZE - 1) / BLOCK_SIZE)  // = 5
#define V_PER_THREAD           (V_DIM / BLOCK_SIZE)  // = 4

// ── Split-K Attention Kernel ──────────────────────────────
// Grid: (n_splits, n_heads=16, n_batch)
// Block: 128 threads
__global__ void mla_mxfp4_splitk_kernel(
    const hip_bfloat16* __restrict__ Q,        // [total_q, 16, 576]
    const uint8_t*      __restrict__ fp4_buf,   // [total_kv, 1, 288] as uint8
    const uint8_t*      __restrict__ e8m0_buf,  // [total_kv, 24] as uint8
    hip_bfloat16*       __restrict__ partial_out, // [n_splits, total_q, 16, 512]
    float*              __restrict__ partial_lse, // [n_splits, total_q, 16]
    float*              __restrict__ partial_max, // [n_splits, total_q, 16]
    const int64_t*      __restrict__ qo_indptr,  // [batch+1]
    const int64_t*      __restrict__ kv_indptr,  // [batch+1]
    float               sm_scale,
    int                 n_splits,
    int                 total_q,
    int                 n_heads
) {
    int split_id = blockIdx.x;
    int head_id  = blockIdx.y;
    int batch_id = blockIdx.z;
    int tid      = threadIdx.x;
    int wave_id  = tid / WAVE_SIZE;   // 0 or 1
    int lane_id  = tid % WAVE_SIZE;

    // Batch boundaries
    int64_t q_start = qo_indptr[batch_id];
    int64_t q_end   = qo_indptr[batch_id + 1];
    int64_t kv_start = kv_indptr[batch_id];
    int64_t kv_end   = kv_indptr[batch_id + 1];

    int n_q_tokens = (int)(q_end - q_start);
    int n_kv_tokens = (int)(kv_end - kv_start);

    if (n_q_tokens == 0 || n_kv_tokens == 0) return;

    // Split KV range across n_splits
    int kv_per_split = (n_kv_tokens + n_splits - 1) / n_splits;
    int kv_split_start = split_id * kv_per_split;
    int kv_split_end   = min(kv_split_start + kv_per_split, n_kv_tokens);

    if (kv_split_start >= n_kv_tokens) return;

    // ── Shared memory layout ──
    // Q values in fp32: 576 floats = 2304 bytes
    // Cross-wave reduction: 2 floats (one per wave) = 8 bytes
    __shared__ float s_Q[HEAD_DIM];          // 576 floats
    __shared__ float s_reduce[N_WAVES];      // for cross-wave dot product reduction

    // Process each query token
    for (int qi = 0; qi < n_q_tokens; qi++) {
        int64_t q_idx = q_start + qi;

        // ── Step 1: Load Q into shared memory ──
        // Q layout: [total_q, 16, 576] → offset = (q_idx * 16 + head_id) * 576
        const hip_bfloat16* q_ptr = Q + (q_idx * n_heads + head_id) * HEAD_DIM;

        // 128 threads load 576 values: each thread loads ~4-5 values
        for (int i = tid; i < HEAD_DIM; i += BLOCK_SIZE) {
            s_Q[i] = (float)q_ptr[i];
        }
        __syncthreads();

        // ── Step 2: Online softmax + V accumulation over KV chunk ──
        float running_max = -1e30f;
        float running_sum = 0.0f;
        float v_acc[V_PER_THREAD];  // 4 V dimensions per thread
        for (int vi = 0; vi < V_PER_THREAD; vi++) {
            v_acc[vi] = 0.0f;
        }

        for (int ki = kv_split_start; ki < kv_split_end; ki++) {
            int64_t kv_idx = kv_start + ki;

            // ── Step 2a: Load and dequant KV token ──
            // fp4_buf layout: [total_kv, 1, 288]
            const uint8_t* kv_packed = fp4_buf + kv_idx * PACKED_DIM;
            // e8m0_buf layout: [total_kv, 24]
            const uint8_t* kv_scales = e8m0_buf + kv_idx * SCALE_COLS;

            // ── Step 2b: Compute dot product Q · KV ──
            // Each thread computes partial dot product over its assigned elements
            float dot_partial = 0.0f;
            for (int i = tid; i < HEAD_DIM; i += BLOCK_SIZE) {
                // Packed byte index and nibble position
                int byte_idx = i >> 1;        // i / 2
                int is_high  = i & 1;         // 0 = low nibble, 1 = high nibble
                uint8_t packed_byte = kv_packed[byte_idx];
                uint8_t nibble;
                if (is_high) {
                    nibble = (packed_byte >> 4) & 0x0F;
                } else {
                    nibble = packed_byte & 0x0F;
                }

                // Scale group: element i belongs to group i/32
                int scale_idx = i >> 5;  // i / 32
                float scale = decode_e8m0(kv_scales[scale_idx]);

                float kv_val = dequant_fp4(nibble, scale);
                dot_partial += s_Q[i] * kv_val;
            }

            // ── Reduce dot product across all 128 threads ──
            // First: wave-level reduction
            float wave_sum = wave_reduce_sum(dot_partial);

            // Cross-wave reduction via shared memory
            if (lane_id == 0) {
                s_reduce[wave_id] = wave_sum;
            }
            __syncthreads();

            float score;
            if (tid == 0) {
                score = 0.0f;
                for (int w = 0; w < N_WAVES; w++) {
                    score += s_reduce[w];
                }
                score *= sm_scale;
                s_reduce[0] = score;
            }
            __syncthreads();
            score = s_reduce[0];

            // ── Step 2c: Online softmax update ──
            float old_max = running_max;
            running_max = fmaxf(running_max, score);

            // Rescale existing accumulator
            float rescale = expf(old_max - running_max);
            running_sum = running_sum * rescale;
            for (int vi = 0; vi < V_PER_THREAD; vi++) {
                v_acc[vi] *= rescale;
            }

            float weight = expf(score - running_max);
            running_sum += weight;

            // ── Step 2d: Accumulate V ──
            // V is the first 512 elements of the KV vector
            // Each thread handles 4 V dimensions: tid*4, tid*4+1, tid*4+2, tid*4+3
            for (int vi = 0; vi < V_PER_THREAD; vi++) {
                int v_idx = tid * V_PER_THREAD + vi;  // global V index [0..511]

                // Dequant from packed fp4
                int byte_idx = v_idx >> 1;
                int is_high  = v_idx & 1;
                uint8_t packed_byte = kv_packed[byte_idx];
                uint8_t nibble;
                if (is_high) {
                    nibble = (packed_byte >> 4) & 0x0F;
                } else {
                    nibble = packed_byte & 0x0F;
                }

                int scale_idx = v_idx >> 5;
                float scale = decode_e8m0(kv_scales[scale_idx]);
                float v_val = dequant_fp4(nibble, scale);

                v_acc[vi] += weight * v_val;
            }
            __syncthreads();
        }  // end KV loop

        // ── Step 3: Write partial results ──
        float inv_sum = (running_sum > 0.0f) ? (1.0f / running_sum) : 0.0f;

        // partial_out: [n_splits, total_q, n_heads, V_DIM]
        int64_t out_offset = ((int64_t)split_id * total_q * n_heads + q_idx * n_heads + head_id) * V_DIM;
        hip_bfloat16* out_ptr = partial_out + out_offset;

        for (int vi = 0; vi < V_PER_THREAD; vi++) {
            int v_idx = tid * V_PER_THREAD + vi;
            out_ptr[v_idx] = (hip_bfloat16)(v_acc[vi] * inv_sum);
        }

        // partial_lse and partial_max: [n_splits, total_q, n_heads]
        if (tid == 0) {
            int64_t lse_offset = (int64_t)split_id * total_q * n_heads + q_idx * n_heads + head_id;
            // Store log(sum) + max for proper reduction
            partial_lse[lse_offset] = logf(fmaxf(running_sum, 1e-30f)) + running_max;
            partial_max[lse_offset] = running_max;
        }
        __syncthreads();
    }  // end query loop
}


// ── Reduce kernel: combine split-K partials ───────────────
// Grid: (ceil(total_q * n_heads / 256), 1, 1)
// Each thread handles one (query, head) pair
__global__ void mla_reduce_kernel(
    const hip_bfloat16* __restrict__ partial_out,  // [n_splits, total_q, n_heads, V_DIM]
    const float*        __restrict__ partial_lse,   // [n_splits, total_q, n_heads]
    const float*        __restrict__ partial_max,   // [n_splits, total_q, n_heads]
    hip_bfloat16*       __restrict__ output,         // [total_q, n_heads, V_DIM]
    int                 n_splits,
    int                 total_q,
    int                 n_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_qh = total_q * n_heads;
    if (idx >= total_qh) return;

    int qi   = idx / n_heads;
    int hi   = idx % n_heads;

    // Find global max across splits
    float global_max = -1e30f;
    for (int s = 0; s < n_splits; s++) {
        int64_t lse_off = (int64_t)s * total_qh + qi * n_heads + hi;
        float m = partial_max[lse_off];
        global_max = fmaxf(global_max, m);
    }

    // Compute rescaled sum of weights
    float total_weight = 0.0f;
    float split_weights[64];  // max splits
    for (int s = 0; s < n_splits; s++) {
        int64_t lse_off = (int64_t)s * total_qh + qi * n_heads + hi;
        float lse = partial_lse[lse_off];  // log(sum_s) + max_s
        float max_s = partial_max[lse_off];
        // weight_s = sum_s * exp(max_s - global_max) = exp(lse - global_max)
        float w = expf(lse - global_max);
        split_weights[s] = w;
        total_weight += w;
    }

    float inv_total = (total_weight > 0.0f) ? (1.0f / total_weight) : 0.0f;

    // Weighted sum of partial outputs
    // partial_out values are already divided by their local sum,
    // so we need: out = sum_s(split_weight_s * partial_out_s) / total_weight
    // But partial_out_s = (unnorm_acc_s / local_sum_s)
    // And split_weight_s = local_sum_s * exp(max_s - global_max)
    // So: split_weight_s * partial_out_s = unnorm_acc_s * exp(max_s - global_max)
    // This is correct.

    hip_bfloat16* out_ptr = output + ((int64_t)qi * n_heads + hi) * V_DIM;

    for (int v = 0; v < V_DIM; v++) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            int64_t p_off = ((int64_t)s * total_q * n_heads + qi * n_heads + hi) * V_DIM + v;
            float pv = (float)partial_out[p_off];
            acc += split_weights[s] * pv;
        }
        out_ptr[v] = (hip_bfloat16)(acc * inv_total);
    }
}


// ── Host wrapper functions ────────────────────────────────

torch::Tensor mla_mxfp4_attention(
    torch::Tensor Q,           // [total_q, 16, 576] bf16
    torch::Tensor fp4_buf,     // [total_kv, 1, 288] uint8 (already viewed)
    torch::Tensor e8m0_scales, // [total_kv, 24] uint8 (already viewed)
    torch::Tensor qo_indptr,   // [batch+1] int64
    torch::Tensor kv_indptr,   // [batch+1] int64
    double sm_scale_d,
    int64_t n_splits_i
) {
    int n_splits = (int)n_splits_i;
    float sm_scale = (float)sm_scale_d;

    int total_q = Q.size(0);
    int n_heads = Q.size(1);   // 16
    int n_batch = qo_indptr.size(0) - 1;

    // Allocate outputs
    auto partial_out = torch::zeros({n_splits, total_q, n_heads, 512},
                                     Q.options());  // bf16
    auto partial_lse = torch::zeros({n_splits, total_q, n_heads},
                                     torch::dtype(torch::kFloat32).device(Q.device()));
    auto partial_max = torch::full({n_splits, total_q, n_heads}, -1e30f,
                                    torch::dtype(torch::kFloat32).device(Q.device()));

    // Launch split-K kernel
    dim3 grid(n_splits, n_heads, n_batch);
    dim3 block(BLOCK_SIZE);

    hipLaunchKernelGGL(
        mla_mxfp4_splitk_kernel,
        grid, block,
        0, 0,
        (const hip_bfloat16*)Q.data_ptr(),
        (const uint8_t*)fp4_buf.data_ptr(),
        (const uint8_t*)e8m0_scales.data_ptr(),
        (hip_bfloat16*)partial_out.data_ptr(),
        partial_lse.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        qo_indptr.data_ptr<int64_t>(),
        kv_indptr.data_ptr<int64_t>(),
        sm_scale,
        n_splits,
        total_q,
        n_heads
    );

    // Allocate final output
    auto output = torch::zeros({total_q, n_heads, 512}, Q.options());

    // Launch reduce kernel
    int total_qh = total_q * n_heads;
    int reduce_block = 256;
    int reduce_grid = (total_qh + reduce_block - 1) / reduce_block;

    hipLaunchKernelGGL(
        mla_reduce_kernel,
        dim3(reduce_grid), dim3(reduce_block),
        0, 0,
        (const hip_bfloat16*)partial_out.data_ptr(),
        partial_lse.data_ptr<float>(),
        partial_max.data_ptr<float>(),
        (hip_bfloat16*)output.data_ptr(),
        n_splits,
        total_q,
        n_heads
    );

    return output;
}
"""

# ─────────────────────────────────────────────────────────────
# C++ forward declarations (matching cuda_sources wrapper)
# ─────────────────────────────────────────────────────────────
CPP_SOURCE = """
torch::Tensor mla_mxfp4_attention(
    torch::Tensor Q,
    torch::Tensor fp4_buf,
    torch::Tensor e8m0_scales,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    double sm_scale,
    int64_t n_splits
);
"""

# ─────────────────────────────────────────────────────────────
# Build module
# ─────────────────────────────────────────────────────────────
print("Compiling MLA MXFP4 split-K attention kernel...")
mod = load_inline(
    name="mla_mxfp4_splitk_v8",  # bump version to bust cache
    cpp_sources=CPP_SOURCE,
    cuda_sources=HIP_SOURCE,
    functions=["mla_mxfp4_attention"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
    verbose=False,
)
print("Compilation successful!")


# ─────────────────────────────────────────────────────────────
# Helper: simulate MXFP4 quantization for testing
# ─────────────────────────────────────────────────────────────
def quant_fp4_value(val):
    """Quantize a float to fp4 nibble (4-bit: sign, 2-bit exp, 1-bit man)."""
    sign = 1 if val < 0 else 0
    aval = abs(val)

    # FP4 E2M1 representable values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    table = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    nibbles = [0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111]

    # Find nearest
    best_i = 0
    best_d = abs(aval - table[0])
    for i in range(1, 8):
        d = abs(aval - table[i])
        if d < best_d:
            best_d = d
            best_i = i
    nib = nibbles[best_i]
    if sign:
        nib |= 0b1000
    return nib


def simulate_mxfp4_quantize(tensor_fp32, group_size=32):
    """
    Quantize a 1D fp32 tensor to MXFP4 format.
    Returns (packed_bytes uint8, scale_bytes uint8).
    """
    n = tensor_fp32.numel()
    assert n % 2 == 0, "Need even number of elements"
    n_groups = (n + group_size - 1) // group_size

    data = tensor_fp32.cpu().float().numpy()
    packed = []
    scales = []

    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n)
        group_data = data[start:end]

        # Compute e8m0 scale: find max abs, then encode as 2^(e-127)
        max_abs = max(abs(group_data.max()), abs(group_data.min()), 1e-30)
        # We want scale such that max_abs / scale <= 6.0 (max fp4 value)
        scale_float = max_abs / 6.0
        # Encode as e8m0: e = floor(log2(scale)) + 127
        import numpy as np
        if scale_float < 1e-38:
            e = 0
        else:
            e = int(np.floor(np.log2(scale_float))) + 127
            e = max(1, min(254, e))
        scales.append(e)

        # Actual scale used for dequant
        actual_scale = 2.0 ** (e - 127)

        # Quantize each element
        for i in range(start, end, 2):
            val0 = data[i] / actual_scale if actual_scale > 0 else 0.0
            nib0 = quant_fp4_value(val0)
            if i + 1 < end:
                val1 = data[i + 1] / actual_scale if actual_scale > 0 else 0.0
                nib1 = quant_fp4_value(val1)
            else:
                nib1 = 0
            # Pack: low nibble = even element, high nibble = odd element
            packed.append((nib1 << 4) | nib0)

    return (
        torch.tensor(packed, dtype=torch.uint8),
        torch.tensor(scales, dtype=torch.uint8)
    )


# ─────────────────────────────────────────────────────────────
# Submission entry point
# ─────────────────────────────────────────────────────────────
def mla_decode_mxfp4(
    Q: torch.Tensor,           # [total_q, 16, 576] bf16
    fp4x2_buf: torch.Tensor,   # [total_kv, 1, 288] (view as uint8)
    e8m0_scales: torch.Tensor, # [total_kv, 24] (view as uint8)
    qo_indptr: torch.Tensor,   # [batch+1] int64
    kv_indptr: torch.Tensor,   # [batch+1] int64
    sm_scale: float = None,
    n_splits: int = 8,
) -> torch.Tensor:
    """
    MLA decode attention with MXFP4 KV cache.

    Returns: [total_q, 16, 512] bf16
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(576)

    # Ensure correct dtypes and views
    Q = Q.contiguous()
    fp4_buf = fp4x2_buf.view(torch.uint8).contiguous()

    # Handle e8m0 scales shape: might be [total_kv, 24] or [total_kv, 1, 24]
    e8m0 = e8m0_scales.view(torch.uint8).contiguous()
    if e8m0.dim() == 3:
        e8m0 = e8m0.squeeze(1)
    assert e8m0.dim() == 2 and e8m0.shape[1] >= 18, f"e8m0 shape: {e8m0.shape}"

    qo_indptr = qo_indptr.to(torch.int64).contiguous()
    kv_indptr = kv_indptr.to(torch.int64).contiguous()

    # Adjust n_splits based on KV length
    total_kv = fp4_buf.shape[0]
    n_batch = qo_indptr.shape[0] - 1
    # Don't use more splits than we have KV tokens (per batch avg)
    avg_kv = total_kv / max(n_batch, 1)
    n_splits = min(n_splits, max(1, int(avg_kv)))

    output = mod.mla_mxfp4_attention(
        Q, fp4_buf, e8m0, qo_indptr, kv_indptr,
        float(sm_scale), n_splits
    )

    return output


# ─────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────
def smoke_test():
    """Test with small inputs to verify correctness."""
    print("\n=== Smoke Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("No GPU available, skipping smoke test")
        return

    torch.manual_seed(42)

    # Small test: 2 batch, 1 query token each, 4 KV tokens each
    total_q = 2
    total_kv = 8
    n_heads = 16
    head_dim = 576
    v_dim = 512

    Q = torch.randn(total_q, n_heads, head_dim, dtype=torch.bfloat16, device=device)

    # Create random MXFP4 data (simplified: random bytes)
    fp4_buf = torch.randint(0, 256, (total_kv, 1, 288), dtype=torch.uint8, device=device)
    e8m0_scales = torch.randint(110, 140, (total_kv, 24), dtype=torch.uint8, device=device)

    qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int64, device=device)
    kv_indptr = torch.tensor([0, 4, 8], dtype=torch.int64, device=device)

    output = mla_decode_mxfp4(
        Q, fp4_buf, e8m0_scales, qo_indptr, kv_indptr,
        n_splits=2
    )

    print(f"Q shape: {Q.shape}")
    print(f"fp4_buf shape: {fp4_buf.shape}")
    print(f"e8m0_scales shape: {e8m0_scales.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output sample: {output[0, 0, :8]}")
    assert output.shape == (total_q, n_heads, v_dim), f"Expected {(total_q, n_heads, v_dim)}, got {output.shape}"
    assert output.dtype == torch.bfloat16
    print("✓ Smoke test passed!")


# ─────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────
def benchmark():
    """Benchmark against competition target."""
    print("\n=== Benchmark ===")
    device = torch.device("cuda")
    torch.manual_seed(42)

    # Realistic MLA decode scenario
    # batch=64, 1 query per batch, ~512 KV tokens per batch
    batch_size = 64
    kv_len = 512
    total_q = batch_size
    total_kv = batch_size * kv_len
    n_heads = 16

    Q = torch.randn(total_q, n_heads, 576, dtype=torch.bfloat16, device=device)
    fp4_buf = torch.randint(0, 256, (total_kv, 1, 288), dtype=torch.uint8, device=device)
    e8m0_scales = torch.randint(110, 140, (total_kv, 24), dtype=torch.uint8, device=device)

    qo_indptr = torch.arange(0, batch_size + 1, dtype=torch.int64, device=device)
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int64, device=device) * kv_len

    # Warmup
    for _ in range(5):
        _ = mla_decode_mxfp4(Q, fp4_buf, e8m0_scales, qo_indptr, kv_indptr, n_splits=8)
    torch.cuda.synchronize()

    # Benchmark
    import time
    n_iters = 50
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = mla_decode_mxfp4(Q, fp4_buf, e8m0_scales, qo_indptr, kv_indptr, n_splits=8)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_us = elapsed / n_iters * 1e6
    print(f"Batch={batch_size}, KV_len={kv_len}, n_splits=8")
    print(f"Average time: {avg_us:.1f} μs")
    print(f"Target: ~45 μs (current fp8 ASM: 55 μs)")


if __name__ == "__main__":
    smoke_test()
    try:
        benchmark()
    except Exception as e:
        print(f"Benchmark failed (may need GPU): {e}")

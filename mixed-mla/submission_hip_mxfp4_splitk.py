
import torch
from torch.utils.cpp_extension import load_inline

# ── HIP/CUDA source ───────────────────────────────────────────────────────────
CUDA_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>
#include <stdint.h>
#include <float.h>
#include <math.h>

#define HEAD_DIM_QK  576
#define HEAD_DIM_V   512
#define PACKED_BYTES 288
#define SCALE_STRIDE 24

// ─── E2M1 + E8M0 dequant ─────────────────────────────────────────────────────
__device__ __forceinline__ float dequant_fp4(uint8_t nibble, float scale) {
    int sign = (nibble >> 3) & 1;
    int exp  = (nibble >> 1) & 3;
    int man  = nibble & 1;
    float mag;
    if (exp == 0) mag = man * 0.5f;
    else          mag = ldexpf(1.0f + man * 0.5f, exp - 1);
    return sign ? -mag * scale : mag * scale;
}

// ─── Split-K main kernel ──────────────────────────────────────────────────────
// Grid:  (n_splits, n_heads, n_batch)
// Block: (128)
__global__ void mla_mxfp4_splitk_kernel(
    const hip_bfloat16* __restrict__ Q,
    const uint8_t*         __restrict__ fp4x2_buf,
    const uint8_t*         __restrict__ e8m0_scales,
    const int*             __restrict__ qo_indptr,
    const int*             __restrict__ kv_indptr,
    hip_bfloat16*        __restrict__ partial_out,
    float*                 __restrict__ partial_lse,
    int n_heads,
    int n_splits,
    float sm_scale
) {
    const int split_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int batch_idx = blockIdx.z;

    const int q_start  = qo_indptr[batch_idx];
    const int kv_start = kv_indptr[batch_idx];
    const int kv_end   = kv_indptr[batch_idx + 1];
    const int kv_len   = kv_end - kv_start;
    const int qi       = q_start;
    const int flat_qh  = qi * n_heads + head_idx;

    if (kv_len <= 0) {
        partial_lse[flat_qh * n_splits + split_idx] = -FLT_MAX;
        int base = (flat_qh * n_splits + split_idx) * HEAD_DIM_V;
        for (int i = threadIdx.x; i < HEAD_DIM_V; i += blockDim.x)
            partial_out[base + i] = (hip_bfloat16)(0.0f);
        return;
    }

    const int chunk_size     = (kv_len + n_splits - 1) / n_splits;
    const int kv_chunk_start = kv_start + split_idx * chunk_size;
    const int kv_chunk_end   = min(kv_start + (split_idx + 1) * chunk_size, kv_end);

    if (kv_chunk_start >= kv_chunk_end) {
        partial_lse[flat_qh * n_splits + split_idx] = -FLT_MAX;
        int base = (flat_qh * n_splits + split_idx) * HEAD_DIM_V;
        for (int i = threadIdx.x; i < HEAD_DIM_V; i += blockDim.x)
            partial_out[base + i] = (hip_bfloat16)(0.0f);
        return;
    }

    // Load Q into shared memory
    __shared__ float sq[HEAD_DIM_QK];
    const hip_bfloat16* q_ptr = Q + flat_qh * HEAD_DIM_QK;
    for (int i = threadIdx.x; i < HEAD_DIM_QK; i += blockDim.x)
        sq[i] = (float)(q_ptr[i]);
    __syncthreads();

    // V accumulators: 512/128 = 4 per thread, strided
    const int vi0 = threadIdx.x;
    const int vi1 = threadIdx.x + 128;
    const int vi2 = threadIdx.x + 256;
    const int vi3 = threadIdx.x + 384;
    float v_acc_0 = 0.0f, v_acc_1 = 0.0f, v_acc_2 = 0.0f, v_acc_3 = 0.0f;

    float running_max = -FLT_MAX;
    float running_sum = 0.0f;

    __shared__ float skv[HEAD_DIM_QK];
    __shared__ float score_buf;

    for (int kv_idx = kv_chunk_start; kv_idx < kv_chunk_end; kv_idx++) {
        const uint8_t* sc_ptr = e8m0_scales + kv_idx * SCALE_STRIDE;
        const uint8_t* kv_ptr = fp4x2_buf   + kv_idx * PACKED_BYTES;

        // Dequant 288 packed bytes -> 576 fp32
        for (int i = threadIdx.x; i < PACKED_BYTES; i += blockDim.x) {
            int g = i >> 4;  // i / 16
            float scale = ldexpf(1.0f, (int)sc_ptr[g] - 127);
            uint8_t packed = kv_ptr[i];
            skv[2*i]     = dequant_fp4(packed & 0xF,        scale);
            skv[2*i + 1] = dequant_fp4((packed >> 4) & 0xF, scale);
        }
        __syncthreads();

        // Dot product Q . KV (576-dim)
        float dot = 0.0f;
        for (int i = threadIdx.x; i < HEAD_DIM_QK; i += blockDim.x)
            dot += sq[i] * skv[i];

        // Wave64 reduction
        dot += __shfl_down(dot, 32);
        dot += __shfl_down(dot, 16);
        dot += __shfl_down(dot,  8);
        dot += __shfl_down(dot,  4);
        dot += __shfl_down(dot,  2);
        dot += __shfl_down(dot,  1);

        // Cross-wave reduction (2 waves per 128-thread block)
        __shared__ float wave_dots[2];
        if ((threadIdx.x & 63) == 0) wave_dots[threadIdx.x >> 6] = dot;
        __syncthreads();

        float score;
        if (threadIdx.x == 0) {
            score_buf = (wave_dots[0] + wave_dots[1]) * sm_scale;
        }
        __syncthreads();
        score = score_buf;

        // Online softmax
        float new_max   = fmaxf(running_max, score);
        float exp_score = expf(score - new_max);
        float rescale   = expf(running_max - new_max);

        v_acc_0 = v_acc_0 * rescale + exp_score * skv[vi0];
        v_acc_1 = v_acc_1 * rescale + exp_score * skv[vi1];
        v_acc_2 = v_acc_2 * rescale + exp_score * skv[vi2];
        v_acc_3 = v_acc_3 * rescale + exp_score * skv[vi3];

        running_sum = running_sum * rescale + exp_score;
        running_max = new_max;

        __syncthreads();
    }

    // Write normalized partial output
    float inv_sum = (running_sum > 0.0f) ? 1.0f / running_sum : 0.0f;
    int base_out = (flat_qh * n_splits + split_idx) * HEAD_DIM_V;
    partial_out[base_out + vi0] = (hip_bfloat16)(v_acc_0 * inv_sum);
    partial_out[base_out + vi1] = (hip_bfloat16)(v_acc_1 * inv_sum);
    partial_out[base_out + vi2] = (hip_bfloat16)(v_acc_2 * inv_sum);
    partial_out[base_out + vi3] = (hip_bfloat16)(v_acc_3 * inv_sum);

    if (threadIdx.x == 0) {
        partial_lse[flat_qh * n_splits + split_idx] =
            (running_sum > 0.0f) ? logf(running_sum) + running_max : -FLT_MAX;
    }
}

// ─── Reduce kernel ────────────────────────────────────────────────────────────
// Grid: (total_q * n_heads,)   Block: (128)
__global__ void mla_mxfp4_splitk_reduce(
    const hip_bfloat16* __restrict__ partial_out,
    const float*           __restrict__ partial_lse,
    hip_bfloat16*        __restrict__ out,
    int n_heads,
    int n_splits,
    int total_q
) {
    const int flat_idx = blockIdx.x;
    if (flat_idx >= total_q * n_heads) return;

    float global_max = -FLT_MAX;
    for (int s = 0; s < n_splits; s++) {
        float lse = partial_lse[flat_idx * n_splits + s];
        if (lse > global_max) global_max = lse;
    }

    if (global_max == -FLT_MAX) {
        int base = flat_idx * HEAD_DIM_V;
        for (int i = threadIdx.x; i < HEAD_DIM_V; i += blockDim.x)
            out[base + i] = (hip_bfloat16)(0.0f);
        return;
    }

    float weights[16];
    float norm = 0.0f;
    for (int s = 0; s < n_splits; s++) {
        float lse = partial_lse[flat_idx * n_splits + s];
        float w   = (lse > -FLT_MAX + 1.0f) ? expf(lse - global_max) : 0.0f;
        weights[s] = w;
        norm += w;
    }
    float inv_norm = (norm > 0.0f) ? 1.0f / norm : 0.0f;

    int out_base = flat_idx * HEAD_DIM_V;
    for (int vi = threadIdx.x; vi < HEAD_DIM_V; vi += blockDim.x) {
        float acc = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            float val = (float)(
                partial_out[(flat_idx * n_splits + s) * HEAD_DIM_V + vi]
            );
            acc += weights[s] * val;
        }
        out[out_base + vi] = (hip_bfloat16)(acc * inv_norm);
    }
}

// ─── Torch launchers ──────────────────────────────────────────────────────────

void launch_splitk(
    torch::Tensor Q,
    torch::Tensor fp4x2_buf,
    torch::Tensor e8m0_scales,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor partial_out,
    torch::Tensor partial_lse,
    int n_heads,
    int n_splits,
    float sm_scale
) {
    int n_batch = (int)qo_indptr.size(0) - 1;
    dim3 grid(n_splits, n_heads, n_batch);
    dim3 block(128);
    hipLaunchKernelGGL(
        mla_mxfp4_splitk_kernel,
        grid, block, 0, 0,
        (const hip_bfloat16*)Q.data_ptr(),
        (const uint8_t*)fp4x2_buf.data_ptr(),
        (const uint8_t*)e8m0_scales.data_ptr(),
        (const int*)qo_indptr.data_ptr(),
        (const int*)kv_indptr.data_ptr(),
        (hip_bfloat16*)partial_out.data_ptr(),
        (float*)partial_lse.data_ptr(),
        n_heads,
        n_splits,
        sm_scale
    );
}

void launch_reduce(
    torch::Tensor partial_out,
    torch::Tensor partial_lse,
    torch::Tensor out,
    int n_heads,
    int n_splits,
    int total_q
) {
    dim3 grid(total_q * n_heads);
    dim3 block(128);
    hipLaunchKernelGGL(
        mla_mxfp4_splitk_reduce,
        grid, block, 0, 0,
        (const hip_bfloat16*)partial_out.data_ptr(),
        (const float*)partial_lse.data_ptr(),
        (hip_bfloat16*)out.data_ptr(),
        n_heads,
        n_splits,
        total_q
    );
}

"""

# ── Python wrapper ────────────────────────────────────────────────────────────

_module = None

def _get_module():
    global _module
    if _module is not None:
        return _module
    import os
    os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
    _module = load_inline(
        name="mla_mxfp4_splitk_v4",
        cpp_sources="torch::Tensor launch_splitk(torch::Tensor Q, torch::Tensor fp4x2_buf, torch::Tensor e8m0_scales, torch::Tensor qo_indptr, torch::Tensor kv_indptr, int64_t n_heads, int64_t n_splits, double sm_scale, int64_t total_q); torch::Tensor launch_reduce(torch::Tensor partial_out, torch::Tensor partial_lse, torch::Tensor qo_indptr, int64_t n_heads, int64_t n_splits, int64_t total_q);",
        cuda_sources=CUDA_SRC,
        functions=["launch_splitk", "launch_reduce"],
        extra_cuda_cflags=[
            "--offload-arch=gfx950",
            "-O3",
            "-ffast-math",
        ],
        verbose=False,
    )
    return _module


def _choose_n_splits(kv_indptr, n_batch):
    if n_batch == 0:
        return 4
    total = int(kv_indptr[-1].item()) - int(kv_indptr[0].item())
    avg_kv = total / n_batch
    if   avg_kv <= 128:   return 2
    elif avg_kv <= 512:   return 4
    elif avg_kv <= 2048:  return 8
    else:                 return 16


def custom_kernel(data):
    queries, kv_data, qo_indptr, kv_indptr, config = data

    fp4x2_buf, e8m0_scales = kv_data["mxfp4"]
    fp4x2_buf   = fp4x2_buf.contiguous().view(torch.uint8)
    e8m0_scales = e8m0_scales.contiguous().view(torch.uint8)

    if fp4x2_buf.dim() == 3:
        fp4x2_buf = fp4x2_buf[:, 0, :].contiguous()

    queries = queries.contiguous()

    total_q  = queries.shape[0]
    n_heads  = queries.shape[1]
    n_batch  = int(qo_indptr.shape[0]) - 1
    n_splits   = _choose_n_splits(kv_indptr, n_batch)
    HEAD_DIM_V = 512
    sm_scale   = 1.0 / (576 ** 0.5)

    partial_out = torch.zeros(
        (total_q, n_heads, n_splits, HEAD_DIM_V),
        dtype=torch.bfloat16, device=queries.device
    )
    partial_lse = torch.full(
        (total_q, n_heads, n_splits),
        float('-inf'), dtype=torch.float32, device=queries.device
    )
    out = torch.empty(
        (total_q, n_heads, HEAD_DIM_V),
        dtype=torch.bfloat16, device=queries.device
    )

    qo_i = qo_indptr.to(torch.int32).contiguous()
    kv_i = kv_indptr.to(torch.int32).contiguous()

    mod = _get_module()
    mod.launch_splitk(
        queries, fp4x2_buf, e8m0_scales,
        qo_i, kv_i,
        partial_out, partial_lse,
        n_heads, n_splits, sm_scale
    )
    mod.launch_reduce(
        partial_out, partial_lse, out,
        n_heads, n_splits, total_q
    )

    return out

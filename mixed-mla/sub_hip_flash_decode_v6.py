#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP Flash-Decoding MLA v6 -- Wave-parallel dot products for AMD MI355X (gfx950).

Key changes vs v5 (scalar dot + shared-memory tree reduction):
  - 4 waves/block; each wave processes one KV token independently per iteration
  - Dot product via __shfl_xor wave-reduction (NO shared memory tree)
  - V accumulation: each thread owns 8 of 512 V dims
  - Heads processed in groups of 4 to manage register pressure (~136 VGPRs/thread)
  - Vectorized 4-byte loads for V data
  - ZERO __syncthreads inside the KV loop
  - Cross-wave merge via LDS after KV loop

v5 bottleneck: per KV token, 16 heads x 9 shared-memory reduction steps = 144 barriers.
v6: zero barriers per KV token. Only barriers are Q-load, and 16-head x 2 for merge.
"""

import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#define NUM_HEADS 16
#define QK_DIM    576
#define V_DIM     512
#define THREADS   256
#define WAVE_SIZE 64
#define NUM_WAVES 4

// 576/64 = 9 QK dims per lane for dot product
#define DIMS_PER_LANE 9
// 512/64 = 8 V dims per lane for output accumulation
#define V_PER_LANE    8
// Process heads in groups to limit register pressure
// 4 heads at a time: 4*(9 Q + 8 V + 2 ml) = 76 regs + 9 kv + 8 vv = 93 peak
#define HEAD_GROUP    4
#define NUM_HGROUPS   (NUM_HEADS / HEAD_GROUP)  // 16/4 = 4

// ---------- wave-level butterfly reduction ----------
__device__ __forceinline__ float wave_reduce_sum(float val) {
    val += __shfl_xor(val, 32);
    val += __shfl_xor(val, 16);
    val += __shfl_xor(val, 8);
    val += __shfl_xor(val, 4);
    val += __shfl_xor(val, 2);
    val += __shfl_xor(val, 1);
    return val;
}

// ============================================================================
// Stage 1 -- flash-decoding partial attention
// Grid : (batch_size, num_splits)
// Block: 256 threads = 4 waves
//
// Register budget (per lane, per head-group of 4 heads):
//   qr [4][9] = 36, ov [4][8] = 32, m_h[4] = 4, l_h[4] = 4 => 76 persistent
//   kv [9] = 9, vv [8] = 8 => 17 temporary
//   Total peak: ~93 VGPRs -- fits 4 waves on CDNA4 (512 VGPRs / 4 = 128 per wave)
//
// We loop over 4 head-groups, reloading Q each time from LDS.
// KV data must be reloaded from global memory for each head-group
// (or we can iterate KV inside head-group).
//
// Actually, the optimal structure: for each KV token, process ALL head-groups,
// loading Q from LDS (cheap) while KV stays in registers. But then we need
// to keep all 16 heads' ov/m/l live = 16*(8+2) = 160 regs. That plus Q(9) +
// kv(9) + vv(8) = 186 regs. With 4 waves: 186 VGPRs needed but only 128
// available per wave (512/4). Spill would occur.
//
// Best compromise: process 2 KV tokens per wave per iteration, 8 heads each.
// Actually simplest: iterate head-groups INSIDE the KV loop.
// Keep only current head-group's state in regs, store/reload between groups.
// But storing to LDS costs time too.
//
// SIMPLEST HIGH-PERF APPROACH: Let compiler handle spill.
// With __launch_bounds__(256, 1) we tell the compiler we want 1 block per CU,
// which gives it all 512 VGPRs / 4 waves = 128 VGPRs per wave.
// Our 93 regs/head-group fit. But across 4 head-groups we need the state
// for all heads to persist. The compiler will spill some to scratch.
//
// ACTUALLY: the compiler sees the full loop and knows all arrays are live.
// q_reg[16][9]=144, ov[16][8]=128, m_h[16]=16, l_h[16]=16 = 304 live regs.
// That's 304 * 4 bytes = 1216 bytes of scratch per lane. Unacceptable.
//
// REAL SOLUTION: make 4 separate passes over the KV range, one per head-group.
// Each pass: load Q for 4 heads from LDS, iterate all KV tokens, write partial.
// This trades KV re-reads (from global DRAM) for eliminating register spill.
// 4x KV loads but zero spill. KV is fp8 (1 byte/elem) so bandwidth is:
//   sp_len * 576 bytes * 4 passes = sp_len * 2304 bytes
// vs. original v5's sp_len * 576 * 1 pass but with shared-mem reduction.
// For kv_seq_len=8192, 8 splits: sp_len~1024, 2304*1024 = 2.3MB.
// L2 cache is 256 MB on MI355X. Second pass hits L2. Negligible cost.
//
// ============================================================================

__global__ __launch_bounds__(256, 1)
void mla_stage1(
    const hip_bfloat16* __restrict__ Q,
    const unsigned char* __restrict__ KV,
    const float          kv_scale,
    const int*  __restrict__ qo_indptr,
    const int*  __restrict__ kv_indptr,
    float*      __restrict__ partial_o,
    float*      __restrict__ partial_lse,
    const float sm_scale,
    const int   num_splits)
{
    const int batch_idx = blockIdx.x;
    const int split_idx = blockIdx.y;
    const int tid       = threadIdx.x;
    const int wave_id   = tid / WAVE_SIZE;
    const int lane_id   = tid % WAVE_SIZE;

    const int kv_start = kv_indptr[batch_idx];
    const int kv_end   = kv_indptr[batch_idx + 1];
    const int kv_len   = kv_end - kv_start;

    const int tps      = (kv_len + num_splits - 1) / num_splits;
    const int sp_st    = tps * split_idx;
    const int sp_en_r  = sp_st + tps;
    const int sp_en    = sp_en_r < kv_len ? sp_en_r : kv_len;
    const int sp_len   = (sp_st < kv_len) ? (sp_en - sp_st) : 0;

    const int o_base   = (batch_idx * num_splits + split_idx) * NUM_HEADS * V_DIM;
    const int lse_base = (batch_idx * num_splits + split_idx) * NUM_HEADS;

    if (sp_len == 0) {
        for (int h = tid; h < NUM_HEADS; h += THREADS)
            partial_lse[lse_base + h] = -1e30f;
        for (int i = tid; i < NUM_HEADS * V_DIM; i += THREADS)
            partial_o[o_base + i] = 0.0f;
        return;
    }

    const int q_idx = qo_indptr[batch_idx];
    const hip_bfloat16* q_base = Q + (long long)q_idx * NUM_HEADS * QK_DIM;

    // ---- shared memory ----
    __shared__ float s_q[NUM_HEADS * QK_DIM];   // 9216 floats = 36864 B
    __shared__ float fp8_lut[256];               //  256 floats =  1024 B
    // merge workspace (reuse s_q after Q no longer needed? No -- Q needed for all passes)
    // separate merge workspace:
    __shared__ float s_wml[NUM_WAVES * 2];       //    8 floats =    32 B
    __shared__ float s_wv[NUM_WAVES * V_DIM];    // 2048 floats =  8192 B
    __shared__ float s_lse[NUM_HEADS];           //   16 floats =    64 B
    // Total: ~46 KB

    // build fp8 LUT
    {
        unsigned char b = (unsigned char)tid;
        if (b == 0x80 || b == 0) {
            fp8_lut[tid] = 0.0f;
        } else {
            int sign = (b >> 7) & 1;
            int ex   = (b >> 3) & 0xF;
            int mn   = b & 0x7;
            float v;
            if (ex == 0) v = ldexpf((float)mn / 8.0f, 1 - 8);
            else         v = ldexpf(1.0f + (float)mn / 8.0f, ex - 8);
            fp8_lut[tid] = sign ? -v : v;
        }
    }
    __syncthreads();

    // load all Q into LDS
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += THREADS)
        s_q[i] = (float)q_base[i];
    __syncthreads();

    // lane dim assignments
    const int qk_base = lane_id * DIMS_PER_LANE;   // 0,9,...,567
    const int v_base  = lane_id * V_PER_LANE;       // 0,8,...,504

    // Global KV base for this split
    const long long kv_split_base = (long long)(kv_start + sp_st) * QK_DIM;

    // ================================================================
    // Process heads in groups of HEAD_GROUP (4), making 4 passes over KV
    // ================================================================
    for (int hg = 0; hg < NUM_HGROUPS; hg++) {
        const int h_off = hg * HEAD_GROUP;  // starting head index

        // Load Q for these 4 heads into registers
        float qr[HEAD_GROUP][DIMS_PER_LANE];
        #pragma unroll
        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++)
                qr[hh][d] = s_q[(h_off + hh) * QK_DIM + qk_base + d];
        }

        // Online softmax state for these 4 heads
        float m_h[HEAD_GROUP], l_h[HEAD_GROUP];
        float ov[HEAD_GROUP][V_PER_LANE];
        #pragma unroll
        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            m_h[hh] = -1e30f;
            l_h[hh] = 0.0f;
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++) ov[hh][v] = 0.0f;
        }

        // KV loop: wave round-robin
        for (int t = wave_id; t < sp_len; t += NUM_WAVES) {
            const long long kv_off = kv_split_base + (long long)t * QK_DIM;

            // load 9 KV dims for dot product (byte-by-byte, non-aligned offset)
            const unsigned char* kp = KV + kv_off + qk_base;
            float kd[DIMS_PER_LANE];
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++)
                kd[d] = fp8_lut[kp[d]] * kv_scale;

            // load 8 V dims (vectorized: v_base is multiple of 8)
            const unsigned char* vp = KV + kv_off + v_base;
            float vv[V_PER_LANE];
            unsigned int vw0 = ((const unsigned int*)vp)[0];
            unsigned int vw1 = ((const unsigned int*)vp)[1];
            vv[0] = fp8_lut[(vw0      ) & 0xFF] * kv_scale;
            vv[1] = fp8_lut[(vw0 >>  8) & 0xFF] * kv_scale;
            vv[2] = fp8_lut[(vw0 >> 16) & 0xFF] * kv_scale;
            vv[3] = fp8_lut[(vw0 >> 24)        ] * kv_scale;
            vv[4] = fp8_lut[(vw1      ) & 0xFF] * kv_scale;
            vv[5] = fp8_lut[(vw1 >>  8) & 0xFF] * kv_scale;
            vv[6] = fp8_lut[(vw1 >> 16) & 0xFF] * kv_scale;
            vv[7] = fp8_lut[(vw1 >> 24)        ] * kv_scale;

            // dot product + softmax + V accum for each head in this group
            #pragma unroll
            for (int hh = 0; hh < HEAD_GROUP; hh++) {
                float dot = qr[hh][0] * kd[0]
                          + qr[hh][1] * kd[1]
                          + qr[hh][2] * kd[2]
                          + qr[hh][3] * kd[3]
                          + qr[hh][4] * kd[4]
                          + qr[hh][5] * kd[5]
                          + qr[hh][6] * kd[6]
                          + qr[hh][7] * kd[7]
                          + qr[hh][8] * kd[8];

                float score = wave_reduce_sum(dot) * sm_scale;

                float old_m = m_h[hh];
                float new_m = fmaxf(old_m, score);
                float e_old = __expf(old_m - new_m);
                float e_new = __expf(score - new_m);

                #pragma unroll
                for (int v = 0; v < V_PER_LANE; v++)
                    ov[hh][v] = ov[hh][v] * e_old + e_new * vv[v];
                l_h[hh] = l_h[hh] * e_old + e_new;
                m_h[hh] = new_m;
            }
        } // end KV loop -- ZERO syncthreads inside

        // ---- cross-wave merge for this head group ----
        __syncthreads();

        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            const int h = h_off + hh;

            if (lane_id == 0) {
                s_wml[wave_id * 2    ] = m_h[hh];
                s_wml[wave_id * 2 + 1] = l_h[hh];
            }
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++)
                s_wv[wave_id * V_DIM + v_base + v] = ov[hh][v];

            __syncthreads();

            float mg_m = -1e30f, mg_l = 0.0f;
            float mg_o[V_PER_LANE];
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++) mg_o[v] = 0.0f;

            #pragma unroll
            for (int w = 0; w < NUM_WAVES; w++) {
                float wm = s_wml[w * 2    ];
                float wl = s_wml[w * 2 + 1];
                if (wl > 0.0f) {
                    float nm = fmaxf(mg_m, wm);
                    float so = __expf(mg_m - nm);
                    float sw = __expf(wm   - nm);
                    #pragma unroll
                    for (int v = 0; v < V_PER_LANE; v++)
                        mg_o[v] = mg_o[v] * so + sw * s_wv[w * V_DIM + v_base + v];
                    mg_l = mg_l * so + sw * wl;
                    mg_m = nm;
                }
            }

            float inv = (mg_l > 0.0f) ? (1.0f / mg_l) : 0.0f;
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++)
                partial_o[o_base + h * V_DIM + v_base + v] = mg_o[v] * inv;

            if (tid == 0)
                s_lse[h] = (mg_l > 0.0f) ? (mg_m + __logf(mg_l)) : -1e30f;

            __syncthreads();
        }
    } // end head-group loop

    // write LSE
    if (tid < NUM_HEADS)
        partial_lse[lse_base + tid] = s_lse[tid];
}


// ============================================================================
// Stage 2 -- reduce across splits
// Grid : (batch_size, NUM_HEADS)
// Block: 256 threads -- each thread owns 2 V dims
// ============================================================================

__global__ __launch_bounds__(256)
void mla_stage2(
    const float* __restrict__ partial_o,
    const float* __restrict__ partial_lse,
    hip_bfloat16* __restrict__ output,
    const int*  __restrict__ qo_indptr,
    const int   num_splits)
{
    const int batch_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int tid       = threadIdx.x;
    const int q_idx     = qo_indptr[batch_idx];
    const int d0 = tid, d1 = tid + 256;

    float e_max = -1e30f, e_sum = 0.0f, a0 = 0.0f, a1 = 0.0f;

    for (int s = 0; s < num_splits; s++) {
        int li = (batch_idx * num_splits + s) * NUM_HEADS + head_idx;
        float lse = partial_lse[li];
        if (lse > -1e29f) {
            int oi = (batch_idx * num_splits + s) * NUM_HEADS * V_DIM + head_idx * V_DIM;
            float v0 = partial_o[oi + d0];
            float v1 = partial_o[oi + d1];
            float nm = fmaxf(e_max, lse);
            float os = __expf(e_max - nm);
            float ns = __expf(lse   - nm);
            a0    = a0    * os + ns * v0;
            a1    = a1    * os + ns * v1;
            e_sum = e_sum * os + ns;
            e_max = nm;
        }
    }

    float inv = (e_sum > 0.0f) ? (1.0f / e_sum) : 0.0f;
    long long ob = (long long)q_idx * NUM_HEADS * V_DIM + (long long)head_idx * V_DIM;
    output[ob + d0] = (hip_bfloat16)(a0 * inv);
    output[ob + d1] = (hip_bfloat16)(a1 * inv);
}


// ============================================================================
// C++ wrapper
// ============================================================================

torch::Tensor mla_flash_decode(
    torch::Tensor Q,
    torch::Tensor KV,
    float kv_scale_val,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    float sm_scale_val,
    int num_splits_val)
{
    const int batch_size = qo_indptr.size(0) - 1;
    const int total_q    = Q.size(0);

    auto of32  = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto po    = torch::empty({batch_size, num_splits_val, NUM_HEADS, V_DIM}, of32);
    auto plse  = torch::empty({batch_size, num_splits_val, NUM_HEADS},        of32);

    auto obf16 = torch::TensorOptions().dtype(torch::kBFloat16).device(Q.device());
    auto out   = torch::empty({total_q, NUM_HEADS, V_DIM}, obf16);

    dim3 g1(batch_size, num_splits_val);
    dim3 b1(THREADS);
    hipLaunchKernelGGL(mla_stage1, g1, b1, 0, 0,
        (const hip_bfloat16*)Q.data_ptr(),
        (const unsigned char*)KV.data_ptr(),
        kv_scale_val,
        (const int*)qo_indptr.data_ptr(),
        (const int*)kv_indptr.data_ptr(),
        (float*)po.data_ptr(),
        (float*)plse.data_ptr(),
        sm_scale_val, num_splits_val);

    dim3 g2(batch_size, NUM_HEADS);
    dim3 b2(256);
    hipLaunchKernelGGL(mla_stage2, g2, b2, 0, 0,
        (const float*)po.data_ptr(),
        (const float*)plse.data_ptr(),
        (hip_bfloat16*)out.data_ptr(),
        (const int*)qo_indptr.data_ptr(),
        num_splits_val);

    return out;
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

_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="mla_flash_decode_v6",
            cpp_sources=CPP_SOURCE,
            cuda_sources=HIP_SOURCE,
            functions=["mla_flash_decode"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
    return _module

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    kv_seq_len = config["kv_seq_len"]
    sm_scale   = config["sm_scale"]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]

    total_kv = kv_buffer_fp8.shape[0]
    kv_flat  = kv_buffer_fp8.view(total_kv, 576)

    kv_scale_val = kv_scale.item()

    total_tokens = batch_size * kv_seq_len
    if total_tokens <= 4096:
        num_splits = 4
    elif total_tokens <= 32768:
        num_splits = 8
    else:
        num_splits = 16

    mod = _get_module()

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

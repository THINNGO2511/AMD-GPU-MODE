#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP Flash-Decoding MLA v7 -- Coalesced KV loads + single-pass all heads.

Key changes vs v6:
  1. Coalesced KV loading: threads load consecutive bytes from global mem
     into LDS, then read strided from LDS. Global reads are fully coalesced.
  2. Vectorized global loads: use uint4 (16 bytes) for KV coalesced loads.
  3. Single-pass over all 16 heads: KV loaded once per token, not 4x.
     All 16 heads' m/l/ov state kept live. Register pressure managed by
     keeping Q in LDS and loading per-head Q from LDS inside the loop.
  4. Pre-multiply kv_scale into Q: eliminates per-element kv_scale multiply
     on KV data. score = sum(Q*kv_scale * KV_fp32) * sm_scale.
     V accum also uses pre-scaled approach via post-multiply.

Architecture: 256 threads = 4 waves, 64 lanes/wave.
  - Each wave processes 1 KV token at a time (round-robin).
  - 576 QK dims: 9 per lane (576/64). Dot via wave shuffle reduce.
  - 512 V dims: 8 per lane (512/64). Output accumulation distributed.
  - 16 heads all processed per KV token (single pass).
  - Register budget per lane: 16*(8 ov + 1 m + 1 l) = 160 persistent
    + 9 kv_dot + 8 kv_v + 9 q_tmp = 186 peak.
    With __launch_bounds__(256,1): 512 VGPRs / 4 waves = 128 per wave.
    186 > 128, so some spill to scratch. But scratch is per-lane private
    memory backed by VRAM, and the access pattern is regular. The
    compiler will optimize spill/reload. This is MUCH better than 4x
    global KV re-reads (v6's approach).
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

// Coalesced load: 256 threads x 4 bytes = 1024 bytes per iteration
// QK_DIM = 576 bytes per token. ceil(576/1024) = 1 iteration (576 threads active)
// Pad to 1024 for simplicity (waste 448 bytes, harmless).
// Actually: 576 = 256*2 + 64. With 2-byte loads: 2 full iters (512 bytes) + 64 leftover.
// Better: use 4-byte loads for first pass (256*4=1024 > 576), only 144 threads active.
// Simplest: each thread loads ceil(576/256)=3 bytes. 3 iterations of 1 byte? No, that's slow.
//
// Best approach: 256 threads, each loads 4 bytes via uint32. That's 1024 bytes.
// 576 bytes needed: first 144 threads load 4 bytes each = 576 bytes. Remaining 112 idle.
// But this requires 4-byte alignment. KV rows are 576 bytes, NOT 4-byte aligned across rows.
// Row t starts at offset t*576. 576 = 144*4. So each row IS 4-byte aligned IF base is aligned.
// fp8 KV tensor from PyTorch: data_ptr is at least 256-byte aligned. Good.
// Row offset = t*576 = t*4*144. Always 4-byte aligned.
//
// Plan: 144 threads load uint32 each (576 bytes). Store to LDS as bytes.
// Then each lane reads its 9 QK dims + 8 V dims from LDS.
//
// Even better: use uint4 (16 bytes) loads. 576/16 = 36 loads needed.
// 36 threads each load 16 bytes. But uint4 requires 16-byte alignment.
// 576 is NOT divisible by 16 (576 = 36*16). It is! 36 exactly.
// Row offset t*576 = t*576. 576 mod 16 = 0. So rows are 16-byte aligned.
// 36 threads load uint4 -> 576 bytes in one shot.

// LDS layout for coalesced KV: one row of 576 bytes per wave
// 4 waves -> 4 rows -> 4*576 = 2304 bytes for KV staging
// Plus Q storage: 16*576*4 = 36864 bytes (float)
// Plus fp8 LUT: 256*4 = 1024 bytes
// Plus merge workspace: (4*2 + 4*512 + 16)*4 = 8288 bytes
// Total: ~48 KB -- fits in 64KB LDS

// Number of uint4 (16-byte) loads to fill 576 bytes
#define KV_UINT4_LOADS   36   // 576/16

// ============================================================================

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
// Stage 1 -- flash-decoding partial attention, single-pass all 16 heads
// Grid : (batch_size, num_splits)
// Block: 256 threads = 4 waves
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

    // ---- shared memory layout ----
    // s_q:     16 heads * 576 dims * 4 bytes = 36864 bytes (Q, pre-scaled by kv_scale)
    // fp8_lut: 256 entries * 4 bytes = 1024 bytes
    // s_kv:    4 waves * 576 bytes = 2304 bytes (coalesced KV staging, raw uint8)
    // s_wml:   4 waves * 2 floats = 32 bytes (merge workspace)
    // s_wv:    4 waves * 512 floats = 8192 bytes (merge workspace)
    // s_lse:   16 floats = 64 bytes
    // Total: ~48.5 KB
    extern __shared__ char smem[];

    float* s_q          = (float*)smem;                               // 36864 B
    float* fp8_lut      = s_q + NUM_HEADS * QK_DIM;                   // 1024 B
    unsigned char* s_kv = (unsigned char*)(fp8_lut + 256);             // 2304 B
    // Align s_wml to 4 bytes (s_kv is 2304 bytes, already aligned)
    float* s_wml        = (float*)(s_kv + NUM_WAVES * QK_DIM);        // 32 B
    float* s_wv         = s_wml + NUM_WAVES * 2;                      // 8192 B
    float* s_lse        = s_wv + NUM_WAVES * V_DIM;                   // 64 B

    // build fp8 LUT (no kv_scale baked in -- we pre-scale Q instead)
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

    // Load all Q into LDS, pre-multiplied by kv_scale
    // score = sum(Q[d] * KV[d]) * kv_scale * sm_scale
    //       = sum((Q[d]*kv_scale) * KV[d]) * sm_scale
    // So we store Q * kv_scale in LDS. Then KV doesn't need kv_scale multiply.
    // For V accumulation: ov += e_new * V[d] (raw, no kv_scale)
    //   but final output = ov / l, and the "true" V = KV * kv_scale.
    //   So we need to multiply output by kv_scale at the end.
    //   OR: bake kv_scale into V at load time (cheap, in LDS read path).
    // Decision: pre-scale Q for dot product. Post-scale V output by kv_scale.
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += THREADS)
        s_q[i] = (float)q_base[i] * kv_scale;
    __syncthreads();

    // Lane dim assignments
    const int qk_off = lane_id * DIMS_PER_LANE;   // 0,9,...,567
    const int v_off  = lane_id * V_PER_LANE;       // 0,8,...,504

    // Global KV base for this split
    const long long kv_split_base = (long long)(kv_start + sp_st) * QK_DIM;

    // Online softmax state for ALL 16 heads
    float m_h[NUM_HEADS], l_h[NUM_HEADS];
    float ov[NUM_HEADS][V_PER_LANE];
    #pragma unroll
    for (int h = 0; h < NUM_HEADS; h++) {
        m_h[h] = -1e30f;
        l_h[h] = 0.0f;
        #pragma unroll
        for (int v = 0; v < V_PER_LANE; v++) ov[h][v] = 0.0f;
    }

    // ================================================================
    // KV loop: wave round-robin, single pass over all heads per token
    // ================================================================
    for (int t = wave_id; t < sp_len; t += NUM_WAVES) {

        // -- Coalesced KV load: all 64 lanes in this wave load 576 bytes --
        // Each wave has 64 lanes. 576 bytes = 36 x uint4 (16 bytes each).
        // Assign: lane 0-35 each load one uint4 (16 bytes). Lanes 36-63 idle.
        // This gives perfectly coalesced 16-byte loads across lanes 0-35.
        // After load, data is in s_kv[wave_id * QK_DIM ... + 575].
        {
            const long long kv_off = kv_split_base + (long long)t * QK_DIM;
            const unsigned char* kv_ptr = KV + kv_off;
            unsigned char* dst = s_kv + wave_id * QK_DIM;

            if (lane_id < KV_UINT4_LOADS) {
                // Load 16 bytes (uint4) at offset lane_id * 16
                // uint4 = {x, y, z, w} each uint32
                const uint4* src4 = (const uint4*)(kv_ptr + lane_id * 16);
                uint4 val = *src4;
                uint4* dst4 = (uint4*)(dst + lane_id * 16);
                *dst4 = val;
            }
        }
        // No sync needed within a wave -- wave executes in lockstep (SIMD).
        // All 64 lanes see the LDS writes from lanes 0-35 after the store completes.
        // Actually, LDS stores from one lane ARE visible to other lanes in the same wave
        // only after a memory fence or barrier. Use __threadfence_block for correctness.
        __threadfence_block();

        // Read KV data for dot product from LDS (strided access, but LDS has no coalescing issues)
        const unsigned char* kv_row = s_kv + wave_id * QK_DIM;

        float kd[DIMS_PER_LANE];
        #pragma unroll
        for (int d = 0; d < DIMS_PER_LANE; d++)
            kd[d] = fp8_lut[kv_row[qk_off + d]];

        // Read V data from LDS (also strided, fine for LDS)
        // Vectorized LDS read: v_off is multiple of 8, read 2x uint32 = 8 bytes
        float vv[V_PER_LANE];
        {
            const unsigned char* vp = kv_row + v_off;
            unsigned int vw0 = ((const unsigned int*)vp)[0];
            unsigned int vw1 = ((const unsigned int*)vp)[1];
            vv[0] = fp8_lut[(vw0      ) & 0xFF];
            vv[1] = fp8_lut[(vw0 >>  8) & 0xFF];
            vv[2] = fp8_lut[(vw0 >> 16) & 0xFF];
            vv[3] = fp8_lut[(vw0 >> 24)        ];
            vv[4] = fp8_lut[(vw1      ) & 0xFF];
            vv[5] = fp8_lut[(vw1 >>  8) & 0xFF];
            vv[6] = fp8_lut[(vw1 >> 16) & 0xFF];
            vv[7] = fp8_lut[(vw1 >> 24)        ];
        }

        // Process ALL 16 heads for this KV token
        // Q is loaded from LDS per-head (cheap LDS reads vs expensive global KV re-reads)
        #pragma unroll
        for (int h = 0; h < NUM_HEADS; h++) {
            // Load Q for this head from LDS (pre-scaled by kv_scale)
            const float* q_head = s_q + h * QK_DIM + qk_off;
            float dot = q_head[0] * kd[0]
                      + q_head[1] * kd[1]
                      + q_head[2] * kd[2]
                      + q_head[3] * kd[3]
                      + q_head[4] * kd[4]
                      + q_head[5] * kd[5]
                      + q_head[6] * kd[6]
                      + q_head[7] * kd[7]
                      + q_head[8] * kd[8];

            float score = wave_reduce_sum(dot) * sm_scale;

            float old_m = m_h[h];
            float new_m = fmaxf(old_m, score);
            float e_old = __expf(old_m - new_m);
            float e_new = __expf(score - new_m);

            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++)
                ov[h][v] = ov[h][v] * e_old + e_new * vv[v];
            l_h[h] = l_h[h] * e_old + e_new;
            m_h[h] = new_m;
        }
    } // end KV loop

    // ================================================================
    // Cross-wave merge for all 16 heads
    // ================================================================
    __syncthreads();

    for (int h = 0; h < NUM_HEADS; h++) {
        if (lane_id == 0) {
            s_wml[wave_id * 2    ] = m_h[h];
            s_wml[wave_id * 2 + 1] = l_h[h];
        }
        #pragma unroll
        for (int v = 0; v < V_PER_LANE; v++)
            s_wv[wave_id * V_DIM + v_off + v] = ov[h][v];

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
                    mg_o[v] = mg_o[v] * so + sw * s_wv[w * V_DIM + v_off + v];
                mg_l = mg_l * so + sw * wl;
                mg_m = nm;
            }
        }

        // Post-scale V output by kv_scale (since we didn't scale KV values for V accum)
        float inv = (mg_l > 0.0f) ? (kv_scale / mg_l) : 0.0f;
        #pragma unroll
        for (int v = 0; v < V_PER_LANE; v++)
            partial_o[o_base + h * V_DIM + v_off + v] = mg_o[v] * inv;

        if (tid == 0)
            s_lse[h] = (mg_l > 0.0f) ? (mg_m + __logf(mg_l)) : -1e30f;

        __syncthreads();
    } // end head loop

    // write LSE
    if (tid < NUM_HEADS)
        partial_lse[lse_base + tid] = s_lse[tid];
}


// ============================================================================
// Stage 2 -- reduce across splits (unchanged from v6)
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

    // Dynamic shared memory size for stage1
    // s_q: NUM_HEADS * QK_DIM * 4 = 36864
    // fp8_lut: 256 * 4 = 1024
    // s_kv: NUM_WAVES * QK_DIM = 2304
    // s_wml: NUM_WAVES * 2 * 4 = 32
    // s_wv: NUM_WAVES * V_DIM * 4 = 8192
    // s_lse: NUM_HEADS * 4 = 64
    // Total: 48480 bytes
    const unsigned int smem_size = NUM_HEADS * QK_DIM * sizeof(float)  // s_q
                                 + 256 * sizeof(float)                  // fp8_lut
                                 + NUM_WAVES * QK_DIM                   // s_kv (bytes)
                                 + NUM_WAVES * 2 * sizeof(float)        // s_wml
                                 + NUM_WAVES * V_DIM * sizeof(float)    // s_wv
                                 + NUM_HEADS * sizeof(float);           // s_lse

    dim3 g1(batch_size, num_splits_val);
    dim3 b1(THREADS);
    hipLaunchKernelGGL(mla_stage1, g1, b1, smem_size, 0,
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
            name="mla_flash_decode_v7",
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

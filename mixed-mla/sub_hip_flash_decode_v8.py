#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
HIP Flash-Decoding MLA v8 -- Coalesced KV loading via LDS for AMD MI355X (gfx950).

Changes vs v6 (wave-parallel dot, __shfl_xor reduction):
  - Coalesced KV loading: ALL 256 threads cooperatively load KV tokens into LDS
    using 4-byte (uint32) aligned loads. Then each thread reads from LDS.
  - 4 KV tokens staged into LDS per batch to amortize load cost.
  - Pre-multiply kv_scale into Q registers (free, avoids per-element KV scaling).
  - Keep v6's 4 head groups (4 waves, ~93 VGPRs) to avoid register spill.
  - fp8 LUT lookup still in LDS; KV bytes read from LDS staging area.

Shared memory budget (~40.7 KB, fits in 48 KB):
  - s_q:      16 * 576 * 4 = 36864 bytes (Q for all heads, pre-scaled)
  - s_kv:     4 * 576      =  2304 bytes (4 KV tokens staging, uint8)
  - fp8_lut:  256 * 4      =  1024 bytes
  - s_wml:    4 * 2 * 4    =    32 bytes  (wave merge m/l)
  - s_wv:     4 * 512 * 4  =  8192 bytes  (wave merge V -- reused with s_kv area? no, keep separate but overlay with s_q after done)
  Actually s_wv overlaps s_kv in time (s_kv only used during KV loop, s_wv only during merge).
  But simpler to keep separate. Total = 36864 + 2304 + 1024 + 32 + 8192 + 64 = 48480. Tight!

  OPTIMIZATION: s_wv and s_kv don't overlap in time -- s_kv is used during KV loop,
  s_wv during merge after KV loop. Use a union. Also s_lse is tiny.
  With union: 36864 + max(2304, 8192+32) + 1024 + 64 = 36864 + 8224 + 1024 + 64 = 46176 bytes. Fits!
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
// Process heads in groups of 4 to limit register pressure
#define HEAD_GROUP    4
#define NUM_HGROUPS   (NUM_HEADS / HEAD_GROUP)  // 16/4 = 4

// Coalesced KV loading: stage 4 KV tokens into LDS per batch
#define KV_BATCH      4
// 576 bytes per token, 144 uint32 loads to cover it. 256 threads available.
// Each token: ceil(576/4) = 144 uint32 loads. 4 tokens = 576 uint32 loads.
// 256 threads: each thread loads 576/256 = 2.25 -> 3 loads max per batch.
// Actually: 4 * 144 = 576 loads needed, 256 threads -> ~2.25 loads/thread.
// Use: total_words = 4 * 144 = 576; each thread loads words tid, tid+256.
#define WORDS_PER_TOKEN 144   // ceil(576 / 4) = 144
#define BYTES_PER_TOKEN 576

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
// Stage 1 -- flash-decoding partial attention with coalesced KV loading
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
    // s_q: Q data pre-scaled by kv_scale (persistent across all head groups)
    // fp8_lut: fp8 -> float LUT (persistent)
    // s_kv_area: staging for KV tokens (used during KV loop)
    // s_merge_area: wave merge workspace (used after KV loop per head group)
    //   -- s_kv_area and s_merge_area can share memory (union)
    // s_lse: final LSE for each head

    __shared__ float s_q[NUM_HEADS * QK_DIM];       // 36864 bytes
    __shared__ float fp8_lut[256];                   //  1024 bytes
    __shared__ float s_lse[NUM_HEADS];               //    64 bytes
    // Union: KV staging vs merge workspace (never used simultaneously)
    // s_kv_area: KV_BATCH * BYTES_PER_TOKEN = 4 * 576 = 2304 bytes
    // s_merge: NUM_WAVES * 2 floats (m/l) + NUM_WAVES * V_DIM floats
    //        = 8 * 4 + 4 * 512 * 4 = 32 + 8192 = 8224 bytes
    // Union size = max(2304, 8224) = 8224 bytes = 2056 floats
    __shared__ union {
        unsigned char kv_staging[KV_BATCH * BYTES_PER_TOKEN];  // 2304 bytes
        struct {
            float wml[NUM_WAVES * 2];       // 32 bytes
            float wv[NUM_WAVES * V_DIM];    // 8192 bytes
        } merge;
    } s_union;
    // Total: 36864 + 1024 + 64 + 8224 = 46176 bytes = ~45.1 KB. Fits!

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

    // Load all Q into LDS, PRE-MULTIPLIED by kv_scale
    // This eliminates per-element kv_scale multiply on KV side
    for (int i = tid; i < NUM_HEADS * QK_DIM; i += THREADS)
        s_q[i] = (float)q_base[i] * kv_scale;
    __syncthreads();

    // Lane dim assignments
    const int qk_base = lane_id * DIMS_PER_LANE;   // 0,9,...,567
    const int v_base  = lane_id * V_PER_LANE;       // 0,8,...,504

    // Global KV base for this split
    const long long kv_split_base = (long long)(kv_start + sp_st) * QK_DIM;

    // ================================================================
    // Process heads in groups of HEAD_GROUP (4), making 4 passes over KV
    // Each pass: coalesced KV load into LDS, then dot+softmax+V accum
    // ================================================================
    for (int hg = 0; hg < NUM_HGROUPS; hg++) {
        const int h_off = hg * HEAD_GROUP;

        // Load Q for these 4 heads into registers (from LDS, already scaled)
        float qr[HEAD_GROUP][DIMS_PER_LANE];
        #pragma unroll
        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            #pragma unroll
            for (int d = 0; d < DIMS_PER_LANE; d++)
                qr[hh][d] = s_q[(h_off + hh) * QK_DIM + qk_base + d];
        }

        // Online softmax state
        float m_h[HEAD_GROUP], l_h[HEAD_GROUP];
        float ov[HEAD_GROUP][V_PER_LANE];
        #pragma unroll
        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            m_h[hh] = -1e30f;
            l_h[hh] = 0.0f;
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++) ov[hh][v] = 0.0f;
        }

        // KV loop with coalesced loading in batches of KV_BATCH
        // Waves no longer round-robin -- ALL threads participate in loading,
        // then each wave processes one of the KV_BATCH tokens.
        for (int tb = 0; tb < sp_len; tb += KV_BATCH) {
            // How many valid tokens in this batch
            int valid = sp_len - tb;
            if (valid > KV_BATCH) valid = KV_BATCH;

            // -- Coalesced load: all 256 threads load KV tokens into s_union.kv_staging --
            // Total bytes to load: valid * 576
            // Load as uint32 (4 bytes at a time) for coalescing.
            // Total uint32 loads: valid * 144
            // 256 threads handle these in rounds.
            {
                const int total_words = valid * WORDS_PER_TOKEN;
                // Each thread loads words at tid, tid+256, tid+512, ...
                const unsigned char* kv_batch_base = KV + kv_split_base + (long long)tb * QK_DIM;
                for (int w = tid; w < total_words; w += THREADS) {
                    // w maps to token index and word offset within token
                    int tok_idx   = w / WORDS_PER_TOKEN;
                    int word_off  = w % WORDS_PER_TOKEN;
                    int byte_off  = word_off * 4;

                    // Source: global memory (coalesced since consecutive threads read consecutive words)
                    // Handle the last word specially: token is 576 bytes = 144 words exactly (576/4=144)
                    // So no partial-word issue.
                    unsigned int val = ((const unsigned int*)(kv_batch_base + (long long)tok_idx * QK_DIM))[word_off];

                    // Dest: LDS staging
                    ((unsigned int*)(s_union.kv_staging + tok_idx * BYTES_PER_TOKEN))[word_off] = val;
                }
            }
            __syncthreads();

            // -- Process: each wave takes one token from the batch --
            // wave_id 0..3 processes token 0..3 (if valid)
            if (wave_id < valid) {
                const unsigned char* kv_lds = s_union.kv_staging + wave_id * BYTES_PER_TOKEN;

                // Load 9 QK dims for dot product from LDS (no kv_scale -- already in Q)
                float kd[DIMS_PER_LANE];
                #pragma unroll
                for (int d = 0; d < DIMS_PER_LANE; d++)
                    kd[d] = fp8_lut[kv_lds[qk_base + d]];

                // Load 8 V dims from LDS (vectorized via uint32 reads from LDS)
                float vv[V_PER_LANE];
                {
                    const unsigned char* vp = kv_lds + v_base;
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

                // Dot product + softmax + V accum for each head
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
            }

            __syncthreads();  // ensure s_union.kv_staging can be overwritten next iteration
        } // end KV batch loop

        // ---- cross-wave merge for this head group ----
        // Now s_union is reused for merge workspace
        __syncthreads();

        for (int hh = 0; hh < HEAD_GROUP; hh++) {
            const int h = h_off + hh;

            if (lane_id == 0) {
                s_union.merge.wml[wave_id * 2    ] = m_h[hh];
                s_union.merge.wml[wave_id * 2 + 1] = l_h[hh];
            }
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++)
                s_union.merge.wv[wave_id * V_DIM + v_base + v] = ov[hh][v];

            __syncthreads();

            float mg_m = -1e30f, mg_l = 0.0f;
            float mg_o[V_PER_LANE];
            #pragma unroll
            for (int v = 0; v < V_PER_LANE; v++) mg_o[v] = 0.0f;

            #pragma unroll
            for (int w = 0; w < NUM_WAVES; w++) {
                float wm = s_union.merge.wml[w * 2    ];
                float wl = s_union.merge.wml[w * 2 + 1];
                if (wl > 0.0f) {
                    float nm = fmaxf(mg_m, wm);
                    float so = __expf(mg_m - nm);
                    float sw = __expf(wm   - nm);
                    #pragma unroll
                    for (int v = 0; v < V_PER_LANE; v++)
                        mg_o[v] = mg_o[v] * so + sw * s_union.merge.wv[w * V_DIM + v_base + v];
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
// Stage 2 -- reduce across splits (identical to v6)
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
            name="mla_flash_decode_v8",
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

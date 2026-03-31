#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP GEMM v5 DIAGNOSTIC for MXFP4 on MI355X (gfx950).
Uses 16x16x128 MFMA FP4 with fused bf16->fp4 A quantization.

DIAGNOSTIC MODES to isolate the 13% error seen in v4:
  Mode 0: REFERENCE  — use Triton gemm_a16wfp4 (known correct)
  Mode 1: SCALES=127 — all scales forced to 127 (E8M0 bias = 2^0 = 1.0)
           If correct: data loading is right, bug is in scale packing
           If wrong: data loading or MFMA operand layout is wrong
  Mode 2: SINGLE K   — process ONLY k_iter=0 (first 128 K-elements)
           Compare HIP vs reference partial sums to find where divergence starts
  Mode 3: FULL DIAG  — full K loop with per-element comparison and debug prints
  Mode 4: ALL_127    — both A and B scales forced to 127, full K loop
           Tests if accumulation across K-chunks is correct with neutral scales
  Mode 5: SWAP_HALF  — test if upper/lower 16 bytes of register should be swapped
           Put data in bytes [16..31] instead of [0..15]

Thread decomposition for 16x16x128 MFMA (from CK WarpGemmAttributeMfmaImpl):
    lr = lane & 15   -- M/N index (0..15)
    lg = lane >> 4   -- K-group index (0..3)
    kABKPerLane = 32 -- each thread handles 32 FP4 K-elements
    K-group lg covers K-positions [lg*32 .. lg*32+31]

Output store: col = lr, row = lg*4 + i for i=0..3
    c_reg[i] -> C[lg*4 + i, lr]
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP kernel source with multiple diagnostic modes
# ---------------------------------------------------------------------------
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

// ---- types for MFMA 16x16x128 ----
typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(4))) float4v;

// ---- tile sizes ----
#define TM 32
#define TN 32
#define TK 128       // K per MFMA iteration
#define NTHREADS 256
#define WARP_SZ 64

// LDS layout per K iteration:
#define LDS_A_FP4   (TM * (TK / 2))        // 2048
#define LDS_A_SCALE (TM * (TK / 32))       // 128
#define LDS_B_FP4   (TN * (TK / 2))        // 2048
#define LDS_TOTAL   (LDS_A_FP4 + LDS_A_SCALE + LDS_B_FP4)  // 4224

// Offsets
#define OFF_A_FP4    0
#define OFF_A_SCALE  LDS_A_FP4
#define OFF_B_FP4   (LDS_A_FP4 + LDS_A_SCALE)

// Debug output buffer layout:
// [0]    = mode flag
// [1..4] = acc[0..3] for thread 0, warp 0 after first K iteration
// [5..8] = sa, sb, a_op[0], b_op[0] for thread 0
// [9..12] = acc[0..3] after FINAL K iteration
// [13]   = number of K iterations executed
// [14..17] = A scale bytes (4 values) for thread 0
// [18..21] = B scale bytes (4 values) for thread 0
// [22..37] = A fp4 bytes (16 values) for thread 0
// [38..53] = B fp4 bytes (16 values) for thread 0
#define DBG_SIZE 256

// ---- bf16 helpers ----
__device__ __forceinline__ float bf16_to_f32(unsigned short x) {
    union { unsigned int u; float f; } v;
    v.u = (unsigned int)x << 16;
    return v.f;
}

__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    union { unsigned int u; float f; } v;
    v.f = f;
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (unsigned short)(v.u >> 16);
}

// ---- Fused bf16 -> FP4 E2M1 quantization ----
// Matches aiter _mxfp4_quant_op: round-up amax, RNE fp4 conversion.
__device__ void quant_bf16x32_to_fp4(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ dst_fp4,
    unsigned char& scale_out,
    int valid)
{
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        if (i < valid) {
            vals[i] = bf16_to_f32(src[i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        } else {
            vals[i] = 0.0f;
        }
    }

    // E8M0 scale: round amax up to power of 2, then log2 - 2
    union { float f; unsigned int u; } au;
    au.f = amax;
    unsigned int ai = au.u;
    ai = (ai + 0x200000u) & 0xFF800000u;
    int biased_exp = (int)((ai >> 23) & 0xFF);
    int su = biased_exp - 127 - 2;
    if (su < -127) su = -127;
    if (su > 127) su = 127;
    scale_out = (unsigned char)(su + 127);

    // Inverse scale for quantization
    float inv_scale;
    {
        int qs = (-su) + 127;
        if (qs < 1) qs = 1;
        if (qs > 254) qs = 254;
        union { unsigned int u; float f; } sv;
        sv.u = (unsigned int)qs << 23;
        inv_scale = sv.f;
    }

    // Quantize to E2M1 with RNE
    for (int i = 0; i < 16; i++) {
        unsigned char nib[2];
        for (int j = 0; j < 2; j++) {
            float qx = vals[2 * i + j] * inv_scale;
            union { float f; unsigned int u; } qu;
            qu.f = qx;
            unsigned int qx_u = qu.u;
            unsigned int sign = qx_u & 0x80000000u;
            qx_u ^= sign;
            qu.u = qx_u;
            float qx_abs = qu.f;

            unsigned char e2m1;
            if (qx_abs >= 6.0f) {
                e2m1 = 0x7;
            } else if (qx_abs < 1.0f) {
                // Denormal
                unsigned int dm = ((127u - 1u) + (23u - 1u) + 1u) << 23;
                union { unsigned int u; float f; } dv;
                dv.u = dm;
                float dn = qx_abs + dv.f;
                union { float f; unsigned int u; } dr;
                dr.f = dn;
                e2m1 = (unsigned char)(dr.u - dm);
            } else {
                // Normal: RNE
                unsigned int mant_odd = (qx_u >> 22) & 1;
                unsigned int val_add = ((1u - 127u) << 23) + (1 << 21) - 1;
                unsigned int nx = qx_u + val_add + mant_odd;
                e2m1 = (unsigned char)(nx >> 22);
            }

            unsigned char sign_bit = (unsigned char)(sign >> 28);
            nib[j] = (e2m1 & 0x7) | sign_bit;
        }
        dst_fp4[i] = nib[0] | (nib[1] << 4);
    }
}


// ========================================================================
// MODE 1: GEMM kernel with ALL SCALES FORCED TO 127
// Tests whether data loading is correct independently of scale packing
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_scales127(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};
    int k_count = 0;

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A into LDS
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            // IGNORE computed scale, force to 127
            lds_a_scale[a_row_local * 4 + a_kblock] = 127;
        }

        // Phase 2: Load B into LDS
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands
        int8v a_op = {};
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
        }

        int8v b_op = {};
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
        }

        // Phase 4: ALL SCALES = 127 (scale 1.0)
        unsigned int sa = 0x7F7F7F7Fu;
        unsigned int sb = 0x7F7F7F7Fu;

        // Phase 5: MFMA
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);

        k_count++;

        // Debug: dump after first K iteration for block(0,0), warp0, thread0
        if (bm == 0 && bn == 0 && tid == 0 && k_iter == 0 && dbg != nullptr) {
            dbg[0] = 1.0f;  // mode=scales127
            dbg[1] = acc[0];
            dbg[2] = acc[1];
            dbg[3] = acc[2];
            dbg[4] = acc[3];
            // dump a_op and b_op
            union { int i; float f; } conv;
            conv.i = a_op[0]; dbg[5] = conv.f;
            conv.i = b_op[0]; dbg[6] = conv.f;
            // dump fp4 bytes for this thread
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const unsigned char* abytes = lds_a_fp4 + a_row * (TK / 2) + a_byte_off;
            for (int x = 0; x < 16; x++) dbg[22 + x] = (float)abytes[x];
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const unsigned char* bbytes = lds_b_fp4 + b_row * (TK / 2) + b_byte_off;
            for (int x = 0; x < 16; x++) dbg[38 + x] = (float)bbytes[x];
        }
    }

    // Debug: dump final accumulator
    if (bm == 0 && bn == 0 && tid == 0 && dbg != nullptr) {
        dbg[9] = acc[0];
        dbg[10] = acc[1];
        dbg[11] = acc[2];
        dbg[12] = acc[3];
        dbg[13] = (float)k_count;
    }

    // Store output
    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


// ========================================================================
// MODE 3: FULL DIAGNOSTIC kernel with real scales + debug output
// Tests the real scale packing logic
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_diag(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};
    int k_count = 0;

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A into LDS
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        // Phase 2: Load B into LDS
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands
        int8v a_op = {};
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
        }

        int8v b_op = {};
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
        }

        // Phase 4: Pack scales
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            const unsigned char* sp = lds_a_scale + a_row * 4;
            sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                 ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
        }

        unsigned int sb;
        {
            int b_row_global = n_base + wn * 16 + lr;
            int kb = k_iter / 32;
            if (b_row_global < N) {
                const unsigned char* sp = B_sc + (long long)b_row_global * Kblocks + kb;
                unsigned char s0 = sp[0];
                unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
                unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
                unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
                sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                     ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
            } else {
                sb = 0x7F7F7F7Fu;
            }
        }

        // Debug: dump after first K for block(0,0), warp0, thread0
        if (bm == 0 && bn == 0 && tid == 0 && k_iter == 0 && dbg != nullptr) {
            dbg[0] = 3.0f;  // mode=diag
            // Scale debug
            union { unsigned int u; float f; } conv;
            conv.u = sa; dbg[5] = conv.f;
            conv.u = sb; dbg[6] = conv.f;
            // Individual scale bytes
            dbg[14] = (float)(sa & 0xFF);
            dbg[15] = (float)((sa >> 8) & 0xFF);
            dbg[16] = (float)((sa >> 16) & 0xFF);
            dbg[17] = (float)((sa >> 24) & 0xFF);
            dbg[18] = (float)(sb & 0xFF);
            dbg[19] = (float)((sb >> 8) & 0xFF);
            dbg[20] = (float)((sb >> 16) & 0xFF);
            dbg[21] = (float)((sb >> 24) & 0xFF);
            // fp4 data bytes
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const unsigned char* abytes = lds_a_fp4 + a_row * (TK / 2) + a_byte_off;
            for (int x = 0; x < 16; x++) dbg[22 + x] = (float)abytes[x];
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const unsigned char* bbytes = lds_b_fp4 + b_row * (TK / 2) + b_byte_off;
            for (int x = 0; x < 16; x++) dbg[38 + x] = (float)bbytes[x];
        }

        // Phase 5: MFMA
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);

        // Debug: accumulator after first K
        if (bm == 0 && bn == 0 && tid == 0 && k_iter == 0 && dbg != nullptr) {
            dbg[1] = acc[0];
            dbg[2] = acc[1];
            dbg[3] = acc[2];
            dbg[4] = acc[3];
        }

        k_count++;
    }

    // Debug: dump final accumulator
    if (bm == 0 && bn == 0 && tid == 0 && dbg != nullptr) {
        dbg[9] = acc[0];
        dbg[10] = acc[1];
        dbg[11] = acc[2];
        dbg[12] = acc[3];
        dbg[13] = (float)k_count;
    }

    // Store output
    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


// ========================================================================
// MODE 5: SWAP HALF — data in upper 16 bytes [16..31] of register
// Tests the hypothesis that kABKPerLane=32 means upper half, not lower
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_swap_half(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A into LDS
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        // Phase 2: Load B into LDS
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands — DATA IN UPPER HALF (bytes 16..31)
        int8v a_op = {};
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            // Put in UPPER half of register
            a_op[4] = src_ptr[0];
            a_op[5] = src_ptr[1];
            a_op[6] = src_ptr[2];
            a_op[7] = src_ptr[3];
            // Lower half stays 0
        }

        int8v b_op = {};
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[4] = src_ptr[0];
            b_op[5] = src_ptr[1];
            b_op[6] = src_ptr[2];
            b_op[7] = src_ptr[3];
        }

        // Phase 4: Pack scales (same as normal)
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            const unsigned char* sp = lds_a_scale + a_row * 4;
            sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                 ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
        }

        unsigned int sb;
        {
            int b_row_global = n_base + wn * 16 + lr;
            int kb = k_iter / 32;
            if (b_row_global < N) {
                const unsigned char* sp = B_sc + (long long)b_row_global * Kblocks + kb;
                unsigned char s0 = sp[0];
                unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
                unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
                unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
                sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                     ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
            } else {
                sb = 0x7F7F7F7Fu;
            }
        }

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


// ========================================================================
// MODE 6: FULL 32 BYTES — load all 64 FP4 like v3's (lg&1)*32 pattern
// Each thread loads 32 bytes = 64 FP4 values = 256 bits
// lg=0,2 load K[0..63], lg=1,3 load K[64..127]
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_full32(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A into LDS
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        // Phase 2: Load B into LDS
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands — FULL 32 bytes (v3 pattern)
        // lg=0,2 -> bytes[0..31], lg=1,3 -> bytes[32..63]
        int8v a_op;
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = (lg & 1) * 32;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
            a_op[4] = src_ptr[4];
            a_op[5] = src_ptr[5];
            a_op[6] = src_ptr[6];
            a_op[7] = src_ptr[7];
        }

        int8v b_op;
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = (lg & 1) * 32;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
            b_op[4] = src_ptr[4];
            b_op[5] = src_ptr[5];
            b_op[6] = src_ptr[6];
            b_op[7] = src_ptr[7];
        }

        // Phase 4: Pack scales
        // For full-32-byte mode, lg=0,2 cover K-blocks 0,1 and lg=1,3 cover K-blocks 2,3
        // So scales should be packed accordingly
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            const unsigned char* sp = lds_a_scale + a_row * 4;
            sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                 ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
        }

        unsigned int sb;
        {
            int b_row_global = n_base + wn * 16 + lr;
            int kb = k_iter / 32;
            if (b_row_global < N) {
                const unsigned char* sp = B_sc + (long long)b_row_global * Kblocks + kb;
                unsigned char s0 = sp[0];
                unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
                unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
                unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
                sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                     ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
            } else {
                sb = 0x7F7F7F7Fu;
            }
        }

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


// ========================================================================
// MODE 7: REMOVED — int4v causes compile error (MFMA needs int8v)
// MODE 7: CK-STYLE — DISABLED
#if 0 // DISABLED: int4v causes compile error
// The simple version used int4v = ext_vector_type(4) = 128 bits
// This is the SIMPLEST possible operand type — matches the simple kernel
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_int4v(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    typedef int __attribute__((ext_vector_type(4))) int4v;

    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A into LDS
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        // Phase 2: Load B into LDS
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands — int4v (128 bits = 16 bytes = 32 FP4)
        int4v a_op;
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            a_op = *(const int4v*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
        }

        int4v b_op;
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            b_op = *(const int4v*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
        }

        // Phase 4: Pack scales
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            const unsigned char* sp = lds_a_scale + a_row * 4;
            sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                 ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
        }

        unsigned int sb;
        {
            int b_row_global = n_base + wn * 16 + lr;
            int kb = k_iter / 32;
            if (b_row_global < N) {
                const unsigned char* sp = B_sc + (long long)b_row_global * Kblocks + kb;
                unsigned char s0 = sp[0];
                unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
                unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
                unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
                sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                     ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
            } else {
                sb = 0x7F7F7F7Fu;
            }
        }

        // MFMA with int4v — compiler may promote to correct register type
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


#endif // DISABLED int4v mode

// ========================================================================
// MODE 8: SINGLE SCALE per thread — test if hardware reads only byte[lg]
// Instead of packing all 4 scales, put ONLY this thread's scale in byte[0]
// and set all other bytes to 0. If hardware only reads byte[lg], this
// should only work for lg=0.
// ========================================================================
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_single_scale(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K,
    float* __restrict__ dbg)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;
    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }
            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        int8v a_op = {};
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
        }

        int8v b_op = {};
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
        }

        // SINGLE SCALE: only this thread's K-group scale, in its byte position
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            unsigned char my_scale = lds_a_scale[a_row * 4 + lg];
            // Put it in byte[lg] position, all others = 0
            sa = (unsigned int)my_scale << (lg * 8);
        }

        unsigned int sb;
        {
            int b_row_global = n_base + wn * 16 + lr;
            int kb = k_iter / 32 + lg;
            unsigned char my_scale = 0;
            if (b_row_global < N && kb < Kblocks) {
                my_scale = B_sc[(long long)b_row_global * Kblocks + kb];
            }
            sb = (unsigned int)my_scale << (lg * 8);
        }

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}


// ========================================================================
// Launcher functions
// ========================================================================
torch::Tensor launch_scales127(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_scales127,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}

torch::Tensor launch_diag(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_diag,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}

torch::Tensor launch_swap_half(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_swap_half,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}

torch::Tensor launch_full32(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_full32,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}

torch::Tensor launch_int4v(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_int4v,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}

torch::Tensor launch_single_scale(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, torch::Tensor dbg)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(hip_mxfp4_gemm_single_scale,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K,
        (float*)dbg.data_ptr());
    return C;
}
"""

CPP_FORWARD = """
torch::Tensor launch_scales127(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);
torch::Tensor launch_diag(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);
torch::Tensor launch_swap_half(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);
torch::Tensor launch_full32(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);
// torch::Tensor launch_int4v(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);  // DISABLED
torch::Tensor launch_single_scale(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, torch::Tensor);
"""

# ---------------------------------------------------------------------------
# Module build
# ---------------------------------------------------------------------------
_module = None
_build_failed = False

def _get_module():
    global _module, _build_failed
    if _build_failed:
        return None
    if _module is not None:
        return _module
    try:
        from torch.utils.cpp_extension import load_inline
        _module = load_inline(
            name="hip_gemm_v5_debug",
            cpp_sources=CPP_FORWARD,
            cuda_sources=HIP_SOURCE,
            functions=[
                "launch_scales127",
                "launch_diag",
                "launch_swap_half",
                "launch_full32",
                # "launch_int4v",  # DISABLED
                "launch_single_scale",
            ],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
        return _module
    except Exception as e:
        print(f"[HIP BUILD FAILED] {e}")
        _build_failed = True
        return None

# ---------------------------------------------------------------------------
# Unshuffle E8M0 (shuffled -> raw layout)
# ---------------------------------------------------------------------------
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_b_cache_key = None
_b_q_u8 = None
_b_sc_raw = None
_y_cache = {}
_diag_done = False

_K7168_CFG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

_K2048_CFG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
    "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
}


def _compute_reference(A, B_q_u8, B_sc_raw, m, n, k):
    """Compute reference result using proven Triton path."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, B_q_u8, B_sc_raw, dtype=torch.bfloat16)


def _compare(name, result, ref, rows=4, cols=8):
    """Compare two tensors and print element-wise differences."""
    h = result[:min(result.shape[0], rows), :min(result.shape[1], cols)].float().cpu()
    r = ref[:min(ref.shape[0], rows), :min(ref.shape[1], cols)].float().cpu()
    max_err = 0.0
    total_err = 0.0
    count = 0
    for i in range(h.shape[0]):
        hip_vals = " ".join(f"{h[i,j]:8.1f}" for j in range(h.shape[1]))
        ref_vals = " ".join(f"{r[i,j]:8.1f}" for j in range(h.shape[1]))
        err_vals = []
        for j in range(h.shape[1]):
            if abs(r[i,j]) > 0.1:
                e = abs(h[i,j] - r[i,j]) / abs(r[i,j]) * 100
            else:
                e = abs(h[i,j] - r[i,j]) * 100
            err_vals.append(f"{e:5.1f}%")
            max_err = max(max_err, e)
            total_err += e
            count += 1
        err_str = " ".join(err_vals)
        print(f"  [{name}] row{i} HIP: {hip_vals}")
        print(f"  [{name}] row{i} REF: {ref_vals}")
        print(f"  [{name}] row{i} ERR: {err_str}")
    if count > 0:
        print(f"  [{name}] max_err={max_err:.1f}% avg_err={total_err/count:.1f}%")
    # Also compute full-tensor error
    full_h = result.float().cpu()
    full_r = ref.float().cpu()
    abs_diff = (full_h - full_r).abs()
    rel_diff = abs_diff / (full_r.abs().clamp(min=0.1))
    print(f"  [{name}] FULL: max_abs={abs_diff.max():.2f} mean_abs={abs_diff.mean():.2f} "
          f"max_rel={rel_diff.max()*100:.1f}% mean_rel={rel_diff.mean()*100:.1f}%")
    return max_err


def _print_debug_buffer(dbg, name):
    """Print the debug output buffer."""
    d = dbg.cpu().numpy()
    print(f"  [{name}] mode={d[0]:.0f}")
    print(f"  [{name}] acc_after_first_K: [{d[1]:.4f}, {d[2]:.4f}, {d[3]:.4f}, {d[4]:.4f}]")
    print(f"  [{name}] acc_final:         [{d[9]:.4f}, {d[10]:.4f}, {d[11]:.4f}, {d[12]:.4f}]")
    print(f"  [{name}] K_iterations={d[13]:.0f}")
    print(f"  [{name}] A_scale_bytes: [{d[14]:.0f}, {d[15]:.0f}, {d[16]:.0f}, {d[17]:.0f}]")
    print(f"  [{name}] B_scale_bytes: [{d[18]:.0f}, {d[19]:.0f}, {d[20]:.0f}, {d[21]:.0f}]")
    a_fp4 = [f"{d[22+x]:.0f}" for x in range(16)]
    b_fp4 = [f"{d[38+x]:.0f}" for x in range(16)]
    print(f"  [{name}] A_fp4_bytes: [{', '.join(a_fp4)}]")
    print(f"  [{name}] B_fp4_bytes: [{', '.join(b_fp4)}]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_q_u8, _b_sc_raw, _diag_done

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B preprocessing
    bkey = B_scale_sh.data_ptr()
    if bkey != _b_cache_key:
        _b_cache_key = bkey
        _b_q_u8 = B_q.view(torch.uint8)
        _b_sc_raw = _unshuffle_e8m0(B_scale_sh)

    mod = _get_module()

    # Run DIAGNOSTIC on first call only, then fall back to Triton for correctness
    if mod is not None and not _diag_done:
        _diag_done = True
        b_sc_slice = _b_sc_raw[:n, :k // 32].contiguous()
        dbg = torch.zeros(256, dtype=torch.float32, device='cuda')

        # Compute reference
        ref = _compute_reference(A, _b_q_u8, _b_sc_raw, m, n, k)

        print(f"\n{'='*80}")
        print(f"  HIP GEMM v5 DIAGNOSTIC: M={m}, N={n}, K={k}")
        print(f"  B_q shape={B_q.shape}, B_sc_raw shape={_b_sc_raw.shape}")
        print(f"  b_sc_slice shape={b_sc_slice.shape}")
        print(f"{'='*80}")

        # MODE 1: scales=127
        print(f"\n--- MODE 1: ALL SCALES = 127 ---")
        dbg.zero_()
        c1 = mod.launch_scales127(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        torch.cuda.synchronize()
        _print_debug_buffer(dbg, "scales127")
        _compare("scales127", c1, ref)

        # MODE 3: real scales with debug
        print(f"\n--- MODE 3: REAL SCALES (DIAG) ---")
        dbg.zero_()
        c3 = mod.launch_diag(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        torch.cuda.synchronize()
        _print_debug_buffer(dbg, "diag")
        _compare("diag", c3, ref)

        # MODE 5: swap half (data in upper 16 bytes)
        print(f"\n--- MODE 5: SWAP HALF (data in upper bytes) ---")
        dbg.zero_()
        c5 = mod.launch_swap_half(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        torch.cuda.synchronize()
        _compare("swap_half", c5, ref)

        # MODE 6: full 32 bytes (v3 pattern)
        print(f"\n--- MODE 6: FULL 32 BYTES (v3 pattern) ---")
        dbg.zero_()
        c6 = mod.launch_full32(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        torch.cuda.synchronize()
        _compare("full32", c6, ref)

        # MODE 7: DISABLED (int4v causes compile error)
        # if False:
        #     print(f"\n--- MODE 7: INT4V operand type ---")
        #     dbg.zero_()
        #     c7 = mod.launch_int4v(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        #     torch.cuda.synchronize()
        #     _compare("int4v", c7, ref)

        # MODE 8: single scale per thread
        print(f"\n--- MODE 8: SINGLE SCALE per thread ---")
        dbg.zero_()
        c8 = mod.launch_single_scale(A, _b_q_u8, b_sc_slice, m, n, k, dbg)
        torch.cuda.synchronize()
        _compare("single_sc", c8, ref)

        # Also compare scales=127 vs diag to see scale impact
        print(f"\n--- SCALE IMPACT: scales127 vs diag ---")
        h1 = c1[:min(m,4), :8].float().cpu()
        h3 = c3[:min(m,4), :8].float().cpu()
        for i in range(min(m,4)):
            v1 = " ".join(f"{h1[i,j]:8.1f}" for j in range(8))
            v3 = " ".join(f"{h3[i,j]:8.1f}" for j in range(8))
            ratio = " ".join(f"{h3[i,j]/h1[i,j]:6.3f}" if abs(h1[i,j])>0.1 else "  inf " for j in range(8))
            print(f"  row{i} sc127: {v1}")
            print(f"  row{i} real:  {v3}")
            print(f"  row{i} ratio: {ratio}")

        print(f"\n{'='*80}")
        print(f"  END DIAGNOSTIC")
        print(f"{'='*80}\n")

    # Always fall back to Triton for correct results
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _b_q_u8, A_scale, _b_sc_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    if k == 7168:
        cfg = _K7168_CFG
    elif k == 2048:
        cfg = _K2048_CFG
    else:
        cfg = None

    return gemm_a16wfp4(A, _b_q_u8, _b_sc_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)

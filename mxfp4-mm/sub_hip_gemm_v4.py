#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP GEMM v4 for MXFP4 on MI355X (gfx950).
Uses 16x16x128 MFMA FP4 with fused bf16->fp4 A quantization.

CORRECTED register mapping based on CK source analysis:
  WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4:
    kAMLane = 16     -- 16 threads span M dimension
    kBNLane = 16     -- 16 threads span N dimension
    kABKLane = 4     -- 4 threads span K dimension
    kABKPerLane = 32 -- 32 K-elements per thread (= 16 fp4x2 bytes)

Thread decomposition for 64 threads:
    lr = lane & 15   -- M/N index (0..15)
    lg = lane >> 4   -- K-group index (0..3)

A operand [16 rows, 128 K]:
    Thread loads 16 fp4x2 bytes (32 FP4 = 32 K-elements) for row lr
    K-range: [lg * 32 .. lg * 32 + 31]
    Stored in FIRST 16 bytes of 32-byte register, upper 16 bytes = 0

B operand [128 K, 16 cols]:
    Thread loads 16 fp4x2 bytes (32 FP4 = 32 K-elements) for column lr
    K-range: [lg * 32 .. lg * 32 + 31]  (same K-group as A)
    Stored in FIRST 16 bytes of 32-byte register, upper 16 bytes = 0

Scales (E8M0, 4 bytes packed into uint32):
    All threads for a row/col provide ALL 4 K-block scales.
    Hardware selects byte[lg] for this thread's K-group.
    sa = scale_a[row, k/32+0] | (scale_a[row, k/32+1] << 8) | ...

Output C [16, 16] = 4 fp32 per thread:
    col = lr (= lane & 15)
    row = lg * 4 + i  (i = 0..3)
    c_reg[i] -> C[lg*4 + i, lr]

Tile: 32M x 32N output via 4 warps of 64 threads (256 total).
  warp_id / 2 = wm (M-half: 0 or 1)
  warp_id % 2 = wn (N-half: 0 or 1)
  Each warp handles one 16x16 MFMA sub-tile.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP kernel source
# ---------------------------------------------------------------------------
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

// ---- types for MFMA 16x16x128 ----
// 256-bit operand = 8 x int32 = 32 bytes = 64 FP4 values
// But only first 16 bytes (32 FP4 = 32 K-elements) are meaningful.
// Upper 16 bytes must be zero-padded.
typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(4))) float4v;

// ---- tile sizes ----
#define TM 32
#define TN 32
#define TK 128       // K per MFMA iteration
#define NTHREADS 256
#define WARP_SZ 64

// LDS layout per K iteration:
//   A fp4: [32 rows, 64 bytes] = 2048 bytes (32 rows x 128 K / 2)
//   A scale: [32 rows, 4 bytes] = 128 bytes (32 rows x 4 scale blocks)
//   B fp4: [32 cols, 64 bytes] = 2048 bytes (32 cols x 128 K / 2)
//   Total: 4224 bytes
#define LDS_A_FP4   (TM * (TK / 2))        // 2048
#define LDS_A_SCALE (TM * (TK / 32))       // 128
#define LDS_B_FP4   (TN * (TK / 2))        // 2048
#define LDS_TOTAL   (LDS_A_FP4 + LDS_A_SCALE + LDS_B_FP4)  // 4224

// Offsets
#define OFF_A_FP4    0
#define OFF_A_SCALE  LDS_A_FP4
#define OFF_B_FP4   (LDS_A_FP4 + LDS_A_SCALE)

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
// Quantizes 32 bf16 values to 16 fp4x2 bytes + 1 E8M0 scale.
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

// ---- Main GEMM kernel ----
// C[M,N] = A[M,K] (bf16) x B_q[N,K/2]^T (fp4x2) with B_sc[N,K/32] E8M0 scales
//
// MFMA 16x16x128 FP4:
//   - 64 threads per wave, 4 waves per block = 256 threads
//   - lr = lane & 15: row index (for A) / col index (for B) within 16x16 tile
//   - lg = lane >> 4: K-group index (0..3), each covering 32 K-elements
//   - Each thread loads 16 fp4x2 bytes (32 K-elements) into lower half of register
//   - Upper 16 bytes of register are zero-padded
//   - Output: 4 fp32 at C[lg*4+i, lr] for i=0..3
//   - Scales: 4 E8M0 bytes packed in uint32, one per K-block of 32
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;          // 0..15: M/N index within 16x16 tile
    const int lg = lane >> 4;          // 0..3:  K-group index

    const int wm = warp_id >> 1;      // 0 or 1: which M-half of 32x32 tile
    const int wn = warp_id & 1;       // 0 or 1: which N-half of 32x32 tile

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    // LDS
    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    // Accumulator: 4 fp32 per thread for 16x16 output
    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // ---- K loop ----
    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // === Phase 1: Load + quantize A tile into LDS ===
        // 32 rows x 4 K-blocks of 32 = 128 quant jobs
        // 256 threads, first 128 handle quant
        if (tid < 128) {
            int a_row_local = tid >> 2;          // 0..31
            int a_kblock    = tid & 3;           // 0..3 (which 32-element K-block)
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

        // === Phase 2: Load B tile (fp4x2) into LDS ===
        // 32 rows x 64 bytes = 2048 bytes. 256 threads x 8 bytes = 2048.
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

        // === Phase 3: Load MFMA operands from LDS ===
        //
        // 16x16x128 MFMA with kABKLane=4, kABKPerLane=32:
        //   lr = lane & 15 -> which row of A (or col of B) within the 16x16 tile
        //   lg = lane >> 4 -> which K-group (0..3), each covering 32 K-elements
        //
        // A: load 16 fp4x2 bytes from row [wm*16 + lr], K-offset = lg * 16 bytes
        //    (lg * 16 bytes = lg * 32 FP4 values = 32 K-elements)
        //    Store in lower half of 32-byte register, zero-pad upper half
        int8v a_op = {};  // zero-initialized (upper half stays 0)
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;  // 16 bytes per K-group of 32 elements
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
            // a_op[4..7] remain 0 (zero-padded upper half)
        }

        // B: load 16 fp4x2 bytes from row [wn*16 + lr], K-offset = lg * 16 bytes
        //    Same K-group as A, same zero-padding
        int8v b_op = {};  // zero-initialized
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
            // b_op[4..7] remain 0
        }

        // === Phase 4: Pack scales into uint32 ===
        // All 4 E8M0 scale bytes for K-blocks 0..3, packed into one uint32.
        // The hardware selects the correct byte based on lg (K-group).
        //
        // A scale: from this thread's row within the tile
        unsigned int sa;
        {
            int a_row = wm * 16 + lr;
            const unsigned char* sp = lds_a_scale + a_row * 4;
            sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                 ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
        }

        // B scale: from global memory for this thread's B column
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

        // === Phase 5: MFMA 16x16x128 FP4 ===
        // cbsz_a=4 (FP4), cbsz_b=4 (FP4), cbsz=0
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    // === Store output ===
    // 16x16x128 output mapping:
    //   col = lr (= lane & 15) -- column within 16x16 tile
    //   row = lg * 4 + i      -- row within 16x16 tile (4 consecutive rows per K-group)
    //   acc[i] -> C[wm*16 + lg*4 + i, wn*16 + lr]
    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}

torch::Tensor launch_hip_gemm(
    torch::Tensor A,
    torch::Tensor B_q,
    torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));

    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);

    hipLaunchKernelGGL(hip_mxfp4_gemm,
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K);

    return C;
}
"""

CPP_FORWARD = "torch::Tensor launch_hip_gemm(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);"

# ---------------------------------------------------------------------------
# Module build with caching
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
            name="hip_gemm_v4",
            cpp_sources=CPP_FORWARD,
            cuda_sources=HIP_SOURCE,
            functions=["launch_hip_gemm"],
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

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_q_u8, _b_sc_raw

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B preprocessing (unshuffle scales, view as uint8)
    bkey = B_scale_sh.data_ptr()
    if bkey != _b_cache_key:
        _b_cache_key = bkey
        _b_q_u8 = B_q.view(torch.uint8)
        _b_sc_raw = _unshuffle_e8m0(B_scale_sh)

    # Try HIP kernel for small M (fused bf16->fp4 quant + MFMA)
    mod = _get_module()
    if mod is not None and m <= 32:
        b_sc_slice = _b_sc_raw[:n, :k // 32].contiguous()
        return mod.launch_hip_gemm(A, _b_q_u8, b_sc_slice, m, n, k)

    # Fallback: proven Triton paths
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

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Tiled MFMA FP4 GEMM: 32x128 output tile, 4 warps, LDS buffering, fused A quant.
"""
import torch, os, subprocess, ctypes, hashlib
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x32_t = fp4x2_t __attribute__((ext_vector_type(32)));
typedef float fp32x16_t __attribute__((ext_vector_type(16)));

// Grid: (ceil(N/128), ceil(M/32)), Block: 256 (4 warps of 64)
// A: bf16 [M,K], B_q: uint8 [N,K/2], Bs: uint8 [N,K/32], C: bf16 [M,N]
__global__ __launch_bounds__(256)
void mfma_tiled_gemm(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ Bs,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    const int tid = threadIdx.x;       // 0-255
    const int warp_id = tid / 64;      // 0-3
    const int lane = tid % 64;         // 0-63
    const int half = lane / 32;        // 0-1 (K-half within MFMA)
    const int sublane = lane % 32;     // 0-31

    const int n_base = blockIdx.x * 128;
    const int m_base = blockIdx.y * 32;
    const int K2 = K / 2;
    const int KB = K / 32;

    // LDS for A (fp4x2) and B (fp4x2) tiles + scales
    __shared__ uint8_t a_lds[32 * 32];        // 32 rows x 32 bytes (64 fp4 per row)
    __shared__ uint8_t b_lds[128 * 32];       // 128 cols x 32 bytes (64 fp4 per col)
    __shared__ float amax_scratch[64];         // 64 groups for amax reduction
    __shared__ uint8_t a_scale_lds[64];        // 32 rows x 2 K-blocks
    __shared__ uint8_t b_scale_lds[256];       // 128 cols x 2 K-blocks

    // Accumulators: each warp computes one 32x32 MFMA tile
    fp32x16_t c_reg = {};

    for (int k64 = 0; k64 < K; k64 += 64) {
        // ============================================================
        // STAGE 1: Cooperative A load + quantize (256 threads, 64 blocks)
        // ============================================================
        // 32 rows x 2 K-blocks = 64 blocks of 32 elements
        // 256/64 = 4 threads per block, each handles 8 elements
        int a_group = tid / 4;       // 0-63: which block
        int a_sub = tid % 4;         // 0-3: which sub-group within block
        int a_row = a_group / 2;     // 0-31: row in A
        int a_kb = a_group % 2;      // 0-1: which K-block (0=first 32, 1=second 32)
        int a_elem_start = a_sub * 8; // starting element within 32-element block

        int a_global_row = m_base + a_row;
        int a_global_k = k64 + a_kb * 32 + a_elem_start;

        // Load 8 bf16 values and find local max
        float vals[8];
        float local_max = 0.0f;
        if (a_global_row < M) {
            const unsigned short* ap = A + (long long)a_global_row * K + a_global_k;
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                if (a_global_k + i < K) {
                    union { unsigned int ui; float f; } u;
                    u.ui = (unsigned int)ap[i] << 16;
                    vals[i] = u.f;
                } else {
                    vals[i] = 0.0f;
                }
                local_max = fmaxf(local_max, fabsf(vals[i]));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++) vals[i] = 0.0f;
        }

        // Amax reduction: store local max, reduce across 4 threads
        amax_scratch[a_group] = 0.0f; // init
        __syncthreads();

        // Atomic max within the group (4 threads per group)
        atomicMax((int*)&amax_scratch[a_group], __float_as_int(local_max));
        __syncthreads();

        // Compute scale (matching dynamic_mxfp4_quant exactly)
        float block_amax = amax_scratch[a_group];
        uint8_t a_scale_byte;
        float quant_scale;
        {
            union { float f; unsigned int ui; } au;
            au.f = block_amax;
            unsigned int amax_int = au.ui;
            // Round up amax to power of 2
            amax_int = ((amax_int + 0x200000u) & 0xFF800000u);
            au.ui = amax_int;
            int biased_exp = (int)((amax_int >> 23) & 0xFF);
            int scale_unbiased = biased_exp - 127 - 2;
            if (scale_unbiased < -127) scale_unbiased = -127;
            if (scale_unbiased > 127) scale_unbiased = 127;
            a_scale_byte = (uint8_t)(scale_unbiased + 127);
            // quant_scale = 2^(-scale_unbiased)
            int qs_biased = (-scale_unbiased) + 127;
            if (qs_biased < 1) qs_biased = 1;
            if (qs_biased > 254) qs_biased = 254;
            union { unsigned int ui; float f; } su;
            su.ui = (unsigned int)qs_biased << 23;
            quant_scale = su.f;
        }

        // Store A scale
        if (a_sub == 0) {
            a_scale_lds[a_group] = a_scale_byte;
        }

        // Quantize 8 values to FP4 and store to A LDS
        // A LDS layout: [row * 32 + kb * 16 + sub * 4]
        int a_lds_offset = a_row * 32 + a_kb * 16 + a_sub * 4;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float qx0 = vals[2*i] * quant_scale;
            float qx1 = vals[2*i+1] * quant_scale;
            // RNE quantization matching _mxfp4_quant_op
            uint8_t nib0, nib1;
            // Simplified: use the exact bit manipulation from Triton source
            {
                union { float f; unsigned int ui; } qu;
                qu.f = qx0;
                unsigned int qx_uint = qu.ui;
                unsigned int sign = qx_uint & 0x80000000u;
                qx_uint ^= sign;
                qu.ui = qx_uint;
                float qx_abs = qu.f;
                if (qx_abs >= 6.0f) {
                    nib0 = 0x7;
                } else if (qx_abs < 1.0f) {
                    // Denormal path
                    const unsigned int denorm_exp = (127 - 1) + (23 - 1) + 1;
                    const unsigned int denorm_mask_int = denorm_exp << 23;
                    union { unsigned int ui; float f; } dm;
                    dm.ui = denorm_mask_int;
                    float dx = qx_abs + dm.f;
                    union { float f; unsigned int ui; } dr;
                    dr.f = dx;
                    nib0 = (uint8_t)(dr.ui - denorm_mask_int);
                } else {
                    // Normal path with RNE
                    unsigned int mant_odd = (qx_uint >> 22) & 1;
                    unsigned int val_to_add = ((1 - 127) << 23) + (1 << 21) - 1;
                    unsigned int nx = qx_uint + val_to_add + mant_odd;
                    nib0 = (uint8_t)(nx >> 22);
                }
                uint8_t sign_bit = (uint8_t)(sign >> 28);
                nib0 = (nib0 & 0x7) | sign_bit;
            }
            {
                union { float f; unsigned int ui; } qu;
                qu.f = qx1;
                unsigned int qx_uint = qu.ui;
                unsigned int sign = qx_uint & 0x80000000u;
                qx_uint ^= sign;
                qu.ui = qx_uint;
                float qx_abs = qu.f;
                if (qx_abs >= 6.0f) {
                    nib1 = 0x7;
                } else if (qx_abs < 1.0f) {
                    const unsigned int denorm_exp = (127 - 1) + (23 - 1) + 1;
                    const unsigned int denorm_mask_int = denorm_exp << 23;
                    union { unsigned int ui; float f; } dm;
                    dm.ui = denorm_mask_int;
                    float dx = qx_abs + dm.f;
                    union { float f; unsigned int ui; } dr;
                    dr.f = dx;
                    nib1 = (uint8_t)(dr.ui - denorm_mask_int);
                } else {
                    unsigned int mant_odd = (qx_uint >> 22) & 1;
                    unsigned int val_to_add = ((1 - 127) << 23) + (1 << 21) - 1;
                    unsigned int nx = qx_uint + val_to_add + mant_odd;
                    nib1 = (uint8_t)(nx >> 22);
                }
                uint8_t sign_bit = (uint8_t)(sign >> 28);
                nib1 = (nib1 & 0x7) | sign_bit;
            }
            a_lds[a_lds_offset + i] = nib0 | (nib1 << 4);
        }

        // ============================================================
        // STAGE 2: Cooperative B load (256 threads, 4096 bytes)
        // ============================================================
        // 128 cols x 32 bytes = 4096 bytes, 256 threads x 16 bytes each
        int b_col = tid / 2;         // 0-127: which N-column
        int b_half_id = tid % 2;     // 0-1: which 16-byte half
        int b_n = n_base + b_col;
        int b_k_byte = k64 / 2 + b_half_id * 16;

        if (b_n < N && b_k_byte + 16 <= K2) {
            const unsigned char* bp = B_q + (long long)b_n * K2 + b_k_byte;
            uint8_t* dst = b_lds + b_col * 32 + b_half_id * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) {
                dst[i] = bp[i];
            }
        } else {
            uint8_t* dst = b_lds + b_col * 32 + b_half_id * 16;
            #pragma unroll
            for (int i = 0; i < 16; i++) dst[i] = 0;
        }

        // Load B scales: 128 cols x 2 K-blocks = 256 bytes
        if (tid < 128) {
            int bs_n = n_base + tid;
            int bs_kb0 = k64 / 32;
            if (bs_n < N) {
                b_scale_lds[tid * 2] = Bs[(long long)bs_n * KB + bs_kb0];
                b_scale_lds[tid * 2 + 1] = Bs[(long long)bs_n * KB + bs_kb0 + 1];
            } else {
                b_scale_lds[tid * 2] = 127;
                b_scale_lds[tid * 2 + 1] = 127;
            }
        }

        __syncthreads();

        // ============================================================
        // STAGE 3: MFMA compute (each warp does one 32x32 tile)
        // ============================================================
        // Load A from LDS: thread lane maps to row=sublane, K-half=half
        fp4x32_t a_reg = {};
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            a_reg[i] = a_lds[sublane * 32 + half * 16 + i];
        }
        uint8_t a_scale = a_scale_lds[sublane * 2 + half]; // row=sublane, kb=half

        // Load B from LDS: warp_id selects which 32-column chunk
        fp4x32_t b_reg = {};
        int b_col_in_tile = warp_id * 32 + sublane; // N-column within 128-col tile
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            b_reg[i] = b_lds[b_col_in_tile * 32 + half * 16 + i];
        }
        uint8_t b_scale = b_scale_lds[b_col_in_tile * 2 + half];

        // MFMA!
        c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, c_reg, 4, 4, 0, a_scale, 0, b_scale);

        __syncthreads();
    }

    // ============================================================
    // STAGE 4: Store C (each warp writes its 32x32 tile)
    // ============================================================
    int n_warp_base = n_base + warp_id * 32;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int r = m_base + half * 4 + j + i * 8;
            int c = n_warp_base + sublane;
            if (r < M && c < N) {
                union { float f; unsigned int ui; } u;
                u.f = c_reg[i * 4 + j];
                u.ui += ((u.ui >> 16) & 1) + 0x7FFF;
                C[(long long)r * N + c] = (unsigned short)(u.ui >> 16);
            }
        }
    }
}

extern "C" void launch_tiled(
    void* A, void* Bq, void* Bs, void* C,
    int M, int N, int K)
{
    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 31) / 32);
    hipLaunchKernelGGL(mfma_tiled_gemm, grid, block, 0, 0,
        (const unsigned short*)A, (const unsigned char*)Bq,
        (const unsigned char*)Bs, (unsigned short*)C, M, N, K);
}
"""

_lib = None
_cache = {}

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mfma_tiled_{h}.hip'
    so = f'/tmp/_mfma_tiled_{h}.so'
    with open(src, 'w') as f:
        f.write(HIP_SRC)
    hipcc = 'hipcc'
    for p in ['/opt/rocm/bin/hipcc', '/opt/rocm/hip/bin/hipcc']:
        if os.path.exists(p):
            hipcc = p
            break
    r = subprocess.run(
        [hipcc, '-shared', '-fPIC', '-O3', '--offload-arch=gfx950',
         '-std=c++17', '-ffast-math', '-o', so, src],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"COMPILE FAILED:\n{r.stderr[:2000]}")
        return None
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.launch_tiled.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    _lib.launch_tiled.restype = None
    return _lib

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    lib = _compile()
    if lib is None:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        bu = B_q.view(torch.uint8)
        bs_raw = _unshuffle_e8m0(B_scale_sh)
        return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

    bk = B_q.data_ptr()
    if bk not in _cache:
        _cache.clear()
        bu = B_q.view(torch.uint8).contiguous()
        bs_raw = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous().view(torch.uint8)
        _cache[bk] = (bu, bs_raw)

    bu, bs = _cache[bk]
    C = torch.zeros(M, N, dtype=torch.bfloat16, device='cuda')

    lib.launch_tiled(
        A.data_ptr(), bu.data_ptr(), bs.data_ptr(), C.data_ptr(),
        M, N, K)
    return C

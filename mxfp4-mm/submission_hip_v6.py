#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP kernel v6: Proper e8m0 A scaling + scale padding fix.
Uses v3's compute_scale logic + factor out scales to end of K-block.
"""
import torch, os, subprocess, ctypes
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>

// FP4 E2M1 dequant: nibble → float (sign in bit3, magnitude in bits 2:0)
__device__ __forceinline__ float fp4_val(unsigned char nib) {
    // Local array avoids __constant__ memory init issues with ctypes-loaded .so
    const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return lut[nib];
}

__device__ __forceinline__ float bf16_f32(unsigned short x) {
    union { unsigned int i; float f; } u;
    u.i = (unsigned int)x << 16;
    return u.f;
}

__device__ __forceinline__ unsigned short f32_bf16(float x) {
    union { float f; unsigned int i; } u;
    u.f = x;
    u.i += ((u.i >> 16) & 1) + 0x7FFF;
    return (unsigned short)(u.i >> 16);
}

__device__ __forceinline__ float e8m0_f32(unsigned char e) {
    union { unsigned int i; float f; } u;
    u.i = (unsigned int)e << 23;
    return u.f;
}

// Compute e8m0 shared exponent for A quantization
// Matches aiter's internal MXFP4 quantization
__device__ __forceinline__ float compute_a_scale(float amax) {
    if (amax == 0.0f) return 0.0f;
    union { float f; unsigned int i; } u;
    u.f = amax;
    int be = (int)((u.i >> 23) & 0xFF);
    unsigned int mantissa = u.i & 0x7FFFFF;
    // If mantissa > 0.5 (number > 1.5*2^exp), use be-1 to avoid clipping
    // Otherwise use be-2 for better precision
    int se = be - ((mantissa > 0x400000) ? 1 : 2);
    if (se < 1) se = 1;
    if (se > 254) se = 254;
    u.i = (unsigned int)se << 23;
    return u.f;
}

// Quantize float to fp4 nibble (4 bits: sign + 3-bit magnitude index)
__device__ __forceinline__ unsigned char qfp4(float v) {
    unsigned char s = (v < 0.0f) ? 8u : 0u;
    float a = fabsf(v);
    unsigned char m;
    if      (a < 0.25f) m = 0;
    else if (a < 0.75f) m = 1;
    else if (a < 1.25f) m = 2;
    else if (a < 1.75f) m = 3;
    else if (a < 2.5f)  m = 4;
    else if (a < 3.5f)  m = 5;
    else if (a < 5.0f)  m = 6;
    else                m = 7;
    return s | m;
}

template<int RPB>
__global__ __launch_bounds__(256)
void mxfp4_gemm(
    const unsigned short* __restrict__ A,
    const unsigned char* __restrict__ Bt,
    const unsigned char* __restrict__ Bs,
    unsigned short* __restrict__ C,
    int M, int N, int K)
{
    const int n = blockIdx.x * 256 + threadIdx.x;
    const int m0 = blockIdx.y * RPB;
    if (n >= N) return;

    float acc[RPB];
    #pragma unroll
    for (int r = 0; r < RPB; r++) acc[r] = 0.0f;

    const int nkb = K >> 5;

    for (int kb = 0; kb < nkb; kb++) {
        // B scale for this K-block
        float bsc = e8m0_f32(Bs[kb * N + n]);

        // Load B: 16 bytes = 32 fp4 values, decode to raw LUT values (no scale)
        const unsigned char* bp = Bt + ((long long)kb * N + n) * 16;
        float bv[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char byte = bp[j];
            bv[2*j]   = fp4_val(byte & 0xF);   // includes sign, no scale
            bv[2*j+1] = fp4_val(byte >> 4);
        }

        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            int m = m0 + r;
            if (m >= M) break;

            // Load A block and find amax
            const unsigned short* ap = A + (long long)m * K + kb * 32;
            float amax = 0.0f;
            float av[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                av[j] = bf16_f32(ap[j]);
                amax = fmaxf(amax, fabsf(av[j]));
            }

            // Compute e8m0 scale for A
            float asc = compute_a_scale(amax);
            float inv = (asc > 0.0f) ? (1.0f / asc) : 0.0f;

            // Quantize A to fp4 nibbles, dot with B fp4 values (both unscaled)
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                dot += fp4_val(qfp4(av[j] * inv)) * bv[j];
            }

            // Apply both scales at end
            acc[r] += dot * asc * bsc;
        }
    }

    #pragma unroll
    for (int r = 0; r < RPB; r++) {
        int m = m0 + r;
        if (m >= M) break;
        C[(long long)m * N + n] = f32_bf16(acc[r]);
    }
}

extern "C" void launch_gemm(
    void* A, void* Bt, void* Bs, void* C,
    int M, int N, int K, int rpb)
{
    dim3 block(256);
    dim3 grid((N + 255) / 256, 1);

    if (rpb <= 4) {
        grid.y = (M + 3) / 4;
        hipLaunchKernelGGL(mxfp4_gemm<4>, grid, block, 0, 0,
            (const unsigned short*)A, (const unsigned char*)Bt,
            (const unsigned char*)Bs, (unsigned short*)C, M, N, K);
    } else if (rpb <= 8) {
        grid.y = (M + 7) / 8;
        hipLaunchKernelGGL(mxfp4_gemm<8>, grid, block, 0, 0,
            (const unsigned short*)A, (const unsigned char*)Bt,
            (const unsigned char*)Bs, (unsigned short*)C, M, N, K);
    } else {
        grid.y = (M + 15) / 16;
        hipLaunchKernelGGL(mxfp4_gemm<16>, grid, block, 0, 0,
            (const unsigned short*)A, (const unsigned char*)Bt,
            (const unsigned char*)Bs, (unsigned short*)C, M, N, K);
    }
}
"""

_lib = None
_cache = {}

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    import hashlib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mxfp4_gemm_{h}.hip'
    so = f'/tmp/_mxfp4_gemm_{h}.so'
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
        raise RuntimeError(f"hipcc failed:\n{r.stderr}")
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.launch_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib.launch_gemm.restype = None
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

    ck = B_q.data_ptr()
    if ck not in _cache:
        _cache.clear()
        bu = B_q.view(torch.uint8)  # [N, K/2]
        Bt = bu.view(N, K // 32, 16).permute(1, 0, 2).contiguous()  # [K/32, N, 16]
        bs_raw = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous()  # [N, K/32] trimmed
        Bs = bs_raw.view(torch.uint8).t().contiguous()  # [K/32, N]
        _cache[ck] = (Bt, Bs)

    Bt, Bs = _cache[ck]
    C = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
    rpb = 4 if M <= 4 else (8 if M <= 64 else 16)

    lib.launch_gemm(
        A.data_ptr(), Bt.data_ptr(), Bs.data_ptr(), C.data_ptr(),
        M, N, K, rpb)
    return C

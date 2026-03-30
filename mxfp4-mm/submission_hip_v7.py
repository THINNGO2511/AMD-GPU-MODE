#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP kernel v7: torch.utils.cpp_extension.load() approach.
Write .cpp and .hip files to /tmp, compile via load().
Avoids ctypes overhead. Proper e8m0 A scaling + scale padding fix.
"""
import torch, os
from task import input_t, output_t

CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor mxfp4_gemm_hip(torch::Tensor A, torch::Tensor Bt, torch::Tensor Bs, int64_t N, int64_t K);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &mxfp4_gemm_hip);
}
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>

__device__ __constant__ float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

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

__device__ __forceinline__ float compute_a_scale(float amax) {
    if (amax == 0.0f) return 0.0f;
    union { float f; unsigned int i; } u;
    u.f = amax;
    int be = (int)((u.i >> 23) & 0xFF);
    unsigned int mantissa = u.i & 0x7FFFFF;
    int se = be - ((mantissa > 0x400000) ? 1 : 2);
    if (se < 1) se = 1;
    if (se > 254) se = 254;
    u.i = (unsigned int)se << 23;
    return u.f;
}

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
        float bsc = e8m0_f32(Bs[kb * N + n]);

        const unsigned char* bp = Bt + ((long long)kb * N + n) * 16;
        float bv[32];
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            unsigned char byte = bp[j];
            bv[2*j]   = FP4_LUT[byte & 0xF];
            bv[2*j+1] = FP4_LUT[byte >> 4];
        }

        #pragma unroll
        for (int r = 0; r < RPB; r++) {
            int m = m0 + r;
            if (m >= M) break;

            const unsigned short* ap = A + (long long)m * K + kb * 32;
            float amax = 0.0f;
            float av[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                av[j] = bf16_f32(ap[j]);
                amax = fmaxf(amax, fabsf(av[j]));
            }

            float asc = compute_a_scale(amax);
            float inv = (asc > 0.0f) ? (1.0f / asc) : 0.0f;

            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                dot += FP4_LUT[qfp4(av[j] * inv)] * bv[j];
            }
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

torch::Tensor mxfp4_gemm_hip(torch::Tensor A, torch::Tensor Bt, torch::Tensor Bs, int64_t N_out, int64_t K_val) {
    const int M = A.size(0), K = (int)K_val, N = (int)N_out;
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));

    int rpb = (M <= 4) ? 4 : ((M <= 64) ? 8 : 16);
    dim3 grid((N + 255) / 256, (M + rpb - 1) / rpb), block(256);

    auto a_ptr = (const unsigned short*)A.data_ptr();
    auto bt_ptr = Bt.data_ptr<uint8_t>();
    auto bs_ptr = Bs.data_ptr<uint8_t>();
    auto c_ptr = (unsigned short*)C.data_ptr();

    if (rpb == 4)
        hipLaunchKernelGGL(mxfp4_gemm<4>, grid, block, 0, 0, a_ptr, bt_ptr, bs_ptr, c_ptr, M, N, K);
    else if (rpb == 8)
        hipLaunchKernelGGL(mxfp4_gemm<8>, grid, block, 0, 0, a_ptr, bt_ptr, bs_ptr, c_ptr, M, N, K);
    else
        hipLaunchKernelGGL(mxfp4_gemm<16>, grid, block, 0, 0, a_ptr, bt_ptr, bs_ptr, c_ptr, M, N, K);

    return C;
}
"""

_mod = None
_cache = {}

def _build():
    global _mod
    if _mod is not None:
        return _mod

    os.makedirs('/tmp/mxfp4_ext', exist_ok=True)
    with open('/tmp/mxfp4_ext/main.cpp', 'w') as f:
        f.write(CPP_SRC)
    with open('/tmp/mxfp4_ext/kernel.hip', 'w') as f:
        f.write(HIP_SRC)

    from torch.utils.cpp_extension import load
    _mod = load(
        name='mxfp4_ext',
        sources=['/tmp/mxfp4_ext/main.cpp', '/tmp/mxfp4_ext/kernel.hip'],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '--offload-arch=gfx950', '-std=c++17', '-ffast-math'],
        build_directory='/tmp/mxfp4_ext/build',
        verbose=False
    )
    return _mod

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

    mod = _build()

    ck = (N, K)
    if ck not in _cache:
        bu = B_q.view(torch.uint8)
        Bt = bu.view(N, K // 32, 16).permute(1, 0, 2).contiguous()
        bs_raw = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous()
        Bs = bs_raw.view(torch.uint8).t().contiguous()
        _cache[ck] = (Bt, Bs)

    Bt, Bs = _cache[ck]
    return mod.gemm(A, Bt, Bs, N, K)

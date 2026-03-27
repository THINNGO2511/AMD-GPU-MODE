"""GEMM HIP: salykova pattern with trivial scales (127=1.0) to test data loading"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(16))) float16v;

__device__ uint8_t make_fp4x2(uint8_t lo, uint8_t hi) {
    return (hi << 4) | (lo & 0x0F);
}

__device__ void quant32_trivial(const hip_bfloat16* src, uint8_t* dst, int valid) {
    // Quantize with scale=1.0 (trivial) — just round to nearest FP4
    for (int i = 0; i < 16; i++) {
        float v0 = (2*i < valid) ? (float)src[2*i] : 0.0f;
        float v1 = (2*i+1 < valid) ? (float)src[2*i+1] : 0.0f;
        auto fp4 = [](float x) -> uint8_t {
            int s = (x < 0) ? 8 : 0;
            float a = fabsf(x);
            return s | ((a < 0.25f) ? 0 : (a < 0.75f) ? 1 : (a < 1.25f) ? 2 :
                        (a < 1.75f) ? 3 : (a < 2.5f) ? 4 : (a < 3.5f) ? 5 :
                        (a < 5.0f) ? 6 : 7);
        };
        dst[i] = make_fp4x2(fp4(v0), fp4(v1));
    }
}

__global__ void gemm_trivial(
    const hip_bfloat16* A, const uint8_t* B_T, hip_bfloat16* C,
    int M, int N, int K)
{
    int tn = blockIdx.x, tm = blockIdx.y, lane = threadIdx.x;
    int mb = tm * 32, nb = tn * 32;
    
    float16v acc = {};
    
    for (int k = 0; k < K; k += 64) {
        // A: thread loads row (lane%32), K[(lane/32)*32 : +32] = 16 bytes
        int ar = mb + (lane % 32);
        int ak = k + (lane / 32) * 32;
        uint8_t ap[16] = {0};
        if (ar < M && ak + 31 < K)
            quant32_trivial(&A[(long long)ar * K + ak], ap, 32);
        int8v a_reg = {};
        __builtin_memcpy(&a_reg, ap, 16);
        
        // B: transposed [K/2, N], load column (lane%32)
        int bc = nb + (lane % 32);
        uint8_t bp[32] = {0};
        if (bc < N) {
            int bk = k / 2;
            const uint8_t* src = B_T + (long long)bk * N + bc;
            // Load with stride N (column access in row-major [K/2, N])
            for (int i = 0; i < 32 && bk + i < K/2; i++)
                bp[i] = src[(long long)i * N];
        }
        int8v b_reg = {};
        __builtin_memcpy(&b_reg, bp, 32);
        
        // ALL SCALES = 127 (trivial 1.0)
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, 127, 0, 127);
    }
    
    // Store
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = mb + (lane/32)*4 + j + i*8;
            int c = nb + (lane%32);
            if (r < M && c < N)
                C[(long long)r * N + c] = (hip_bfloat16)acc[i*4+j];
        }
}

torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B_T,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_trivial, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const uint8_t*)B_T.data_ptr(),
        (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B_T, int64_t M, int64_t N, int64_t K);"

print("Compiling trivial-scale GEMM...")
_mod = load_inline(name="trivial_scale_v1", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                   functions=["launch_gemm"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"])
print("OK!")

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    # Transpose B_q from [N, K/2] to [K/2, N]
    B_T = B_q.view(torch.uint8).t().contiguous()
    return _mod.launch_gemm(A, B_T, M, N, K)

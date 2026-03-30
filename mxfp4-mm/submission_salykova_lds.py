"""GEMM HIP: salykova pattern with LDS staging for B tile"""
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

// LDS: 32 K-rows × 32 N-cols = 1024 bytes for B tile


__global__ void gemm_lds(
    const hip_bfloat16* A, const uint8_t* B_q,
    hip_bfloat16* C, int M, int N, int K)
{
    int tn = blockIdx.x, tm = blockIdx.y, lane = threadIdx.x;
    __shared__ uint8_t lds_b[1024];
    int mb = tm * 32, nb = tn * 32;
    int Kp = K / 2;
    
    float16v acc = {};
    
    for (int k = 0; k < K; k += 64) {
        // ---- Copy B tile to LDS ----
        // B_q is [N, K/2] row-major
        // Copy 32 N-rows × 32 K-bytes into LDS as [K/2=32, N=32]
        // Each of 64 threads copies 16 bytes (32*32/64 = 16 bytes per thread)
        {
            int bytes_per_thread = 1024 / 64;  // 16
            int my_start = lane * bytes_per_thread;
            for (int b = 0; b < bytes_per_thread; b++) {
                int flat = my_start + b;
                int k_byte = flat / 32;  // which K-row in the tile (0..31)
                int n_col = flat % 32;   // which N-col in the tile (0..31)
                
                int global_n = nb + n_col;
                int global_k = k / 2 + k_byte;
                
                uint8_t val = 0;
                if (global_n < N && global_k < Kp)
                    val = B_q[(long long)global_n * Kp + global_k];
                
                // Store in LDS as [K/2, N] = k_byte * 32 + n_col
                lds_b[k_byte * 32 + n_col] = val;
            }
        }
        __syncthreads();
        
        // ---- Load A (salykova pattern) ----
        int ar = mb + (lane % 32);
        int ak = k + (lane / 32) * 32;
        uint8_t ap[16] = {0};
        if (ar < M && ak + 31 < K)
            quant32_trivial(&A[(long long)ar * K + ak], ap, 32);
        int8v a_reg = {};
        __builtin_memcpy(&a_reg, ap, 16);
        
        // ---- Load B from LDS (EXACT salykova pattern) ----
        // LDS is [32, 32] = k_byte * 32 + n_col
        // ldg_b = lds_b + (lane%32)/2 + 16*32*(lane/32)
        const uint8_t* ldg_b = lds_b + (lane % 32) / 2 + 16 * 32 * (lane / 32);
        int b_extract = lane % 2;
        
        uint8_t bp[16];
        for (int i = 0; i < 16; i++) {
            uint8_t byte0 = *(ldg_b + 16 * 2 * i);
            uint8_t byte1 = *(ldg_b + 16 * (2 * i + 1));
            uint8_t nib0 = (b_extract == 0) ? (byte0 & 0x0F) : ((byte0 >> 4) & 0x0F);
            uint8_t nib1 = (b_extract == 0) ? (byte1 & 0x0F) : ((byte1 >> 4) & 0x0F);
            bp[i] = make_fp4x2(nib0, nib1);
        }
        int8v b_reg = {};
        __builtin_memcpy(&b_reg, bp, 16);
        
        __syncthreads();
        
        // ALL SCALES = 127 (trivial) for now
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, 127, 0, 127);
    }
    
    // Store (salykova pattern)
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = mb + (lane/32)*4 + j + i*8;
            int c = nb + (lane%32);
            if (r < M && c < N)
                C[(long long)r * N + c] = (hip_bfloat16)acc[i*4+j];
        }
}

torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor Bq,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_lds, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const uint8_t*)Bq.data_ptr(),
        (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor Bq, int64_t M, int64_t N, int64_t K);"

print("Compiling LDS version...")
_mod = load_inline(name="salykova_lds_v2", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                   functions=["launch_gemm"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"])
print("OK!")

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    Bq = B_q.view(torch.uint8)
    return _mod.launch_gemm(A, Bq, M, N, K)

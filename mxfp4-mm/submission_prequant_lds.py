"""GEMM HIP: Pre-quantize A with aiter, load from LDS with salykova B pattern"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline
from aiter.utility.fp4_utils import dynamic_mxfp4_quant

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(16))) float16v;

__device__ uint8_t make_fp4x2(uint8_t lo, uint8_t hi) {
    return (hi << 4) | (lo & 0x0F);
}

__global__ void gemm_pq(
    const uint8_t* A_q,     // [M, K/2] pre-quantized FP4x2
    const uint8_t* B_q,     // [N, K/2] FP4x2
    hip_bfloat16* C,
    int M, int N, int K)
{
    __shared__ uint8_t lds_b[1024];  // 32 K-rows × 32 N-cols
    
    int tn = blockIdx.x, tm = blockIdx.y, lane = threadIdx.x;
    int mb = tm * 32, nb = tn * 32;
    int Kp = K / 2;
    
    float16v acc = {};
    
    for (int k = 0; k < K; k += 64) {
        // Copy B tile to LDS: [32 K-bytes × 32 N-cols] from B_q[N, K/2]
        {
            for (int b = 0; b < 16; b++) {
                int flat = lane * 16 + b;
                int kb = flat / 32;  // 0..31
                int nc = flat % 32;  // 0..31
                int gn = nb + nc;
                int gk = k / 2 + kb;
                uint8_t v = 0;
                if (gn < N && gk < Kp)
                    v = B_q[(long long)gn * Kp + gk];
                lds_b[kb * 32 + nc] = v;
            }
        }
        __syncthreads();
        
        // Load A: pre-quantized [M, K/2], salykova pattern
        // lane loads A_q[row=mb+lane%32, k_byte=k/2 + (lane/32)*16 .. +16]
        int ar = mb + (lane % 32);
        int ak = k / 2 + (lane / 32) * 16;
        uint8_t ap[16] = {0};
        if (ar < M && ak + 15 < Kp) {
            const uint8_t* src = A_q + (long long)ar * Kp + ak;
            for (int i = 0; i < 16; i++) ap[i] = src[i];
        }
        int8v a_reg = {};
        __builtin_memcpy(&a_reg, ap, 16);
        
        // Load B from LDS: salykova pattern
        const uint8_t* ldg_b = lds_b + (lane % 32) / 2 + 16 * 32 * (lane / 32);
        int bx = lane % 2;
        uint8_t bp[16];
        for (int i = 0; i < 16; i++) {
            uint8_t b0 = *(ldg_b + 16 * 2 * i);
            uint8_t b1 = *(ldg_b + 16 * (2 * i + 1));
            uint8_t n0 = (bx == 0) ? (b0 & 0x0F) : ((b0 >> 4) & 0x0F);
            uint8_t n1 = (bx == 0) ? (b1 & 0x0F) : ((b1 >> 4) & 0x0F);
            bp[i] = make_fp4x2(n0, n1);
        }
        int8v b_reg = {};
        __builtin_memcpy(&b_reg, bp, 16);
        
        __syncthreads();
        
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, 127, 0, 127);
    }
    
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            int r = mb + (lane/32)*4 + j + i*8;
            int c = nb + (lane%32);
            if (r < M && c < N)
                C[(long long)r * N + c] = (hip_bfloat16)acc[i*4+j];
        }
}

torch::Tensor launch_gemm(torch::Tensor Aq, torch::Tensor Bq,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(Aq.device()));
    dim3 g((N+31)/32, (M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_pq, g, b, 0, 0,
        (const uint8_t*)Aq.data_ptr(), (const uint8_t*)Bq.data_ptr(),
        (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor Aq, torch::Tensor Bq, int64_t M, int64_t N, int64_t K);"

print("Compiling prequant LDS...")
_mod = load_inline(name="prequant_lds_v1", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                   functions=["launch_gemm"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"])
print("OK!")

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    # Pre-quantize A with aiter (exact same quant as reference)
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_q_u8 = A_q.view(torch.uint8)
    B_q_u8 = B_q.view(torch.uint8)
    # Still using trivial scales for now — testing data loading only
    return _mod.launch_gemm(A_q_u8, B_q_u8, M, N, K)

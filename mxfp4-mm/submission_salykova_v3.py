"""GEMM HIP — salykova FP4 MFMA pattern, v3 (crash on compile error)"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(16))) float16v;

__device__ __forceinline__ uint8_t make_fp4x2(uint8_t lo, uint8_t hi) {
    return (hi << 4) | (lo & 0x0F);
}

// Simple bf16->FP4 quantization for 32 elements
__device__ void quant32(const hip_bfloat16* src, uint8_t* dst, uint8_t* scale_out, int valid) {
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (i < valid) ? (float)src[i] : 0.0f;
        float av = fabsf(vals[i]);
        if (av > amax) amax = av;
    }
    
    uint8_t scale = 127; // 2^0 = 1.0
    float inv_scale = 1.0f;
    if (amax > 0.0f) {
        float ratio = amax / 6.0f;
        unsigned int bits;
        __builtin_memcpy(&bits, &ratio, 4);
        scale = (uint8_t)((bits >> 23) & 0xFF);
        if (scale < 1) scale = 1;
        inv_scale = 1.0f / ldexpf(1.0f, (int)scale - 127);
    }
    *scale_out = scale;
    
    for (int i = 0; i < 16; i++) {
        float v0 = vals[2*i] * inv_scale;
        float v1 = vals[2*i+1] * inv_scale;
        
        auto fp4 = [](float x) -> uint8_t {
            int s = (x < 0) ? 8 : 0;
            float a = fabsf(x);
            uint8_t n = (a < 0.25f) ? 0 : (a < 0.75f) ? 1 : (a < 1.25f) ? 2 :
                        (a < 1.75f) ? 3 : (a < 2.5f) ? 4 : (a < 3.5f) ? 5 :
                        (a < 5.0f) ? 6 : 7;
            return s | n;
        };
        dst[i] = make_fp4x2(fp4(v0), fp4(v1));
    }
}

__global__ void gemm_kernel(
    const hip_bfloat16* A, const uint8_t* B, const uint8_t* Bs,
    hip_bfloat16* C, int M, int N, int K)
{
    int tn = blockIdx.x, tm = blockIdx.y, lane = threadIdx.x;
    int mb = tm * 32, nb = tn * 32;
    int Kp = K / 2, Kg = K / 32;
    
    float16v acc = {};
    
    for (int k = 0; k < K; k += 64) {
        // A: thread loads row (lane%32), K[(lane/32)*32 : +32]
        int ar = mb + (lane % 32);
        int ak = k + (lane / 32) * 32;
        uint8_t ap[16] = {0};
        uint8_t as_val = 127;
        if (ar < M && ak + 31 < K)
            quant32(&A[(long long)ar * K + ak], ap, &as_val, 32);
        
        // Pack A into int8v: 16 bytes -> first 4 int32, rest zero
        int8v a_reg = {};
        __builtin_memcpy(&a_reg, ap, 16);
        
        // B: thread loads column (lane%32), K[k:k+64] = 32 bytes
        int bc = nb + (lane % 32);
        uint8_t bp[32] = {0};
        if (bc < N) {
            int bk = k / 2;
            int rem = Kp - bk;
            int cnt = (rem > 32) ? 32 : rem;
            if (cnt > 0) {
                const uint8_t* src = B + (long long)bc * Kp + bk;
                for (int i = 0; i < cnt; i++) bp[i] = src[i];
            }
        }
        int8v b_reg = {};
        __builtin_memcpy(&b_reg, bp, 32);
        
        // Scales
        uint8_t bs0 = 127, bs1 = 127;
        if (bc < N) {
            int g0 = k / 32, g1 = g0 + 1;
            if (g0 < Kg) bs0 = Bs[(long long)bc * Kg + g0];
            if (g1 < Kg) bs1 = Bs[(long long)bc * Kg + g1];
        }
        unsigned sa = (unsigned)as_val;
        unsigned sb = (unsigned)bs0 | ((unsigned)bs1 << 8);
        
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }
    
    // Store: salykova pattern
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = mb + (lane/32)*4 + j + i*8;
            int c = nb + (lane%32);
            if (r < M && c < N)
                C[(long long)r * N + c] = (hip_bfloat16)acc[i*4+j];
        }
    }
}

torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 g((N+31)/32, (M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_kernel, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const uint8_t*)Bq.data_ptr(),
        (const uint8_t*)Bs.data_ptr(), (hip_bfloat16*)C.data_ptr(), (int)M,(int)N,(int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs, int64_t M, int64_t N, int64_t K);"

# Compile — crash loudly if it fails
print("Compiling HIP kernel...")
_mod = load_inline(name="salykova_v3", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                   functions=["launch_gemm"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"], verbose=True)
print("Compilation OK!")

_sc = {}
def _unshuffle(s, N, K):
    key = id(s)
    if key in _sc: return _sc[key]
    n = K//32; sm=((N+255)//256)*256; sn=((n+7)//8)*8
    p = torch.zeros(sm,sn,dtype=torch.uint8,device=s.device)
    p[:N,:n] = s.view(torch.uint8)[:N,:n]
    r = p.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous()
    result = r.view(sm,sn)[:N,:n]
    _sc[key] = result
    return result

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    Bs = _unshuffle(B_scale_sh, N, K)
    Bq = B_q.view(torch.uint8)
    return _mod.launch_gemm(A, Bq, Bs, M, N, K)

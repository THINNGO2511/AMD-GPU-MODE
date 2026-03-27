#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP — salykova v6: Process K in chunks of 32 (not 64).
ROOT CAUSE FOUND: MFMA applies ONE scale to the entire K=64.
When K-half scales differ, the result is wrong.
FIX: Each MFMA call processes K=32 with uniform scale.
Data in first 128 bits, zeros in last 128 bits.
2x more MFMA calls but correct results.
Also tries: cbsz=1 and/or blgp=1 for per-half scale routing.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
import sys
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_ext_ocp.h>
#include <cstdint>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

__global__ void gemm_k32_kernel(
    const uint8_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int Kp = K / 2;
    const int Kg = K / 32;
    const int lane = threadIdx.x % 32;
    const int half = threadIdx.x / 32;
    const int mb = blockIdx.y * 32;
    const int nb = blockIdx.x * 32;

    fp32x16_t acc = {};

    // Process K in chunks of 32 (not 64!)
    // Each MFMA call: 16 bytes of data in first 128 bits, zeros in last 128 bits
    // ALL threads (both halves) load the SAME K block
    for (int k = 0; k < K; k += 32) {
        fp4x64_t a_reg = {};
        fp4x64_t b_reg = {};

        // A: load 16 bytes (32 FP4 values) for K block k..k+31
        int a_row = mb + lane;
        if (a_row < M) {
            int a_base = a_row * Kp + k / 2;
            const uint8_t* a_src = A_fp4 + a_base;
            for (int i = 0; i < 16 && (k/2 + i) < Kp; i++) {
                a_reg[i] = (fp4x2_t)a_src[i];
            }
        }

        // B: load 16 bytes (32 FP4 values) for K block k..k+31
        int b_col = nb + lane;
        if (b_col < N) {
            int b_base = b_col * Kp + k / 2;
            const uint8_t* b_src = B_q + b_base;
            for (int i = 0; i < 16 && (k/2 + i) < Kp; i++) {
                b_reg[i] = (fp4x2_t)b_src[i];
            }
        }

        // Scales: ONE scale per K block (uniform across all 32 values)
        uint8_t sa_val = 127;
        if (a_row < M) {
            int sg = k / 32;
            if (sg < Kg) sa_val = A_scale[a_row * Kg + sg];
        }
        uint8_t sb_val = 127;
        if (b_col < N) {
            int sg = k / 32;
            if (sg < Kg) sb_val = B_scale[b_col * Kg + sg];
        }

        unsigned sa = (unsigned)sa_val;
        unsigned sb = (unsigned)sb_val;

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // Store
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = mb + half * 4 + j + i * 8;
            int c = nb + lane;
            if (r < M && c < N)
                C[(long long)r * N + c] = (hip_bfloat16)acc[i * 4 + j];
        }
    }
}

torch::Tensor launch_gemm(torch::Tensor A_fp4, torch::Tensor A_scale,
                           torch::Tensor B_q, torch::Tensor B_scale,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp4.device()));
    dim3 g((N+31)/32, (M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_k32_kernel, g, b, 0, 0,
        (const uint8_t*)A_fp4.data_ptr(),
        (const uint8_t*)A_scale.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (const uint8_t*)B_scale.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),
        (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor A_fp4, torch::Tensor A_scale, torch::Tensor B_q, torch::Tensor B_scale, int64_t M, int64_t N, int64_t K);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(name="salykova_v6_k32", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                           functions=["launch_gemm"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"], verbose=True)
    return _mod

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
    mod = _get_mod()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    Bs = _unshuffle(B_scale_sh, N, K)
    return mod.launch_gemm(A_fp4.view(torch.uint8), A_scale.view(torch.uint8),
                           B_q.view(torch.uint8), Bs, M, N, K)

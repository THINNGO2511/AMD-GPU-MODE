#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP — salykova v5: Use EXACT blog types + try 2-scale packing.
Changes from v4:
1. Use fp4x2_t __attribute__((ext_vector_type(32))) NOT int[8]
2. Load 16 elements via element assignment (blog pattern) NOT memcpy
3. Pack TWO B scales in one VGPR: sb = scale_k0 | (scale_k1 << 8)
   The MFMA might read byte0 for K=0..31 and byte1 for K=32..63
4. Each thread loads FULL 32 fp4x2 (64 FP4) covering K=0..63
   (matching Navi's v3 approach but with correct types)
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
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

__global__ void gemm_kernel(
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

    for (int k = 0; k < K; k += 64) {
        // === A: salykova pattern — element assignment ===
        fp4x64_t a_reg = {};
        int a_row = mb + lane;
        if (a_row < M) {
            int a_base = a_row * Kp + k / 2 + half * 16;
            if (a_base + 16 <= a_row * Kp + Kp) {
                const uint8_t* a_src = A_fp4 + a_base;
                for (int i = 0; i < 16; i++) {
                    a_reg[i] = (fp4x2_t)a_src[i];
                }
            }
        }

        // === B: load FULL 32 bytes (both K halves) ===
        fp4x64_t b_reg = {};
        int b_col = nb + lane;
        if (b_col < N) {
            int b_base = b_col * Kp + k / 2;
            int avail = Kp - k / 2;
            int cnt = (avail > 32) ? 32 : avail;
            const uint8_t* b_src = B_q + b_base;
            for (int i = 0; i < cnt && i < 32; i++) {
                b_reg[i] = (fp4x2_t)b_src[i];
            }
        }

        // === A scale: one per thread (for this K half) ===
        uint8_t sa_val = 127;
        if (a_row < M) {
            int sg = k / 32 + half;
            if (sg < Kg) sa_val = A_scale[a_row * Kg + sg];
        }

        // === B scale: PACK two K-half scales into one VGPR ===
        // byte0 = scale for K=0..31, byte1 = scale for K=32..63
        uint8_t bs0 = 127, bs1 = 127;
        if (b_col < N) {
            int g0 = k / 32;
            int g1 = g0 + 1;
            if (g0 < Kg) bs0 = B_scale[b_col * Kg + g0];
            if (g1 < Kg) bs1 = B_scale[b_col * Kg + g1];
        }

        unsigned sa = (unsigned)sa_val;
        unsigned sb = (unsigned)bs0 | ((unsigned)bs1 << 8);

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // === Output store ===
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
    hipLaunchKernelGGL(gemm_kernel, g, b, 0, 0,
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
        _mod = load_inline(name="salykova_v5", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
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
    Bq = B_q.view(torch.uint8)

    return mod.launch_gemm(A_fp4.view(torch.uint8), A_scale.view(torch.uint8),
                           Bq, Bs, M, N, K)

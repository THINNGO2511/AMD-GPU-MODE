#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP — salykova v4 FIX: B register must be HALF-FILLED.
ROOT CAUSE FOUND: Each thread must load only 16 bytes (32 FP4, one K half).
The remaining 16 bytes MUST be zero. The MFMA splits the 256-bit register
into two 128-bit halves: threads 0-31 provide K=0..31, threads 32-63 K=32..63.

v3 bug: loaded ALL 32 bytes = 64 FP4 per thread, mixing both K halves.
v4 fix: load ONLY 16 bytes per thread, matching salykova's pattern.

Also uses aiter's dynamic_mxfp4_quant for A (proven correct).
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
#include <cstdint>

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(16))) float16v;

__global__ void gemm_kernel(
    const uint8_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int Kp = K / 2;   // fp4x2 bytes per row
    const int Kg = K / 32;  // scale blocks per row
    const int lane = threadIdx.x;
    const int half = lane / 32;  // 0 or 1
    const int col = lane % 32;
    const int mb = blockIdx.y * 32;
    const int nb = blockIdx.x * 32;

    float16v acc = {};

    for (int k = 0; k < K; k += 64) {
        // === A loading: 16 bytes per thread (32 FP4 for this K half) ===
        int a_row = mb + col;
        int a_k_byte = k / 2 + half * 16;  // byte offset for this half
        uint8_t ap[32] = {0};  // 32 bytes, second half stays zero
        if (a_row < M && a_k_byte + 16 <= Kp) {
            const uint8_t* a_src = A_fp4 + (long long)a_row * Kp + a_k_byte;
            for (int i = 0; i < 16; i++) ap[i] = a_src[i];
        }
        int8v a_reg = {};
        __builtin_memcpy(&a_reg, ap, 32);

        // === B loading: 16 bytes per thread (32 FP4 for this K half) ===
        // KEY FIX: Only load 16 bytes, NOT 32! Zero-pad the rest.
        int b_col = nb + col;
        int b_k_byte = k / 2 + half * 16;  // byte offset for this half
        uint8_t bp[32] = {0};  // 32 bytes, second half stays zero
        if (b_col < N && b_k_byte + 16 <= Kp) {
            const uint8_t* b_src = B_q + (long long)b_col * Kp + b_k_byte;
            for (int i = 0; i < 16; i++) bp[i] = b_src[i];
        }
        int8v b_reg = {};
        __builtin_memcpy(&b_reg, bp, 32);

        // === Scales: one per thread (for this thread's K half) ===
        uint8_t sa_val = 127;
        if (a_row < M) {
            int sg = k / 32 + half;
            if (sg < Kg) sa_val = A_scale[(long long)a_row * Kg + sg];
        }

        uint8_t sb_val = 127;
        if (b_col < N) {
            int sg = k / 32 + half;
            if (sg < Kg) sb_val = B_scale[(long long)b_col * Kg + sg];
        }

        unsigned sa = (unsigned)sa_val;
        unsigned sb = (unsigned)sb_val;

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // === Output store: salykova pattern ===
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = mb + half * 4 + j + i * 8;
            int c = nb + col;
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
        _mod = load_inline(name="salykova_v4_fix", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
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

    # A: use aiter's quantization (proven correct)
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_fp4_u8 = A_fp4.view(torch.uint8)
    A_scale_u8 = A_scale.view(torch.uint8)

    # B: unshuffle scales
    Bs = _unshuffle(B_scale_sh, N, K)
    Bq = B_q.view(torch.uint8)

    return mod.launch_gemm(A_fp4_u8, A_scale_u8, Bq, Bs, M, N, K)

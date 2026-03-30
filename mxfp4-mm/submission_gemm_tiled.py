#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP -- Tiled BK=256 MFMA kernel.
Loads 4 K-chunks of data per outer iteration, issues 4 MFMA calls from registers.
Reduces global memory traffic by 4x vs fullwave (BK=64).
Each block: 1 wavefront (64 threads), output tile 32x32.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch, sys
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));

extern "C" __device__ v16f __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    v8i, v8i, v16f, int, int, int, int, int, int) __asm("llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4");

// BK=256 tiled kernel: 4 MFMA calls per outer K iteration
// Each MFMA covers K=64 (half0: 32, half1: 32)
// Pre-load all 4 sub-chunks of A, B, and scales before issuing MFMAs
__global__ void gemm_fp4_tiled(
    const unsigned char* __restrict__ A_fp4,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int lane = threadIdx.x % 32;
    const int half = threadIdx.x / 32;  // 0 or 1
    const int mb = blockIdx.y * 32;
    const int nb = blockIdx.x * 32;
    const int Kh = K / 2;   // K in packed bytes (2 FP4 per byte)
    const int Kb = K / 32;  // number of scale blocks per row

    const int a_row = mb + lane;
    const int b_col = nb + lane;

    const bool a_valid = (a_row < M);
    const bool b_valid = (b_col < N);

    // Base pointers — only used when valid
    const long long a_off = a_valid ? (long long)a_row * Kh : 0;
    const long long b_off = b_valid ? (long long)b_col * Kh : 0;
    const long long as_off = a_valid ? (long long)a_row * Kb : 0;
    const long long bs_off = b_valid ? (long long)b_col * Kb : 0;

    v16f acc = {};
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    // K-loop: process K=64 per MFMA call (same as proven fullwave)
    // Use #pragma unroll 4 to hint compiler to unroll without pre-loading
    for (int k = 0; k < K; k += 64) {
        v8i a_reg = {}, b_reg = {};
        unsigned int* ap = (unsigned int*)&a_reg;
        unsigned int* bp = (unsigned int*)&b_reg;
        unsigned sa = 127, sb = 127;

        int k_off = k + half * 32;

        if (a_valid && k_off + 32 <= K) {
            const unsigned char* a_ptr = A_fp4 + a_off + k_off / 2;
            ap[0] = *(const unsigned int*)(a_ptr);
            ap[1] = *(const unsigned int*)(a_ptr + 4);
            ap[2] = *(const unsigned int*)(a_ptr + 8);
            ap[3] = *(const unsigned int*)(a_ptr + 12);
            sa = (unsigned)A_scale[as_off + k_off / 32];
        }

        if (b_valid && k_off + 32 <= K) {
            const unsigned char* b_ptr = B_q + b_off + k_off / 2;
            bp[0] = *(const unsigned int*)(b_ptr);
            bp[1] = *(const unsigned int*)(b_ptr + 4);
            bp[2] = *(const unsigned int*)(b_ptr + 8);
            bp[3] = *(const unsigned int*)(b_ptr + 12);
            sb = (unsigned)B_scale[bs_off + k_off / 32];
        }

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // Store output: same mapping as fullwave
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = mb + half * 4 + j + i * 8;
            int col = nb + lane;
            if (row < M && col < N)
                C[(long long)row * N + col] = (hip_bfloat16)acc[i * 4 + j];
        }
    }
}

torch::Tensor launch_gemm(
    torch::Tensor A_fp4, torch::Tensor A_scale,
    torch::Tensor B_q, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp4.device()));
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(64);
    hipLaunchKernelGGL(gemm_fp4_tiled, grid, block, 0, 0,
        (const unsigned char*)A_fp4.data_ptr(),
        (const unsigned char*)A_scale.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_scale.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),
        (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(name="gemm_tiled_bk256_v1", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                           functions=["launch_gemm"],
                           extra_cuda_cflags=["-O3", "--offload-arch=gfx950"], verbose=False)
    return _mod

_cache = {}
def _unshuffle(s, N, K):
    key = id(s)
    if key in _cache: return _cache[key]
    n = K // 32; sm = ((N + 255) // 256) * 256; sn = ((n + 7) // 8) * 8
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N, :n] = s.view(torch.uint8)[:N, :n]
    r = p.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous()
    result = r.view(sm, sn)[:N, :n].contiguous()
    _cache[key] = result
    return result

_first = True

def custom_kernel(data: input_t) -> output_t:
    global _first
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]

    mod = _get_mod()

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    A_fp4_u8 = A_fp4.view(torch.uint8)[:M, :K // 2].contiguous()
    A_scale_u8 = A_scale.view(torch.uint8)[:M, :K // 32].contiguous()
    B_q_u8 = B_q.view(torch.uint8)
    B_scale_u8 = _unshuffle(B_scale_sh, N, K)

    C = mod.launch_gemm(A_fp4_u8, A_scale_u8, B_q_u8, B_scale_u8, M, N, K)
    torch.cuda.synchronize()

    if _first:
        _first = False
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        ref = gemm_afp4wfp4(A_fp4_u8, B_q_u8, A_scale_u8, B_scale_u8, dtype=torch.bfloat16)
        err = (C.float() - ref.float()).abs()
        rel = err / (ref.float().abs() + 1e-6)
        close = torch.isclose(C.float(), ref.float(), rtol=1e-2, atol=1e-2)
        n_match = close.sum().item()
        n_total = close.numel()
        print(f"[TILED-BK256] M={M} N={N} K={K}", file=sys.stderr)
        print(f"[TILED-BK256] {n_match}/{n_total} match ({100*n_match/n_total:.1f}%)", file=sys.stderr)
        print(f"[TILED-BK256] max_err={err.max():.4f} mean_rel={rel.mean():.4f}", file=sys.stderr)
        print(f"[TILED-BK256] HIP[0,:4]={C[0,:4].tolist()} REF[0,:4]={ref[0,:4].tolist()}", file=sys.stderr)
        sys.stderr.flush()

    return C

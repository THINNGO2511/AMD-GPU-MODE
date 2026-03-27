#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP K=32 with aiter quant: Process K in chunks of 32 (not 64).
Uses dynamic_mxfp4_quant for A quantization.
Both halves load identical K=32 data, divide result by 2 at end.
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
#include <cstdint>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));

extern "C" __device__ v16f __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    v8i, v8i, v16f, int, int, int, int, int, int) __asm("llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4");

__global__ __launch_bounds__(64)
void gemm_k32_kernel(
    const uint8_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int Kp = K / 2;       // number of uint8 (fp4x2) per row in K dim
    const int Kg = K / 32;      // number of scale groups
    const int lane = threadIdx.x % 32;
    const int half = threadIdx.x / 32;
    const int mb = blockIdx.y * 32;
    const int nb = blockIdx.x * 32;

    const int a_row = mb + lane;
    const int b_col = nb + lane;

    v16f acc = {};

    // Process K in chunks of 32
    // Each MFMA call: 16 bytes of data in register bytes 0-15, zeros in bytes 16-31
    // BOTH halves (half=0 and half=1) load the SAME 16 bytes and SAME scale
    // This means both halves contribute identical results -> 2x the correct value
    // We divide by 2.0 at the end
    for (int k = 0; k < K; k += 32) {
        v8i a_reg = {};
        v8i b_reg = {};
        uint8_t* a_bytes = (uint8_t*)&a_reg;
        uint8_t* b_bytes = (uint8_t*)&b_reg;

        // A: load 16 bytes (32 FP4 values for K block k..k+31)
        // Both half=0 and half=1 load the SAME data into bytes 0-15
        if (a_row < M) {
            int a_off = a_row * Kp + k / 2;
            const uint8_t* a_src = A_fp4 + a_off;
            for (int i = 0; i < 16; i++) {
                a_bytes[i] = a_src[i];
            }
        }
        // bytes 16-31 are already zero from initialization

        // B: load 16 bytes (32 FP4 values for K block k..k+31)
        // Both half=0 and half=1 load the SAME data into bytes 0-15
        if (b_col < N) {
            int b_off = b_col * Kp + k / 2;
            const uint8_t* b_src = B_q + b_off;
            for (int i = 0; i < 16; i++) {
                b_bytes[i] = b_src[i];
            }
        }
        // bytes 16-31 are already zero from initialization

        // Scale: single E8M0 value for this K=32 block
        // Same scale for both halves
        unsigned sa = 127u;
        if (a_row < M) {
            int sg = k / 32;
            sa = (unsigned)A_scale[a_row * Kg + sg];
        }
        unsigned sb = 127u;
        if (b_col < N) {
            int sg = k / 32;
            sb = (unsigned)B_scale[b_col * Kg + sg];
        }

        // MFMA: cbsz_a=4 (FP4), cbsz_b=4 (FP4), cbsz=0
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    // Store output as bf16, dividing by 2.0 because both halves contributed
    // identical data (exact 2x doubling)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = mb + half * 4 + j + i * 8;
            int col = nb + lane;
            if (row < M && col < N) {
                C[(long long)row * N + col] = (hip_bfloat16)(acc[i * 4 + j] * 0.5f);
            }
        }
    }
}

torch::Tensor launch_gemm(torch::Tensor A_fp4, torch::Tensor A_scale,
                           torch::Tensor B_q, torch::Tensor B_scale,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp4.device()));
    dim3 g((N + 31) / 32, (M + 31) / 32);
    dim3 b(64);
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
        _mod = load_inline(
            name="gemm_k32_aiter_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_gemm"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod

_sc = {}
def _unshuffle(s, N, K):
    key = id(s)
    if key in _sc:
        return _sc[key]
    n = K // 32
    sm = ((N + 255) // 256) * 256
    sn = ((n + 7) // 8) * 8
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N, :n] = s.view(torch.uint8)[:N, :n]
    r = p.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous()
    result = r.view(sm, sn)[:N, :n]
    _sc[key] = result
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

    Bs = _unshuffle(B_scale_sh, N, K)

    C_hip = mod.launch_gemm(
        A_fp4.view(torch.uint8),
        A_scale.view(torch.uint8),
        B_q.view(torch.uint8),
        Bs,
        M, N, K,
    )

    if _first:
        _first = False
        torch.cuda.synchronize()
        try:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            ref = gemm_afp4wfp4(
                A_fp4.view(torch.uint8),
                B_q.view(torch.uint8),
                A_scale.view(torch.uint8),
                Bs,
                dtype=torch.bfloat16,
            )
            err = (C_hip.float() - ref.float()).abs()
            rel = err / (ref.float().abs() + 1e-6)
            close = torch.isclose(C_hip.float(), ref.float(), rtol=1e-2, atol=1e-2)
            n_match = close.sum().item()
            n_total = close.numel()
            print(f"[K32_AITER] M={M} N={N} K={K}", file=sys.stderr)
            print(f"[K32_AITER] {n_match}/{n_total} match ({100.0*n_match/n_total:.1f}%)", file=sys.stderr)
            print(f"[K32_AITER] max_err={err.max():.4f} mean_rel={rel.mean():.4f}", file=sys.stderr)
            print(f"[K32_AITER] ref[0,:4]={ref[0,:4].tolist()} hip[0,:4]={C_hip[0,:4].tolist()}", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            print(f"[K32_AITER] diag error: {e}", file=sys.stderr)
            sys.stderr.flush()

    return C_hip

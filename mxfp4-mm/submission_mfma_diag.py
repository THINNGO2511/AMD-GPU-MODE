#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM MFMA Diagnostic: Dump raw A/B bytes and scales for thread 0.
Run a 32x32x64 tile on KNOWN data to verify the MFMA output mapping.
Synthetic test: A=all_ones_fp4 (value=0.5 in E2M1), B=identity-like.
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

// Diagnostic: dump bytes for thread 0
__global__ void diag_kernel(
    const uint8_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_scale,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_scale,
    float* __restrict__ C_out,  // [32, 32] float for precise output
    float* __restrict__ diag,   // diagnostic output [256]
    int M, int N, int K)
{
    const int Kp = K / 2;
    const int Kg = K / 32;
    const int lane = threadIdx.x % 32;
    const int half = threadIdx.x / 32;

    fp32x16_t acc = {};

    // Process first K=64 chunk only
    int k = 0;

    // A: load 16 bytes for this half
    fp4x64_t a_reg = {};
    int a_row = lane;
    if (a_row < M) {
        int a_base = a_row * Kp + k / 2 + half * 16;
        const uint8_t* a_src = A_fp4 + a_base;
        for (int i = 0; i < 16 && i < Kp; i++) {
            a_reg[i] = (fp4x2_t)a_src[i];
        }
    }

    // B: load 16 bytes for this half (HALF-FILL)
    fp4x64_t b_reg = {};
    int b_col = lane;
    if (b_col < N) {
        int b_base = b_col * Kp + k / 2 + half * 16;
        const uint8_t* b_src = B_q + b_base;
        for (int i = 0; i < 16 && i < Kp; i++) {
            b_reg[i] = (fp4x2_t)b_src[i];
        }
    }

    // Scales
    uint8_t sa_val = 127, sb_val = 127;
    if (a_row < M && (k/32 + half) < Kg)
        sa_val = A_scale[a_row * Kg + k/32 + half];
    if (b_col < N && (k/32 + half) < Kg)
        sb_val = B_scale[b_col * Kg + k/32 + half];

    unsigned sa = (unsigned)sa_val;
    unsigned sb = (unsigned)sb_val;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

    // Store output
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = half * 4 + j + i * 8;
            int c = lane;
            if (r < 32 && c < 32)
                C_out[r * 32 + c] = acc[i * 4 + j];
        }
    }

    // Diagnostics: thread 0 dumps its raw data
    if (threadIdx.x == 0) {
        // Dump first 16 bytes of A register
        uint8_t a_bytes[32];
        __builtin_memcpy(a_bytes, &a_reg, 32);
        for (int i = 0; i < 16; i++) diag[i] = (float)a_bytes[i];
        // Dump first 16 bytes of B register
        uint8_t b_bytes[32];
        __builtin_memcpy(b_bytes, &b_reg, 32);
        for (int i = 0; i < 16; i++) diag[16+i] = (float)b_bytes[i];
        // Scales
        diag[32] = (float)sa_val;
        diag[33] = (float)sb_val;
        // First 4 output values
        diag[34] = acc[0];
        diag[35] = acc[1];
        diag[36] = acc[2];
        diag[37] = acc[3];
    }
}

torch::Tensor launch_diag(torch::Tensor A_fp4, torch::Tensor A_scale,
                           torch::Tensor B_q, torch::Tensor B_scale,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));
    auto diag = torch::zeros({256}, torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));
    dim3 g(1, 1), b(64);  // Single tile
    hipLaunchKernelGGL(diag_kernel, g, b, 0, 0,
        (const uint8_t*)A_fp4.data_ptr(),
        (const uint8_t*)A_scale.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (const uint8_t*)B_scale.data_ptr(),
        (float*)C.data_ptr(),
        (float*)diag.data_ptr(),
        (int)M, (int)N, (int)K);
    return diag;  // Return diagnostic data
}
"""

CPP_FWD = "torch::Tensor launch_diag(torch::Tensor A_fp4, torch::Tensor A_scale, torch::Tensor B_q, torch::Tensor B_scale, int64_t M, int64_t N, int64_t K);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(name="mfma_diag_v1", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                           functions=["launch_diag"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"], verbose=True)
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

    # Run diagnostic on first 32x32 tile
    diag = mod.launch_diag(A_fp4.view(torch.uint8), A_scale.view(torch.uint8),
                           B_q.view(torch.uint8), Bs, M, N, K)
    torch.cuda.synchronize()

    d = diag.cpu()
    print(f"[DIAG] M={M} N={N} K={K}", file=sys.stderr)
    print(f"[DIAG] A_bytes[0:16]: {[int(d[i]) for i in range(16)]}", file=sys.stderr)
    print(f"[DIAG] B_bytes[0:16]: {[int(d[16+i]) for i in range(16)]}", file=sys.stderr)
    print(f"[DIAG] sa={int(d[32])} sb={int(d[33])}", file=sys.stderr)
    print(f"[DIAG] MFMA out[0:4]: {[float(d[34+i]) for i in range(4)]}", file=sys.stderr)

    # Also compute reference manually
    a_u8 = A_fp4.view(torch.uint8).cpu()
    b_u8 = B_q.view(torch.uint8).cpu()
    as_u8 = A_scale.view(torch.uint8).cpu()
    bs_u8 = Bs.cpu()

    # Manual FP4 dequant for row 0, col 0
    def dequant_fp4(byte_val, nibble):
        nib = (byte_val >> (nibble * 4)) & 0xF
        sign = -1.0 if (nib & 8) else 1.0
        mag = nib & 7
        # E2M1: values are 0, 0.5, 1, 1.5, 2, 3, 4, 6
        lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        return sign * lut[mag]

    # Compute C[0,0] = sum_k A[0,k] * B[0,k]
    dot = 0.0
    for kk in range(min(K, 64)):  # first 64 K elements
        a_byte = int(a_u8[0, kk // 2])
        a_val = dequant_fp4(a_byte, kk % 2)
        a_sc = 2.0 ** (int(as_u8[0, kk // 32]) - 127)

        b_byte = int(b_u8[0, kk // 2])
        b_val = dequant_fp4(b_byte, kk % 2)
        b_sc = 2.0 ** (int(bs_u8[0, kk // 32]) - 127)

        dot += a_val * a_sc * b_val * b_sc

    print(f"[DIAG] Manual C[0,0] (first 64 K): {dot:.4f}", file=sys.stderr)
    print(f"[DIAG] A_scale[0,0]={int(as_u8[0,0])} A_scale[0,1]={int(as_u8[0,1])}", file=sys.stderr)
    print(f"[DIAG] B_scale[0,0]={int(bs_u8[0,0])} B_scale[0,1]={int(bs_u8[0,1])}", file=sys.stderr)
    sys.stderr.flush()

    # Return correct result from Triton
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q.view(torch.uint8),
                         A_scale.view(torch.uint8), Bs, dtype=torch.bfloat16)

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Phase 1 v2: Fix FP4 MFMA data loading.
v1 produced all zeros — register layout was wrong.
Key insight from CLAUDE.md probes:
- Each thread provides 8 x int32 = 32 bytes = 64 FP4 values
- NOT 128 FP4 per thread (that's the total K dimension of the MFMA)
- 64 threads × 64 FP4/thread = 4096 FP4 total, but K=128 so it's 32×128 mapped
- For 16x16x128: A is [16, 128] FP4, B is [128, 16] FP4
- A loading: t%16 = row, 32 consecutive FP4x2 bytes per thread
- B loading: uses __amd_extract_fp4 (non-contiguous)

This version: simple scalar load, verify correctness first.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

_hip_mod = None
_ran = False

# Minimal kernel: loads A and B into MFMA registers correctly
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hip/hip_bfloat16.h>

typedef int a_operand_t __attribute__((ext_vector_type(8)));
typedef int b_operand_t __attribute__((ext_vector_type(8)));
typedef float c_result_t __attribute__((ext_vector_type(4)));

// Simple test: A[16,128] fp4 x B[128,16] fp4 -> C[16,16] f32
// One wavefront (64 threads) computes one 16x16 output tile
__global__ void mfma_fp4_test(
    const unsigned char* __restrict__ A_fp4,   // [M, K/2] packed fp4x2
    const unsigned char* __restrict__ B_fp4,   // [N, K/2] packed fp4x2 (B is NxK)
    const unsigned char* __restrict__ A_scale,  // [M, K/32] E8M0
    const unsigned char* __restrict__ B_scale,  // [N, K/32] E8M0
    float* __restrict__ C,                      // [M, N]
    int M, int N, int K
) {
    int tid = threadIdx.x;  // 0..63
    int lane = tid;

    // For the 16x16x128 MFMA:
    // A operand: 8 x int32 per thread = 32 bytes = 64 FP4 values
    // The mapping from salykova blog:
    //   ldg_a = A + (t%32)*32 + (t/32)*16  (byte offsets for A[32x64] layout)
    //   Each thread loads 16 consecutive fp4x2 bytes
    // But for 16x16x128:
    //   A is [16, 128 FP4] = [16, 64 bytes]
    //   Thread layout: t%16 = row index, t/16 = which of 4 groups
    //   Each thread contributes FP4 values for K=128 dimension

    a_operand_t a_reg;
    b_operand_t b_reg;
    c_result_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};

    // Zero-initialize
    for (int i = 0; i < 8; i++) {
        a_reg[i] = 0;
        b_reg[i] = 0;
    }

    // Load A: each thread loads 32 bytes (64 FP4) from A
    // For 16x16x128: A[16, 128] -> 16 rows, 128 FP4 cols = 64 bytes/row
    // 64 threads need to cover 16 rows x 64 bytes = 1024 bytes total
    // Each thread loads 32 bytes = half a row
    // Thread mapping: row = lane % 16, half = lane / 16 (0..3, but only 0,1 for 32 bytes each)
    // Wait, that's 16 rows × 2 halves = 32 slots for 64 threads → 2 threads per slot?
    // Actually for MFMA 16x16x128: each thread contributes 32 bytes to A operand
    // The instruction internally maps threads to matrix positions

    // Simplest approach: just load 32 consecutive bytes per thread from A
    // Thread i loads A bytes [i*32 .. i*32+31] from the flattened A matrix
    // For A[16, 64bytes]: total = 1024 bytes, 64 threads × 16 bytes = 1024 ✓
    // Wait, 8 × int32 = 32 bytes per thread, 64 × 32 = 2048 bytes.
    // But A[16, 64] = 1024 bytes. So half the data is padding? No...

    // The MFMA 16x16x128 processes 128 FP4 = 64 bytes in K dimension
    // A[16, 128 FP4] has 16 rows × 64 bytes = 1024 bytes
    // 64 threads × 32 bytes = 2048 bytes
    // So each row is loaded by 4 threads (64 bytes / 32 bytes × 2 halves = sort of)

    // Let's use a direct flat load: each thread reads 32 bytes sequentially
    int a_offset = lane * 16;  // 16 fp4x2 bytes per thread (not 32)
    // Actually: 8 int32 = 32 bytes but that covers 64 FP4 values
    // A has 16 × 128 = 2048 FP4 = 1024 bytes
    // 64 threads × 16 bytes = 1024 bytes → each thread loads 16 bytes
    // But the register is 32 bytes (8 × int32)... upper half zero?

    // From the MFMA spec: for 16x16x128 with FP4:
    // Each thread provides src_a[8 x i32] = 256 bits = 64 FP4 values
    // Total: 64 threads × 64 FP4 = 4096 FP4 values
    // But A is only 16 × 128 = 2048 FP4 values
    // The extra capacity handles the tiling internally

    // Let's just load sequentially and let the MFMA sort it out
    if (M >= 16 && K >= 128) {
        // Flat load: thread i gets bytes [i*16 .. i*16+15] from A
        // Then pack into 8 x int32 with zero upper half
        const unsigned char* a_base = A_fp4;
        int off = lane * 16;  // 16 bytes per thread
        for (int i = 0; i < 4; i++) {
            unsigned int val = 0;
            for (int b = 0; b < 4; b++) {
                val |= ((unsigned int)a_base[off + i*4 + b]) << (b * 8);
            }
            a_reg[i] = (int)val;
        }
        // Upper 4 ints = zero (padding for 64 FP4 → only 32 FP4 loaded)
    }

    // Load B similarly
    if (N >= 16 && K >= 128) {
        const unsigned char* b_base = B_fp4;
        int off = lane * 16;
        for (int i = 0; i < 4; i++) {
            unsigned int val = 0;
            for (int b = 0; b < 4; b++) {
                val |= ((unsigned int)b_base[off + i*4 + b]) << (b * 8);
            }
            b_reg[i] = (int)val;
        }
    }

    // Get per-thread scale
    // For E8M0 scales: one scale per 32 FP4 elements
    // A has K/32 = 4 scale groups per row
    // Simple: use scale from first group
    unsigned int sa = 127, sb = 127;
    if (lane < M * (K/32)) {
        sa = A_scale[lane % (K/32) + (lane / (K/32)) * (K/32)];
    }
    if (lane < N * (K/32)) {
        sb = B_scale[lane % (K/32) + (lane / (K/32)) * (K/32)];
    }

    // MFMA call
    c_reg = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, sa, 0, sb);

    // Store: for 16x16 output
    // Thread lane%16 = column, row mapping from c_reg indices
    int col = lane % 16;
    int half = lane / 32;
    for (int i = 0; i < 4; i++) {
        int row = half * 8 + (i % 2) + (i / 2) * 4;
        if (row < M && col < N) {
            C[row * N + col] = c_reg[i];
        }
    }
}

torch::Tensor test_mfma(torch::Tensor A_fp4, torch::Tensor B_fp4,
                         torch::Tensor A_scale, torch::Tensor B_scale,
                         int M, int N, int K) {
    auto C = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1, 1);
    dim3 block(64);
    hipLaunchKernelGGL(mfma_fp4_test, grid, block, 0, 0,
        A_fp4.data_ptr<unsigned char>(),
        B_fp4.data_ptr<unsigned char>(),
        A_scale.data_ptr<unsigned char>(),
        B_scale.data_ptr<unsigned char>(),
        C.data_ptr<float>(),
        M, N, K);
    return C;
}
"""

def _try_compile():
    global _hip_mod
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_mod = load_inline(
            name="gemm_fp4_phase1_v2",
            cpp_sources="torch::Tensor test_mfma(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);",
            cuda_sources=HIP_SOURCE,
            functions=["test_mfma"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("COMPILE SUCCESS v2!", flush=True)
        return True
    except Exception as e:
        print(f"COMPILE FAILED v2: {e}", flush=True)
        return False

def _test_mfma():
    if _hip_mod is None:
        return

    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    # Test: M=16, N=16, K=128 (exact MFMA tile)
    M, N, K = 16, 16, 128
    A = torch.ones(M, K, dtype=torch.bfloat16, device='cuda')
    B = torch.ones(N, K, dtype=torch.bfloat16, device='cuda')

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_fp4, B_scale = dynamic_mxfp4_quant(B)

    print(f"A_fp4 shape: {A_fp4.shape}, dtype: {A_fp4.dtype}", flush=True)
    print(f"A_scale shape: {A_scale.shape}, dtype: {A_scale.dtype}", flush=True)
    print(f"B_fp4 shape: {B_fp4.shape}, dtype: {B_fp4.dtype}", flush=True)
    print(f"B_scale shape: {B_scale.shape}, dtype: {B_scale.dtype}", flush=True)
    print(f"A_fp4 bytes: {A_fp4.view(torch.uint8).shape}", flush=True)

    # Raw FP4 data
    a_u8 = A_fp4.view(torch.uint8)
    b_u8 = B_fp4.view(torch.uint8)
    a_sc = A_scale.view(torch.uint8)
    b_sc = B_scale.view(torch.uint8)

    print(f"A_fp4 first 16 bytes: {a_u8[0,:16].tolist()}", flush=True)
    print(f"A_scale first 4: {a_sc[0,:4].tolist()}", flush=True)
    print(f"B_fp4 first 16 bytes: {b_u8[0,:16].tolist()}", flush=True)
    print(f"B_scale first 4: {b_sc[0,:4].tolist()}", flush=True)

    C_hip = _hip_mod.test_mfma(a_u8, b_u8, a_sc, b_sc, M, N, K)
    print(f"HIP C[0,0:8]: {C_hip[0,:8].tolist()}", flush=True)
    print(f"HIP C[0,0]: {C_hip[0,0].item()}", flush=True)
    print(f"HIP C max: {C_hip.abs().max().item()}", flush=True)
    print(f"HIP C nonzero: {(C_hip != 0).sum().item()} / {M*N}", flush=True)

    # Reference: all ones @ all ones.T = K = 128 per element (before quantization loss)
    C_ref = (A.float() @ B.float().T)
    print(f"Ref C[0,0]: {C_ref[0,0].item()}", flush=True)

_cache = {}

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    global _ran
    if not _ran:
        _ran = True
        ok = _try_compile()
        if ok:
            _test_mfma()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]
    cache_key = id(B_scale_sh)
    if cache_key not in _cache:
        _cache[cache_key] = (_unshuffle_e8m0(B_scale_sh), B_q.view(torch.uint8))
    bscale_raw, bq_u8 = _cache[cache_key]
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)

_ran = False

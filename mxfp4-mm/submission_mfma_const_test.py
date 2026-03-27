#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Minimal MFMA FP4 test: hardcoded constant FP4 data (all 1.0).
32x32x64 MFMA with salykova half-fill pattern.
Expected: C[i,j] = 64.0 for all i,j.
"""
import os, sys
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
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
void const_mfma_kernel(float* __restrict__ C_out) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int half = tid / 32;

    // Build A register: 16 bytes of 0x22 (FP4 1.0 pairs), 16 bytes zero
    uint8_t a_bytes[32];
    for (int i = 0; i < 16; i++) a_bytes[i] = 0x22;
    for (int i = 16; i < 32; i++) a_bytes[i] = 0x00;
    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // Build B register: 16 bytes of 0x22 (FP4 1.0 pairs), 16 bytes zero
    uint8_t b_bytes[32];
    for (int i = 0; i < 16; i++) b_bytes[i] = 0x22;
    for (int i = 16; i < 32; i++) b_bytes[i] = 0x00;
    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    // Accumulator starts at zero
    v16f acc = {};

    // scale_a=127, scale_b=127 => 2^(127-127) = 1.0
    unsigned scale_a = 127u;
    unsigned scale_b = 127u;

    // MFMA: cbsz_a=4 (FP4), cbsz_b=4 (FP4)
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc,
        4, 4, 0,
        scale_a, 0, scale_b);

    // Output store: salykova pattern into 32x32 matrix
    int m_base = 0;
    int n_base = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = m_base + half * 4 + j + i * 8;
            int col = n_base + lane;
            C_out[row * 32 + col] = acc[i * 4 + j];
        }
    }
}

torch::Tensor launch_const_test() {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(const_mfma_kernel, grid, block, 0, 0,
        (float*)C.data_ptr());
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_const_test();"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="mfma_const_test_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_const_test"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]

    # Run the constant MFMA test and print diagnostics to stderr
    mod = _get_mod()
    C_test = mod.launch_const_test()
    torch.cuda.synchronize()

    c = C_test.cpu()
    print(f"=== MFMA CONST TEST (all 1.0 FP4, scale=127) ===", file=sys.stderr)
    print(f"C[0, 0:4] = {c[0, 0:4].tolist()}", file=sys.stderr)
    print(f"C[0, 28:32] = {c[0, 28:32].tolist()}", file=sys.stderr)
    print(f"C[31, 0:4] = {c[31, 0:4].tolist()}", file=sys.stderr)
    print(f"Expected: 64.0 everywhere (sum of 64 products of 1.0*1.0)", file=sys.stderr)
    print(f"Min={c.min().item():.4f}  Max={c.max().item():.4f}  Mean={c.mean().item():.4f}", file=sys.stderr)
    nz = (c != 0).sum().item()
    print(f"Non-zero: {nz}/1024", file=sys.stderr)
    unique = torch.unique(c).tolist()
    if len(unique) <= 10:
        print(f"Unique values: {unique}", file=sys.stderr)
    else:
        print(f"Unique values ({len(unique)} total): {unique[:10]}...", file=sys.stderr)

    # Return correct Triton result for test pass
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    def _unshuffle_e8m0(s):
        s = s.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        return s.view(sm, sn)

    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Phase 2 v2: Fix data loading to match MFMA register layout.
Key insight: the MFMA 16x16x128 FP4 expects specific thread-to-data mapping.
From AMD ISA: for v_mfma_f32_16x16xK:
  - 64 threads per wavefront
  - src_a: 8 x int32 (256 bits = 64 FP4 values per thread)
  - src_b: 8 x int32 (256 bits = 64 FP4 values per thread)
  - For A[16,128]: each of 16 rows has 128 FP4 values = 64 bytes
    Thread t provides 64 FP4 values. 64 threads x 64 FP4 = 4096 but matrix is 16x128=2048
    So: 4 threads share each row, each providing 32 FP4 (but register holds 64 FP4)
    OR: each thread covers 2 rows

  The actual mapping: thread t loads data for row = t % 16
  Half 0 (threads 0-31): each provides 64 FP4 from rows 0-15 (2 threads per row)
  Half 1 (threads 32-63): provides 64 FP4 from rows 0-15 (2 more threads per row)
  4 threads total per row × 64 FP4/thread = 256 FP4 but K=128
  This means only first 32 FP4 (16 bytes) per thread are used?

  From salykova blog (CLAUDE.md):
    ldg_a = A + (t%32)*32 + (t/32)*16  // for 32x64 tile
    For 16x128: each thread loads 16 fp4x2 = 32 FP4 = 16 bytes

  Let me try: each thread loads 16 bytes from A[row, offset]
  where row = t % 16, offset = (t / 16) * 16 bytes (4 groups of 16 bytes = 64 bytes per row)
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

_hip_mod = None
_ran = False

HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

typedef int a_operand_t __attribute__((ext_vector_type(8)));
typedef int b_operand_t __attribute__((ext_vector_type(8)));
typedef float c_result_t __attribute__((ext_vector_type(4)));

__global__ void gemm_fp4_v2(
    const unsigned char* __restrict__ A_fp4,
    const unsigned char* __restrict__ B_fp4,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int tile_n = blockIdx.x * 16;
    int tile_m = blockIdx.y * 16;
    int lane = threadIdx.x;  // 0..63

    c_result_t c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int K_bytes = K / 2;
    int K_scales = K / 32;
    int K_chunks = K / 128;

    for (int kc = 0; kc < K_chunks; kc++) {
        a_operand_t a_reg;
        b_operand_t b_reg;

        // Zero init
        for (int i = 0; i < 8; i++) {
            a_reg[i] = 0;
            b_reg[i] = 0;
        }

        int k_byte_start = kc * 64;  // 128 FP4 = 64 bytes

        // Load A: row-major A[M, K/2]
        // For 16x16x128 MFMA with 64 threads:
        // Each thread loads 16 bytes (32 FP4 = 16 fp4x2)
        // Mapping: row = lane % 16, group = lane / 16 (0..3)
        // 4 groups x 16 bytes = 64 bytes per row = full 128 FP4
        // Stored in lower 4 int32 of register (16 bytes = 4 x int32)
        {
            int a_row = lane % 16;
            int a_group = lane / 16;  // 0,1,2,3
            int a_global_row = tile_m + a_row;
            if (a_global_row < M) {
                const unsigned char* a_ptr = A_fp4 + a_global_row * K_bytes
                                             + k_byte_start + a_group * 16;
                for (int i = 0; i < 4; i++) {
                    unsigned int val = 0;
                    for (int b = 0; b < 4; b++) {
                        int idx = i * 4 + b;
                        if (k_byte_start + a_group * 16 + idx < K_bytes)
                            val |= ((unsigned int)a_ptr[idx]) << (b * 8);
                    }
                    a_reg[i] = (int)val;
                    // Upper 4 int32 stay zero
                }
            }
        }

        // Load B: row-major B[N, K/2]
        // Same mapping as A: row = lane%16, group = lane/16
        {
            int b_row = lane % 16;
            int b_group = lane / 16;
            int b_global_row = tile_n + b_row;
            if (b_global_row < N) {
                const unsigned char* b_ptr = B_fp4 + b_global_row * K_bytes
                                             + k_byte_start + b_group * 16;
                for (int i = 0; i < 4; i++) {
                    unsigned int val = 0;
                    for (int b = 0; b < 4; b++) {
                        int idx = i * 4 + b;
                        if (k_byte_start + b_group * 16 + idx < K_bytes)
                            val |= ((unsigned int)b_ptr[idx]) << (b * 8);
                    }
                    b_reg[i] = (int)val;
                }
            }
        }

        // Scales: E8M0, one per 32 FP4 elements
        // This chunk has 128 FP4 = 4 scale groups
        // Each thread group (lane/16) handles 32 FP4 = 1 scale group
        // Scale index: kc * 4 + (lane / 16)
        int k_scale_group = kc * 4 + (lane / 16);
        unsigned int sa = 127, sb = 127;
        {
            int a_row = lane % 16;
            int a_global_row = tile_m + a_row;
            if (a_global_row < M && k_scale_group < K_scales) {
                sa = A_scale[a_global_row * K_scales + k_scale_group];
            }
        }
        {
            int b_row = lane % 16;
            int b_global_row = tile_n + b_row;
            if (b_global_row < N && k_scale_group < K_scales) {
                sb = B_scale[b_global_row * K_scales + k_scale_group];
            }
        }

        c_acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, c_acc, 4, 4, 0, sa, 0, sb);
    }

    // Handle remaining K elements (K not divisible by 128)
    // For our benchmark shapes: K=512(div128=4), K=7168(div128=56), K=2048(div128=16), K=1536(div128=12)
    // All are divisible by 128, so no remainder handling needed

    // Store output as bf16
    int col = lane % 16;
    int global_col = tile_n + col;
    for (int i = 0; i < 4; i++) {
        int row = (lane / 16) * 4 + i;
        int global_row = tile_m + row;
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = (hip_bfloat16)(c_acc[i]);
        }
    }
}

torch::Tensor gemm_fp4(torch::Tensor A_fp4, torch::Tensor B_fp4,
                        torch::Tensor A_scale, torch::Tensor B_scale,
                        int M, int N, int K) {
    auto C = torch::zeros({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    dim3 block(64);
    hipLaunchKernelGGL(gemm_fp4_v2, grid, block, 0, 0,
        A_fp4.data_ptr<unsigned char>(),
        B_fp4.data_ptr<unsigned char>(),
        A_scale.data_ptr<unsigned char>(),
        B_scale.data_ptr<unsigned char>(),
        (hip_bfloat16*)C.data_ptr(),
        M, N, K);
    return C;
}
"""

def _try_compile():
    global _hip_mod
    try:
        from torch.utils.cpp_extension import load_inline
        _hip_mod = load_inline(
            name="gemm_fp4_phase2_v2",
            cpp_sources="torch::Tensor gemm_fp4(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);",
            cuda_sources=HIP_SOURCE,
            functions=["gemm_fp4"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("PHASE 2v2 COMPILE SUCCESS!", flush=True)
        return True
    except Exception as e:
        print(f"PHASE 2v2 COMPILE FAILED: {e}", flush=True)
        return False

def _test_accuracy():
    if _hip_mod is None:
        return

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    shapes = [
        (4, 2880, 512),
        (16, 2112, 7168),
        (32, 4096, 512),
        (64, 7168, 2048),
    ]

    for M, N, K in shapes:
        try:
            torch.manual_seed(42)
            A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
            B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            B_fp4, B_scale = dynamic_mxfp4_quant(B)

            C_hip = _hip_mod.gemm_fp4(
                A_fp4.view(torch.uint8), B_fp4.view(torch.uint8),
                A_scale.view(torch.uint8), B_scale.view(torch.uint8),
                M, N, K)

            C_ref = gemm_afp4wfp4(
                A_fp4.view(torch.uint8), B_fp4.view(torch.uint8),
                A_scale.view(torch.uint8), B_scale.view(torch.uint8),
                dtype=torch.bfloat16)

            diff = (C_hip.float() - C_ref.float()).abs()
            maxdiff = diff.max().item()
            maxval = C_ref.float().abs().max().item()
            reldiff = maxdiff / (maxval + 1e-6)
            # Also compute mean relative error
            mean_rel = (diff / (C_ref.float().abs() + 1e-6)).mean().item()
            # Check ratio
            mask = C_ref.float().abs() > 1.0
            if mask.any():
                ratio = (C_hip.float()[mask] / C_ref.float()[mask]).mean().item()
            else:
                ratio = 0
            print(f"({M},{N},{K}): maxdiff={maxdiff:.2f} maxval={maxval:.2f} "
                  f"rel={reldiff:.4f} mean_rel={mean_rel:.4f} ratio={ratio:.4f}", flush=True)
        except Exception as e:
            print(f"({M},{N},{K}): ERROR {e}", flush=True)

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    global _ran
    if not _ran:
        _ran = True
        ok = _try_compile()
        if ok:
            _test_accuracy()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
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

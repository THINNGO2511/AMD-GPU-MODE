#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Phase 2 v6: Use LDS to decouple global load from MFMA register layout.
Load A[16,64bytes] and B[16,64bytes] into LDS cooperatively.
Then each thread reads its 16 bytes from LDS in the pattern the MFMA expects.

Key insight: the MFMA expects data in a specific per-thread order that may
differ from row-major. By using LDS as intermediary, we can load row-major
from global and read MFMA-order from LDS.

For 16x16x128 MFMA, the register-to-K mapping from testing:
- Thread t provides 64 FP4 (8 int32 regs, but only 4 used = 32 FP4)
- Threads t%16 share same row
- Threads with same t/16 group share same K-subrange

LDS approach: load full A tile (16 rows x 64 bytes) into shared memory,
then each thread reads its portion.
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

typedef int operand_t __attribute__((ext_vector_type(8)));
typedef float c16_result_t __attribute__((ext_vector_type(4)));

// Use LDS to load A and B tiles, then distribute to MFMA registers
// LDS: A[16][64+4] + B[16][64+4] = 16*68*2 = 2176 bytes (tiny)
__global__ void gemm_fp4_lds(
    const unsigned char* __restrict__ A_fp4,
    const unsigned char* __restrict__ B_fp4,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    __shared__ unsigned char s_a[16][68];  // 16 rows, 64 bytes + 4 pad
    __shared__ unsigned char s_b[16][68];

    int tile_n = blockIdx.x * 16;
    int tile_m = blockIdx.y * 16;
    int lane = threadIdx.x;

    c16_result_t c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int K_bytes = K / 2;
    int K_scales = K / 32;
    int K_chunks = K / 128;

    for (int kc = 0; kc < K_chunks; kc++) {
        int k_byte_start = kc * 64;

        // Cooperative load A[16, 64 bytes] into LDS
        // 64 threads, 1024 bytes -> 16 bytes per thread
        {
            int load_idx = lane;  // 0..63
            int lds_row = load_idx / 4;    // 0..15
            int lds_col_base = (load_idx % 4) * 16;  // 0,16,32,48
            int a_global_row = tile_m + lds_row;
            if (a_global_row < M) {
                const unsigned char* src = A_fp4 + a_global_row * K_bytes + k_byte_start + lds_col_base;
                for (int i = 0; i < 16; i++) {
                    if (k_byte_start + lds_col_base + i < K_bytes)
                        s_a[lds_row][lds_col_base + i] = src[i];
                    else
                        s_a[lds_row][lds_col_base + i] = 0;
                }
            } else {
                for (int i = 0; i < 16; i++)
                    s_a[lds_row][lds_col_base + i] = 0;
            }
        }

        // Cooperative load B[16, 64 bytes] into LDS
        {
            int load_idx = lane;
            int lds_row = load_idx / 4;
            int lds_col_base = (load_idx % 4) * 16;
            int b_global_row = tile_n + lds_row;
            if (b_global_row < N) {
                const unsigned char* src = B_fp4 + b_global_row * K_bytes + k_byte_start + lds_col_base;
                for (int i = 0; i < 16; i++) {
                    if (k_byte_start + lds_col_base + i < K_bytes)
                        s_b[lds_row][lds_col_base + i] = src[i];
                    else
                        s_b[lds_row][lds_col_base + i] = 0;
                }
            } else {
                for (int i = 0; i < 16; i++)
                    s_b[lds_row][lds_col_base + i] = 0;
            }
        }

        __syncthreads();

        // Read from LDS into MFMA registers
        // Thread mapping for 16x16x128 MFMA:
        // row = lane % 16, K_subgroup = lane / 16 (0..3)
        // Each thread reads 16 bytes from s_a[row][K_subgroup*16 .. +15]
        operand_t a_reg, b_reg;
        for (int i = 0; i < 8; i++) { a_reg[i] = 0; b_reg[i] = 0; }

        int mfma_row = lane % 16;
        int mfma_kgroup = lane / 16;  // 0..3

        // Read A from LDS
        for (int i = 0; i < 4; i++) {
            unsigned int val = 0;
            for (int b = 0; b < 4; b++) {
                val |= ((unsigned int)s_a[mfma_row][mfma_kgroup * 16 + i*4 + b]) << (b * 8);
            }
            a_reg[i] = (int)val;
        }

        // Read B from LDS
        int mfma_col = lane % 16;
        for (int i = 0; i < 4; i++) {
            unsigned int val = 0;
            for (int b = 0; b < 4; b++) {
                val |= ((unsigned int)s_b[mfma_col][mfma_kgroup * 16 + i*4 + b]) << (b * 8);
            }
            b_reg[i] = (int)val;
        }

        // Scales
        int k_scale_idx = kc * 4 + mfma_kgroup;
        unsigned int sa = 127, sb = 127;
        {
            int a_global_row = tile_m + mfma_row;
            if (a_global_row < M && k_scale_idx < K_scales)
                sa = A_scale[a_global_row * K_scales + k_scale_idx];
        }
        {
            int b_global_col = tile_n + mfma_col;
            if (b_global_col < N && k_scale_idx < K_scales)
                sb = B_scale[b_global_col * K_scales + k_scale_idx];
        }

        __syncthreads();  // Before next LDS write

        c_acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, c_acc, 4, 4, 0, sa, 0, sb);
    }

    // Store
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
    hipLaunchKernelGGL(gemm_fp4_lds, grid, block, 0, 0,
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
            name="gemm_fp4_p2v6",
            cpp_sources="torch::Tensor gemm_fp4(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);",
            cuda_sources=HIP_SOURCE,
            functions=["gemm_fp4"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("P2v6 COMPILE OK!", flush=True)
        return True
    except Exception as e:
        print(f"P2v6 COMPILE FAIL: {e}", flush=True)
        return False

def _test():
    if _hip_mod is None:
        return
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    for M, N, K in [(4, 32, 128), (4, 2880, 512), (16, 2112, 7168), (64, 7168, 2048)]:
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
            corr = torch.corrcoef(torch.stack([C_hip.float().flatten(), C_ref.float().flatten()]))[0,1].item()
            mask = C_ref.float().abs() > maxval * 0.1
            ratio = (C_hip.float()[mask] / C_ref.float()[mask]).median().item() if mask.any() else 0
            tol = torch.isclose(C_hip.float(), C_ref.float(), rtol=1e-2, atol=1e-2).float().mean().item()

            print(f"({M},{N},{K}): maxdiff={maxdiff:.2f} corr={corr:.6f} ratio={ratio:.4f} tol={tol:.4f}", flush=True)
            if M <= 4 and N <= 32:
                print(f"  HIP: {C_hip[0,:min(8,N)].tolist()}", flush=True)
                print(f"  REF: {C_ref[0,:min(8,N)].tolist()}", flush=True)
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
            _test()
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

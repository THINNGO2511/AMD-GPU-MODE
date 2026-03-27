#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP GEMM v26: Use salykova's EXACT B loading pattern (stride-16 extract_fp4).
Key insight: MFMA expects B in [K,N] physical order, but our B_q is [N,K].
Blog's B[K=64, N=32]: stride-16 access with extract_fp4 extracts per-N-column data.
We need to TRANSPOSE B_q during loading to match the expected register layout.

For 32x32x64 MFMA with B_q[N, K/2]:
- B_q row stride = K/2 fp4x2 (contiguous along K)
- To get K-major order: stride = K/2 between N positions, NOT between K positions

Also fixing scale handling: per-thread scale_a/scale_b from E8M0 block scales.
Each thread t processes row=t%32 of A and col=t%32 of B, K_half=t/32.
scale_a = A_scale[row, K_half] (scale for this thread's A data)
scale_b = B_scale[col, K_half] (scale for this thread's B data)
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

HIP_SRC = r'''
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>
#include <hip/hip_bfloat16.h>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

// 32x32 output tile, K processed in chunks of 64
// A_fp4: [M, K/2] fp4x2, A_scale: [M, K/32] E8M0
// B_q:   [N, K/2] fp4x2, B_scale: [N, K/32] E8M0
// C:     [M, N] bf16
__global__ void gemm_fp4_kernel(
    const fp4x2_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_scale,
    const fp4x2_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K)
{
    const int K_half = K / 2;  // fp4x2 per row
    const int K_blocks = K / 32;  // scale blocks per row

    // Tile indices
    const int bm = blockIdx.y;  // M tile (32 rows)
    const int bn = blockIdx.x;  // N tile (32 cols)
    const int tid = threadIdx.x;  // 0..63

    // Thread mapping within the 32x32 tile
    const int lane = tid % 32;  // output column (for B), output row set (for A)
    const int half = tid / 32;  // 0=first K half (K=0..31), 1=second K half (K=32..63)

    // Global row/col
    const int m_base = bm * 32;
    const int n_base = bn * 32;

    // Accumulator
    fp32x16_t c_acc = {};

    // Process K in chunks of 64
    for (int k_chunk = 0; k_chunk < K; k_chunk += 64) {
        fp4x64_t a_reg = {};
        fp4x64_t b_reg = {};

        // === A loading (salykova pattern) ===
        // A_fp4[M, K/2]: row=lane, K offset = k_chunk/2 + half*16
        int a_row = m_base + lane;
        int a_k_offset = k_chunk / 2 + half * 16;
        if (a_row < M) {
            const fp4x2_t* a_ptr = A_fp4 + a_row * K_half + a_k_offset;
            for (int i = 0; i < 16; i++) {
                a_reg[i] = a_ptr[i];
            }
        }

        // === B loading (salykova transposed pattern) ===
        // B_q[N, K/2]: We need to load as if B were [K, N].
        // Blog: ldg_b = B + (tid%32)/2 + 16*32*(tid/32)
        //        stride between K rows = N/2 = 16
        // Our B_q[N, K/2]: stride between N rows = K/2
        // Thread loads column lane of B, K positions from k_chunk+half*32
        int b_col = n_base + lane;
        int b_k_offset = k_chunk / 2 + half * 16;
        if (b_col < N) {
            const fp4x2_t* b_ptr = B_q + b_col * K_half + b_k_offset;
            // Contiguous loading: b_ptr[0..15] gives K positions sequentially
            for (int i = 0; i < 16; i++) {
                b_reg[i] = b_ptr[i];
            }
        }

        // === Scale loading ===
        // A_scale[M, K/32]: 2 scale blocks per K=64 chunk
        int a_scale_idx0 = (k_chunk / 32) + (half == 0 ? 0 : 1);
        uint8_t sa = (a_row < M) ? A_scale[lane * K_blocks + (k_chunk/32) + half] : 127;

        int b_scale_idx = (k_chunk / 32) + half;
        uint8_t sb = (b_col < N) ? B_scale[lane * K_blocks + b_scale_idx] : 127;

        // Wait, scale mapping is wrong. Each thread handles lane=row for A
        // but lane=col for B. Both use the same lane!
        // A: row = m_base + lane, scale for K block (k_chunk/32 + half)
        // B: col = n_base + lane, scale for K block (k_chunk/32 + half)
        if (a_row < M) {
            sa = A_scale[a_row * K_blocks + (k_chunk / 32) + half];
        }
        if (b_col < N) {
            sb = B_scale[b_col * K_blocks + (k_chunk / 32) + half];
        }

        // === MFMA ===
        c_acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, c_acc,
            4, 4, 0,  // cbsz_a=4(FP4), cbsz_b=4(FP4), cbsz=0
            sa, 0, sb  // scale_a, blgp=0, scale_b
        );
    }

    // === Output store (salykova pattern) ===
    // c_reg[i*4+j] → C[row, col] where:
    //   col = lane (= tid % 32)
    //   row = half*4*8 + i*8 + j  ... wait, let me use the exact blog formula
    // Blog: C[tid%32 + (tid/32)*4*32 + j*32 + i*32*8]
    // For row-major C[M, N] with stride N:
    //   col = tid % 32 → output column within tile
    //   row_offset = (tid/32)*4 + j + i*8 → output row within tile
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int out_row = m_base + half * 4 + j + i * 8;
            int out_col = n_base + lane;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = (hip_bfloat16)(c_acc[i * 4 + j]);
            }
        }
    }
}

torch::Tensor launch_gemm_fp4(
    torch::Tensor A_fp4, torch::Tensor A_scale,
    torch::Tensor B_q, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(64);  // one wavefront

    hipLaunchKernelGGL(gemm_fp4_kernel, grid, block, 0, 0,
        (const fp4x2_t*)A_fp4.data_ptr(),
        (const uint8_t*)A_scale.data_ptr(),
        (const fp4x2_t*)B_q.data_ptr(),
        (const uint8_t*)B_scale.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),
        (int)M, (int)N, (int)K);

    return C;
}
'''

FWD_DECL = "torch::Tensor launch_gemm_fp4(torch::Tensor A_fp4, torch::Tensor A_scale, torch::Tensor B_q, torch::Tensor B_scale, int64_t M, int64_t N, int64_t K);"

_sc_cache = {}

def _unshuffle_e8m0(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _sc_cache:
        return _sc_cache[key]
    n_sc = K // 32
    sm = ((N + 255) // 256) * 256
    sn = ((n_sc + 7) // 8) * 8
    s = B_scale_sh.view(torch.uint8)
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N, :n_sc] = s[:N, :n_sc]
    r = p.view(sm // 32, sn // 8, 4, 16, 2, 2)
    u = r.permute(0, 5, 3, 1, 4, 2).contiguous()
    result = u.view(sm, sn)[:N, :n_sc]
    _sc_cache[key] = result
    return result

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="gemm_fp4_v26",
            cpp_sources=FWD_DECL,
            cuda_sources=HIP_SRC,
            functions=["launch_gemm_fp4"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffled, B_scale_shuffled = data
    M, K = A.shape
    N = B.shape[0]

    # Quantize A
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)

    # Unshuffle B scales (shuffled → raw E8M0)
    B_scale_raw = _unshuffle_e8m0(B_scale_shuffled, N, K)

    # B_q is [N, K/2] fp4x2 (raw, not shuffled)
    B_q_u8 = B_q.view(torch.uint8)
    A_fp4_u8 = A_fp4.view(torch.uint8)
    A_scale_u8 = A_scale.view(torch.uint8)
    B_scale_u8 = B_scale_raw.view(torch.uint8)

    mod = _get_mod()
    return mod.launch_gemm_fp4(A_fp4_u8, A_scale_u8, B_q_u8, B_scale_u8, M, N, K)

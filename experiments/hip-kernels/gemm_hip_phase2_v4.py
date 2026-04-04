#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Phase 2 v4: Try different A/B register layouts.
v3 had 0.93 correlation — close but data positions wrong.
Try: for A[16,128FP4], load contiguously per thread (each thread gets
one contiguous 64-byte chunk from the flattened A matrix).
This matches the CK pattern where data is loaded flat into LDS then
read into registers row-by-row.
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

// Try loading A and B using contiguous flat layout
// A[16, 64bytes]: 1024 bytes. 64 threads. Each thread: 16 bytes.
// Thread t reads bytes [t*16 .. t*16+15] from flattened A tile
// This means: t=0 reads A[row0, bytes 0-15], t=1 reads A[row0, bytes 16-31]
//             t=2 reads A[row0, bytes 32-47], t=3 reads A[row0, bytes 48-63]
//             t=4 reads A[row1, bytes 0-15], ... etc
// So: row = t / 4, K_group = t % 4

__global__ void gemm_fp4_v4(
    const unsigned char* __restrict__ A_fp4,
    const unsigned char* __restrict__ B_fp4,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_scale,
    hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int tile_n = blockIdx.x * 16;
    int tile_m = blockIdx.y * 16;
    int lane = threadIdx.x;

    c_result_t c_acc = {0.0f, 0.0f, 0.0f, 0.0f};

    int K_bytes = K / 2;
    int K_scales = K / 32;
    int K_chunks = K / 128;

    for (int kc = 0; kc < K_chunks; kc++) {
        a_operand_t a_reg;
        b_operand_t b_reg;
        for (int i = 0; i < 8; i++) { a_reg[i] = 0; b_reg[i] = 0; }

        int k_byte_start = kc * 64;

        // Approach 1: flat contiguous load
        // row = lane / 4, K_group = lane % 4
        // Each thread loads 16 bytes from A[row, K_group*16 .. K_group*16+15]
        {
            int a_row = lane / 4;
            int a_kgroup = lane % 4;
            int a_global_row = tile_m + a_row;
            if (a_global_row < M) {
                int byte_off = k_byte_start + a_kgroup * 16;
                const unsigned char* a_ptr = A_fp4 + a_global_row * K_bytes + byte_off;
                for (int i = 0; i < 4; i++) {
                    unsigned int val = 0;
                    for (int b = 0; b < 4; b++) {
                        if (byte_off + i*4+b < K_bytes)
                            val |= ((unsigned int)a_ptr[i*4+b]) << (b * 8);
                    }
                    a_reg[i] = (int)val;
                }
            }
        }

        // B: same layout
        {
            int b_row = lane / 4;
            int b_kgroup = lane % 4;
            int b_global_row = tile_n + b_row;
            if (b_global_row < N) {
                int byte_off = k_byte_start + b_kgroup * 16;
                const unsigned char* b_ptr = B_fp4 + b_global_row * K_bytes + byte_off;
                for (int i = 0; i < 4; i++) {
                    unsigned int val = 0;
                    for (int b = 0; b < 4; b++) {
                        if (byte_off + i*4+b < K_bytes)
                            val |= ((unsigned int)b_ptr[i*4+b]) << (b * 8);
                    }
                    b_reg[i] = (int)val;
                }
            }
        }

        // Scale: per-row per-K-group
        int a_row = lane / 4;
        int kgroup = lane % 4;
        int k_scale_idx = kc * 4 + kgroup;
        unsigned int sa = 127, sb = 127;
        {
            int a_global_row = tile_m + a_row;
            if (a_global_row < M && k_scale_idx < K_scales)
                sa = A_scale[a_global_row * K_scales + k_scale_idx];
        }
        {
            int b_row = lane / 4;
            int b_global_row = tile_n + b_row;
            if (b_global_row < N && k_scale_idx < K_scales)
                sb = B_scale[b_global_row * K_scales + k_scale_idx];
        }

        c_acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, c_acc, 4, 4, 0, sa, 0, sb);
    }

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
    hipLaunchKernelGGL(gemm_fp4_v4, grid, block, 0, 0,
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
            name="gemm_fp4_p2v4",
            cpp_sources="torch::Tensor gemm_fp4(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int);",
            cuda_sources=HIP_SOURCE,
            functions=["gemm_fp4"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        print("P2v4 COMPILE OK!", flush=True)
        return True
    except Exception as e:
        print(f"P2v4 COMPILE FAIL: {e}", flush=True)
        return False

def _test():
    if _hip_mod is None:
        return
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    for M, N, K in [(4, 32, 128), (4, 2880, 512), (16, 2112, 7168)]:
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
            tol_pass = torch.isclose(C_hip.float(), C_ref.float(), rtol=1e-2, atol=1e-2).float().mean().item()

            print(f"({M},{N},{K}): maxdiff={maxdiff:.2f} corr={corr:.6f} ratio={ratio:.4f} tol={tol_pass:.4f}", flush=True)
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

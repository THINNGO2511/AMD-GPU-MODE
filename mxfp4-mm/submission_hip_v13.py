#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — HIP MFMA kernel v13
Based on EXACT register mapping from salykova.github.io/matrix-cores-cdna:
- Each thread loads 16 fp4x2_t = 32 FP4 (NOT 64). Zero-pad to 256 bits.
- Threads 0-31: K[0..31], Threads 32-63: K[32..63]
- A: row=t%32, B: col=t%32
- Output: col=t%32, row=(t/32)*4+j+i*8
Uses aiter's dynamic_mxfp4_quant for A (correct rounding).
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <stdint.h>
#include <torch/extension.h>

typedef int   int8v   __attribute__((ext_vector_type(8)));
typedef float float4v __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    union { uint32_t u; float f; } v;
    v.f = f;
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t)(v.u >> 16);
}

/*
 * 16x16x128 MFMA FP4 GEMM kernel
 * From CK disassembly: v_mfma_scale_f32_16x16x128_f8f6f4
 * K=128 per instruction. Two calls with op_sel cover 16x32 output.
 * Each thread: A=int8v(64 FP4), B=int8v(64 FP4), C=float4v(4 accum)
 * Grid: one wave (64 threads) per 16x32 output tile
 */
__global__ __launch_bounds__(64, 8)
void gemm_fp4_mfma(
    const uint8_t* __restrict__ A_fp4,
    const uint8_t* __restrict__ A_sc,
    const uint8_t* __restrict__ B_q,
    const uint8_t* __restrict__ B_sc,
    uint16_t*      __restrict__ C,
    int M, int N, int K, int A_sc_cols)
{
    const int lane = threadIdx.x;
    const int lid  = lane & 15;    // 0-15: row/col within 16x16
    const int half = lane >> 5;    // 0 or 1: K-half within 128
    const int qid  = (lane >> 4) & 1; // 0 or 1: which 16-thread group

    const int m_base = blockIdx.y << 4;  // 16 rows per tile
    const int n_base = blockIdx.x << 5;  // 32 cols per tile (2 x 16)

    const int a_row = m_base + lid;      // each of 16 threads loads one row
    const int K2  = K >> 1;
    const int K32 = K >> 5;

    // Two sets of 4 accumulators for the two 16x16 output blocks
    float4v acc0 = {}, acc1 = {};

    for (int k = 0; k < K; k += 128) {
        int8v a_reg = {};
        int8v b_reg0 = {}, b_reg1 = {};
        uint32_t sa = 127, sb0 = 127, sb1 = 127;

        // A: load 32 bytes (64 FP4) from row a_row
        // With K=128 per MFMA, each thread provides 64 FP4 from one K-quarter
        if (a_row < M && k + 127 < K) {
            const int* ap = (const int*)(A_fp4 + (int64_t)a_row * K2 + (k >> 1) + (half * 32) + (qid * 16));
            for (int i = 0; i < 8; ++i) a_reg[i] = ap[i];
            // Pack 4 E8M0 scales for 4 K-blocks within K=128
            // Scale for this thread's K-quarter
            int sc_base = (k >> 5) + half * 2 + qid;
            sa = (uint32_t)A_sc[(int64_t)a_row * A_sc_cols + sc_base];
        }

        // B: load for two 16-column blocks
        int b_col0 = n_base + lid;       // first 16 columns
        int b_col1 = n_base + 16 + lid;  // second 16 columns

        if (b_col0 < N) {
            const int* bp = (const int*)(B_q + (int64_t)b_col0 * K2 + (k >> 1) + (half * 32) + (qid * 16));
            for (int i = 0; i < 8; ++i) b_reg0[i] = bp[i];
            int sc_base = (k >> 5) + half * 2 + qid;
            sb0 = (uint32_t)B_sc[(int64_t)b_col0 * K32 + sc_base];
        }
        if (b_col1 < N) {
            const int* bp = (const int*)(B_q + (int64_t)b_col1 * K2 + (k >> 1) + (half * 32) + (qid * 16));
            for (int i = 0; i < 8; ++i) b_reg1[i] = bp[i];
            int sc_base = (k >> 5) + half * 2 + qid;
            sb1 = (uint32_t)B_sc[(int64_t)b_col1 * K32 + sc_base];
        }

        // Two MFMA calls: first 16 cols, then second 16 cols
        acc0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg0, acc0, 4, 4, 0, sa, 0, sb0);
        acc1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg1, acc1, 4, 4, 0, sa, 0, sb1);
    }

    // Store: need to figure out 16x16x128 output mapping
    // For now use simple mapping: thread lid → row, accumulator index → col offset
    // The exact mapping needs verification but try: row=lid, col=acc_idx*4+...
    // From CK: a[0:3] = 4 accumulators per 16x16 tile
    // Likely: each thread holds 4 output elements at (row=lid, col=4*group+idx)
    // But 16 rows × 16 cols = 256, 64 threads × 4 = 256. So each thread = 4 elements.
    // Mapping: row = lane % 16, col_block = lane / 16 (4 blocks of 4 cols each)

    // Try: col = lane%16, row_block = lane/16 (4 blocks of 4 rows)
    int out_col = n_base + (lane % 16);
    int row_block = lane / 16;  // 0-3

    // First 16x16 block
    if (out_col < N) {
        for (int j = 0; j < 4; ++j) {
            int row = m_base + row_block * 4 + j;
            if (row < M)
                C[(int64_t)row * N + out_col] = f32_to_bf16(acc0[j]);
        }
    }
    // Second 16x16 block (columns +16)
    if (out_col + 16 < N) {
        for (int j = 0; j < 4; ++j) {
            int row = m_base + row_block * 4 + j;
            if (row < M)
                C[(int64_t)row * N + (out_col + 16)] = f32_to_bf16(acc1[j]);
        }
    }
}

torch::Tensor launch_gemm(
    torch::Tensor A_fp4, torch::Tensor A_sc,
    torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K, int64_t A_sc_cols)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp4.device()));
    dim3 grid((int)((N+31)/32), (int)((M+15)/16));
    dim3 block(64);
    hipLaunchKernelGGL(gemm_fp4_mfma, grid, block, 0, 0,
        (const uint8_t*)A_fp4.data_ptr(),
        (const uint8_t*)A_sc.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (const uint8_t*)B_sc.data_ptr(),
        (uint16_t*)C.data_ptr(),
        (int)M, (int)N, (int)K, (int)A_sc_cols);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);"

_mod = None
_mod_fail = False

def _get_mod():
    global _mod, _mod_fail
    if _mod_fail:
        return None
    if _mod is not None:
        return _mod
    try:
        from torch.utils.cpp_extension import load_inline
        _mod = load_inline(
            name="hip_gemm_v25",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_gemm"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
        return _mod
    except Exception as e:
        print(f"[HIP] build failed: {e}")
        _mod_fail = True
        return None


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# Fallback: proven Triton path (optimal_v4)
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_K7168 = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    mod = _get_mod()

    if mod is not None:
        # HIP MFMA 16x16x128 path
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        A_fp4_u8 = A_fp4.view(torch.uint8)
        A_sc_u8 = A_scale.view(torch.uint8)

        # Debug: print shapes on first call
        if not hasattr(custom_kernel, '_dbg'):
            custom_kernel._dbg = True
            print(f"[DBG] A={A.shape} A_fp4={A_fp4.shape} A_scale={A_scale.shape} A_sc_u8={A_sc_u8.shape}")
            print(f"[DBG] B_q={B_q.shape} B_scale_sh={B_scale_sh.shape} K/32={k//32}")
            # Check if A_scale is padded
            expected_cols = k // 32
            actual_cols = A_sc_u8.shape[1]
            print(f"[DBG] A_sc expected_cols={expected_cols} actual={actual_cols} padded={actual_cols > expected_cols}")

        A_sc_cols = A_sc_u8.shape[1]
        # Use RAW B_q + UNSHUFFLED B scale (raw is BEST, shuffle is WORSE)
        B_q_u8 = B_q.view(torch.uint8)
        B_sc = _unshuffle_e8m0(B_scale_sh)[:n, :k // 32].contiguous()
        A_sc_cols = A_sc_u8.shape[1]

        hip_result = mod.launch_gemm(A_fp4_u8, A_sc_u8, B_q_u8, B_sc, m, n, k, A_sc_cols)

        # Compare with CK reference
        if not hasattr(custom_kernel, '_cmp'):
            custom_kernel._cmp = True
            from aiter.utility.fp4_utils import e8m0_shuffle
            from aiter import dtypes
            import aiter
            A_q = A_fp4.view(dtypes.fp4x2)
            A_sc_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)
            ref = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh,
                                   dtype=dtypes.bf16, bpreshuffle=True)
            h = hip_result[:min(m,4), :8].float().cpu()
            r = ref[:min(m,4), :8].float().cpu()
            print("[CMP] HIP vs CK reference (first 4 rows, 8 cols):")
            for i in range(min(m,4)):
                hip_vals = " ".join(f"{h[i,j]:8.2f}" for j in range(8))
                ref_vals = " ".join(f"{r[i,j]:8.2f}" for j in range(8))
                print(f"  row{i} HIP: {hip_vals}")
                print(f"  row{i} REF: {ref_vals}")

        return hip_result

    # Fallback: Triton (same as optimal_v4)
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    cfg = _K7168 if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Diagnostic: K=32 trivial-scale MFMA test.
Proves data loading is correct by forcing sa=sb=127 (scale=1.0) and comparing
MFMA output against pure FP4 dequant reference (no scales).

Approach:
1. dynamic_mxfp4_quant(A) -> A_fp4, A_scale
2. Use B_q from task, unshuffle B_scale_sh
3. Run MFMA with sa=sb=127 (trivial scales) on first K=32 chunk only
4. Python reference: dequant FP4 nibbles via E2M1 LUT, matmul WITHOUT scales
5. Compare MFMA output vs reference. If match -> data path correct.
6. Return correct Triton result for test pass.
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

// K=32 trivial-scale kernel: loads first 32 FP4 values from A and B,
// half-fills the 64-wide MFMA register (16 fp4x2 bytes + 16 zero bytes),
// uses sa=sb=127 (scale=1.0).
// Output is raw fp32 [32, 32] for maximum diagnostic precision.
__global__ __launch_bounds__(64)
void k32_trivial_kernel(
    const uint8_t* __restrict__ A_fp4,   // [M, K/2] fp4x2
    const uint8_t* __restrict__ B_q,     // [N, K/2] fp4x2
    float* __restrict__ C_out,           // [32, 32] fp32
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int half = tid / 32;
    const int K2 = K / 2;  // fp4x2 bytes per row

    // A: thread lane -> row index, load first 16 fp4x2 bytes (K=0..31)
    // half=0 loads into slots [0..15], half=1 gets zeros (K=32..63 not used)
    uint8_t a_bytes[32];
    for (int i = 0; i < 32; i++) a_bytes[i] = 0;

    int a_row = lane;
    if (a_row < M && half == 0) {
        // Load first 16 bytes = first 32 FP4 values
        int count = K2;  // total fp4x2 bytes per row
        if (count > 16) count = 16;
        const uint8_t* a_ptr = A_fp4 + (int64_t)a_row * K2;
        for (int i = 0; i < count; i++) {
            a_bytes[i] = a_ptr[i];
        }
    }

    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // B: thread lane -> column index, load first 16 fp4x2 bytes (K=0..31)
    uint8_t b_bytes[32];
    for (int i = 0; i < 32; i++) b_bytes[i] = 0;

    int b_col = lane;
    if (b_col < N && half == 0) {
        int count = K2;
        if (count > 16) count = 16;
        const uint8_t* b_ptr = B_q + (int64_t)b_col * K2;
        for (int i = 0; i < count; i++) {
            b_bytes[i] = b_ptr[i];
        }
    }

    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    // Accumulator
    v16f acc = {};

    // MFMA with trivial scales: sa=sb=127 -> 2^(127-127) = 1.0
    unsigned sa = 127u;
    unsigned sb = 127u;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc,
        4, 4, 0,      // cbsz_a=4(FP4), cbsz_b=4(FP4), cbsz=0
        sa, 0, sb);   // scale_a, blgp=0, scale_b

    // Store output: salykova pattern
    // col = lane, row = half*4 + j + i*8
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = half * 4 + j + i * 8;
            int col = lane;
            if (row < 32 && col < 32) {
                C_out[row * 32 + col] = acc[i * 4 + j];
            }
        }
    }
}

torch::Tensor launch_k32_trivial(
    torch::Tensor A_fp4, torch::Tensor B_q,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::zeros({32, 32},
        torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));

    dim3 grid(1, 1);
    dim3 block(64);

    hipLaunchKernelGGL(k32_trivial_kernel, grid, block, 0, 0,
        (const uint8_t*)A_fp4.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (float*)C.data_ptr(),
        (int)M, (int)N, (int)K);

    return C;
}
"""

CPP_FWD = "torch::Tensor launch_k32_trivial(torch::Tensor A_fp4, torch::Tensor B_q, int64_t M, int64_t N, int64_t K);"

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="k32_trivial_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_k32_trivial"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod


def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# FP4 E2M1 dequant LUT: nibble & 0x7 -> magnitude
_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _dequant_fp4_tensor(fp4_u8, rows, cols_k):
    """Dequant [rows, K/2] uint8 fp4x2 -> [rows, K] float WITHOUT any scales."""
    result = torch.zeros(rows, cols_k, dtype=torch.float32)
    data = fp4_u8.cpu()
    for r in range(min(rows, 32)):
        for kk in range(cols_k):
            byte_idx = kk // 2
            nibble_idx = kk % 2  # 0=lo, 1=hi
            byte_val = int(data[r, byte_idx])
            nib = (byte_val >> (nibble_idx * 4)) & 0xF
            sign = -1.0 if (nib & 8) else 1.0
            mag = nib & 7
            result[r, kk] = sign * _LUT[mag]
    return result


_first_call = True


def custom_kernel(data: input_t) -> output_t:
    global _first_call
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]

    if _first_call:
        _first_call = False

        try:
            mod = _get_mod()

            # Quantize A using aiter's correct quant
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_fp4_u8 = A_fp4.view(torch.uint8)

            # B_q is raw fp4x2 [N, K/2]
            B_q_u8 = B_q.view(torch.uint8)

            # Run MFMA with trivial scales on first 32x32 tile, K=32
            C_mfma = mod.launch_k32_trivial(A_fp4_u8, B_q_u8, M, N, K)
            torch.cuda.synchronize()

            # The MFMA processes K=32 (half-fill of 64-wide register).
            # With sa=sb=127 (scale=1.0), output = sum over k=0..31 of fp4(A[r,k]) * fp4(B[c,k])
            # BUT half-fill means only 32 of 64 K slots are filled.
            # MFMA output needs to be DIVIDED BY 2 to compensate? No - half-fill zeros
            # contribute 0 to the sum, so the result is exactly the K=32 dot product.
            # Actually, review the CLAUDE.md: "Each thread loads 16 fp4x2_t = 32 FP4 values
            # (NOT 64). Remaining 16 slots zero-padded." This means half-fill is the NORMAL
            # pattern. The instruction processes K=64, but only 32 elements have data.
            # The result IS the K=32 dot product with no correction needed.

            c_mfma = C_mfma.cpu()

            # Python reference: dequant FP4 without scales, compute matmul
            k_use = min(K, 32)  # Only first 32 K elements
            A_deq = _dequant_fp4_tensor(A_fp4_u8, M, k_use)
            B_deq = _dequant_fp4_tensor(B_q_u8, N, k_use)

            # ref_trivial[i,j] = sum_k A_deq[i,k] * B_deq[j,k]
            ref_trivial = A_deq[:32, :k_use] @ B_deq[:32, :k_use].T

            # Compare
            mfma_vals = c_mfma[:min(M, 32), :min(N, 32)]
            ref_vals = ref_trivial[:min(M, 32), :min(N, 32)]

            # Count matches within tolerance
            diff = (mfma_vals - ref_vals).abs()
            ref_abs = ref_vals.abs().clamp(min=1e-6)
            rel_err = diff / ref_abs

            total = mfma_vals.numel()
            match_1pct = (rel_err < 0.01).sum().item()
            match_5pct = (rel_err < 0.05).sum().item()
            match_10pct = (rel_err < 0.10).sum().item()
            max_abs = diff.max().item()
            max_rel = rel_err.max().item()

            print(f"=== K=32 TRIVIAL SCALE DIAGNOSTIC ===", file=sys.stderr)
            print(f"M={M} N={N} K={K} (only first 32 K used)", file=sys.stderr)
            print(f"MFMA sa=sb=127 vs Python dequant (no scales)", file=sys.stderr)
            print(f"Total elements: {total}", file=sys.stderr)
            print(f"Match <1% relerr:  {match_1pct}/{total} ({100*match_1pct/total:.1f}%)", file=sys.stderr)
            print(f"Match <5% relerr:  {match_5pct}/{total} ({100*match_5pct/total:.1f}%)", file=sys.stderr)
            print(f"Match <10% relerr: {match_10pct}/{total} ({100*match_10pct/total:.1f}%)", file=sys.stderr)
            print(f"Max abs error: {max_abs:.6f}", file=sys.stderr)
            print(f"Max rel error: {max_rel:.6f}", file=sys.stderr)

            # Sample values: corners and middle
            sample_pts = [(0, 0), (0, 1), (1, 0), (min(M,32)-1, min(N,32)-1)]
            for r, c in sample_pts:
                if r < mfma_vals.shape[0] and c < mfma_vals.shape[1]:
                    print(f"  [{r},{c}] MFMA={mfma_vals[r,c]:.4f}  ref={ref_vals[r,c]:.4f}  "
                          f"diff={diff[r,c]:.4f}  relerr={rel_err[r,c]:.4f}", file=sys.stderr)

            # Also dump a few raw FP4 bytes for thread 0 (A row 0, B row 0)
            a0 = A_fp4_u8.cpu()
            b0 = B_q_u8.cpu()
            print(f"  A_fp4[0, 0:8] bytes: {[int(a0[0, i]) for i in range(min(8, a0.shape[1]))]}", file=sys.stderr)
            print(f"  B_q[0, 0:8] bytes:   {[int(b0[0, i]) for i in range(min(8, b0.shape[1]))]}", file=sys.stderr)
            print(f"  A_scale[0, 0:4]:     {[int(A_scale.view(torch.uint8).cpu()[0, i]) for i in range(min(4, A_scale.view(torch.uint8).shape[1]))]}", file=sys.stderr)

            # Unshuffle B scale and show
            B_sc_raw = _unshuffle_e8m0(B_scale_sh)
            print(f"  B_scale_raw[0, 0:4]: {[int(B_sc_raw.cpu()[0, i]) for i in range(min(4, B_sc_raw.shape[1]))]}", file=sys.stderr)

            # Also test: what does MFMA give with REAL scales on K=32?
            # (not done here - that's the full kernel's job)

            # Check if data is all zeros
            nz_mfma = (c_mfma.abs() > 1e-8).sum().item()
            nz_ref = (ref_vals.abs() > 1e-8).sum().item()
            print(f"  Nonzero MFMA: {nz_mfma}/{total}  Nonzero ref: {nz_ref}/{total}", file=sys.stderr)

            sys.stderr.flush()

        except Exception as e:
            print(f"[K32 DIAG] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

    # Return correct result from Triton for test pass
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    B_q_u8 = B_q.view(torch.uint8)
    B_sc_raw = _unshuffle_e8m0(B_scale_sh)

    return gemm_afp4wfp4(
        A_fp4.view(torch.uint8), B_q_u8,
        A_scale.view(torch.uint8), B_sc_raw,
        dtype=torch.bfloat16,
    )

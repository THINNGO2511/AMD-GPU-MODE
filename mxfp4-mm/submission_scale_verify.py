#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Scale Verify Diagnostic: Isolate exactly where the scale mismatch occurs.

PROVEN:
- MFMA data loading is 100% correct (trivial scale test: 256/256 ZERO error)
- MFMA scale formula IS 2^(sa-127) (scale sweep confirms ratio=1.000 for all values)
- But K=32 + real scales gives ~8% error

This test isolates the mismatch by:
1. Running dynamic_mxfp4_quant(A) to get A_fp4 + A_scale
2. Unshuffling B_scale_sh to get B_scale
3. MANUALLY dequantizing (Python, CPU) using FP4 LUT + E8M0 scales
4. Running a HIP kernel with single K=32 block (half=0 only, half=1 zeros)
5. Comparing MFMA output vs manual dequant reference element-by-element
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import sys
import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>
#include <cstdint>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

// Single 32x32 tile, K=32 only (half=0 fills, half=1 zeros)
// Each thread: lane=tid%32, half=tid/32
// half=0: load 16 fp4x2 bytes from K[0..31], sa from A_scale[row,0], sb from B_scale[col,0]
// half=1: all zeros, sa=sb=127 (neutral)
__global__ __launch_bounds__(64)
void scale_verify_kernel(
    const uint8_t* __restrict__ A_fp4,   // [M, K/2] uint8
    const uint8_t* __restrict__ A_scale, // [M, K/32] uint8 E8M0
    const uint8_t* __restrict__ B_q,     // [N, K/2] uint8
    const uint8_t* __restrict__ B_scale, // [N, K/32] uint8 E8M0
    float* __restrict__ C_out,           // [32, 32] float
    float* __restrict__ diag_out,        // [128] diagnostics
    int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int half = tid / 32;

    const int Kh = K / 2;   // fp4x2 per row
    const int Kb = K / 32;  // scale blocks per row

    fp32x16_t acc = {};
    fp4x64_t a_reg = {};
    fp4x64_t b_reg = {};
    unsigned sa = 127u;
    unsigned sb = 127u;

    if (half == 0) {
        // === HALF 0: Load real data for first K=32 block ===
        int a_row = lane;
        if (a_row < M) {
            // Load 16 fp4x2 bytes = 32 FP4 values from K[0..31]
            const uint8_t* a_ptr = A_fp4 + (long long)a_row * Kh;
            for (int i = 0; i < 16; i++) {
                a_reg[i] = (fp4x2_t)a_ptr[i];
            }
            // Scale for first K block
            sa = (unsigned)A_scale[(long long)a_row * Kb];
        }

        int b_col = lane;
        if (b_col < N) {
            // Load 16 fp4x2 bytes = 32 FP4 values from K[0..31]
            const uint8_t* b_ptr = B_q + (long long)b_col * Kh;
            for (int i = 0; i < 16; i++) {
                b_reg[i] = (fp4x2_t)b_ptr[i];
            }
            // Scale for first K block
            sb = (unsigned)B_scale[(long long)b_col * Kb];
        }
    }
    // half=1: a_reg, b_reg stay zero; sa, sb stay 127 (neutral scale)

    // === MFMA ===
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

    // === STORE C ===
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int r = half * 4 + j + i * 8;
            int c = lane;
            if (r < 32 && c < 32)
                C_out[r * 32 + c] = acc[i * 4 + j];
        }
    }

    // === DIAGNOSTICS: threads 0-3 dump their sa, sb, first 4 A bytes, first 4 B bytes ===
    if (tid < 4) {
        int base = tid * 16;
        // sa, sb
        diag_out[base + 0] = (float)sa;
        diag_out[base + 1] = (float)sb;
        // First 4 A_fp4 bytes (raw data this thread loaded)
        uint8_t a_bytes[32];
        __builtin_memcpy(a_bytes, &a_reg, 32);
        for (int i = 0; i < 4; i++) diag_out[base + 2 + i] = (float)a_bytes[i];
        // First 4 B_q bytes
        uint8_t b_bytes[32];
        __builtin_memcpy(b_bytes, &b_reg, 32);
        for (int i = 0; i < 4; i++) diag_out[base + 6 + i] = (float)b_bytes[i];
        // First 4 acc values
        for (int i = 0; i < 4; i++) diag_out[base + 10 + i] = acc[i];
        // Padding
        diag_out[base + 14] = (float)lane;
        diag_out[base + 15] = (float)half;
    }
}

torch::Tensor launch_scale_verify(
    torch::Tensor A_fp4, torch::Tensor A_scale,
    torch::Tensor B_q, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K)
{
    // C_out [32,32] float + diag [128] float, concatenated
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));
    auto diag = torch::zeros({128}, torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));

    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(scale_verify_kernel, grid, block, 0, 0,
        (const uint8_t*)A_fp4.data_ptr(),
        (const uint8_t*)A_scale.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (const uint8_t*)B_scale.data_ptr(),
        (float*)C.data_ptr(),
        (float*)diag.data_ptr(),
        (int)M, (int)N, (int)K);

    // Return both packed: first 1024 = C, next 128 = diag
    auto result = torch::zeros({1024 + 128}, torch::TensorOptions().dtype(torch::kFloat32).device(A_fp4.device()));
    result.narrow(0, 0, 1024).copy_(C.reshape({1024}));
    result.narrow(0, 1024, 128).copy_(diag);
    return result;
}
"""

CPP_FWD = "torch::Tensor launch_scale_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);"

_mod = None

def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="scale_verify_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_scale_verify"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# FP4 E2M1 magnitude LUT (sign handled separately)
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _manual_dequant_block(fp4_bytes, scale_u8, num_rows, K_vals=32):
    """
    Manually dequantize a [num_rows, K_vals] block of FP4 data.
    fp4_bytes: [num_rows, K_vals/2] uint8 tensor (CPU)
    scale_u8: [num_rows] uint8 tensor (CPU) - E8M0 scales for the first K block
    Returns: [num_rows, K_vals] float64 tensor
    """
    result = torch.zeros(num_rows, K_vals, dtype=torch.float64)
    for r in range(num_rows):
        sa = int(scale_u8[r])
        scale_factor = 2.0 ** (sa - 127)
        for k in range(K_vals):
            byte_idx = k // 2
            nibble = k % 2
            byte_val = int(fp4_bytes[r, byte_idx])
            nib = (byte_val >> (nibble * 4)) & 0xF
            sign = -1.0 if (nib & 8) else 1.0
            mag = FP4_LUT[nib & 7]
            result[r, k] = sign * mag * scale_factor
    return result


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    # =============================================
    # Step 1: Quantize A
    # =============================================
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_fp4_u8 = A_fp4.view(torch.uint8)
    A_scale_u8 = A_scale.view(torch.uint8)

    # =============================================
    # Step 2: Unshuffle B scale
    # =============================================
    B_scale_raw = _unshuffle_e8m0(B_scale_sh)
    B_q_u8 = B_q.view(torch.uint8)
    B_scale_u8 = B_scale_raw.view(torch.uint8)

    # =============================================
    # Step 3: Manual CPU dequant for first 32x32 tile, K=32
    # =============================================
    num_a_rows = min(M, 32)
    num_b_cols = min(N, 32)

    a_cpu = A_fp4_u8[:num_a_rows, :16].cpu()      # first 16 bytes = 32 FP4 values
    a_sc_cpu = A_scale_u8[:num_a_rows, 0].cpu()    # scale for first K block

    b_cpu = B_q_u8[:num_b_cols, :16].cpu()
    b_sc_cpu = B_scale_u8[:num_b_cols, 0].cpu()

    A_dequant = _manual_dequant_block(a_cpu, a_sc_cpu, num_a_rows, 32)
    B_dequant = _manual_dequant_block(b_cpu, b_sc_cpu, num_b_cols, 32)

    ref_scaled = (A_dequant @ B_dequant.T).float()   # [32, 32] reference

    # =============================================
    # Step 4: Run HIP MFMA kernel
    # =============================================
    mod = _get_mod()
    raw = mod.launch_scale_verify(A_fp4_u8, A_scale_u8, B_q_u8, B_scale_u8, M, N, K)
    torch.cuda.synchronize()

    raw_cpu = raw.cpu()
    mfma_C = raw_cpu[:1024].reshape(32, 32)
    diag = raw_cpu[1024:]

    # =============================================
    # Step 5: Compare element-by-element
    # =============================================
    P = sys.stderr

    print("=" * 70, file=P)
    print("SCALE VERIFY DIAGNOSTIC", file=P)
    print(f"M={M} N={N} K={K}", file=P)
    print(f"A_fp4={A_fp4.shape} A_scale={A_scale.shape} B_q={B_q.shape}", file=P)
    print(f"A_scale_u8={A_scale_u8.shape} B_scale_u8={B_scale_u8.shape}", file=P)
    print("=" * 70, file=P)

    # Diagnostic: thread data
    for t in range(min(4, num_a_rows)):
        base = t * 16
        sa_hw = int(diag[base + 0])
        sb_hw = int(diag[base + 1])
        a_bytes_hw = [int(diag[base + 2 + i]) for i in range(4)]
        b_bytes_hw = [int(diag[base + 6 + i]) for i in range(4)]
        acc_hw = [float(diag[base + 10 + i]) for i in range(4)]
        lane_hw = int(diag[base + 14])
        half_hw = int(diag[base + 15])

        sa_cpu = int(a_sc_cpu[t])
        sb_cpu = int(b_sc_cpu[t])
        a_bytes_cpu = [int(a_cpu[t, i]) for i in range(4)]
        b_bytes_cpu = [int(b_cpu[t, i]) for i in range(4)]

        print(f"\n--- Thread {t} (lane={lane_hw}, half={half_hw}) ---", file=P)
        print(f"  A_scale: HW={sa_hw}  CPU={sa_cpu}  {'MATCH' if sa_hw == sa_cpu else 'MISMATCH!'}", file=P)
        print(f"  B_scale: HW={sb_hw}  CPU={sb_cpu}  {'MATCH' if sb_hw == sb_cpu else 'MISMATCH!'}", file=P)
        print(f"  A_bytes[0:4]: HW={a_bytes_hw}  CPU={a_bytes_cpu}  {'MATCH' if a_bytes_hw == a_bytes_cpu else 'MISMATCH!'}", file=P)
        print(f"  B_bytes[0:4]: HW={b_bytes_hw}  CPU={b_bytes_cpu}  {'MATCH' if b_bytes_hw == b_bytes_cpu else 'MISMATCH!'}", file=P)
        print(f"  acc[0:4]: {acc_hw}", file=P)

    # Element-by-element comparison for first 8x8
    print(f"\n{'=' * 70}", file=P)
    print("ELEMENT-BY-ELEMENT: MFMA vs Manual Dequant Reference", file=P)
    print(f"{'=' * 70}", file=P)

    cmp_rows = min(8, num_a_rows)
    cmp_cols = min(8, num_b_cols)
    total_err = 0.0
    total_abs = 0.0
    max_err = 0.0
    err_count = 0
    total_count = 0

    for r in range(cmp_rows):
        for c in range(cmp_cols):
            mfma_val = float(mfma_C[r, c])
            ref_val = float(ref_scaled[r, c])
            err = abs(mfma_val - ref_val)
            rel_err = err / max(abs(ref_val), 1e-6)
            total_err += err
            total_abs += abs(ref_val)
            total_count += 1
            if rel_err > 0.01:
                err_count += 1
            max_err = max(max_err, rel_err)

    print(f"\nComparison over {cmp_rows}x{cmp_cols} = {total_count} elements:", file=P)
    print(f"  Total abs error: {total_err:.6f}", file=P)
    print(f"  Total abs ref:   {total_abs:.6f}", file=P)
    print(f"  Mean rel error:  {total_err / max(total_abs, 1e-6):.6f}", file=P)
    print(f"  Max  rel error:  {max_err:.6f}", file=P)
    print(f"  Elements >1% err: {err_count}/{total_count}", file=P)

    # Detailed first 4x4
    print(f"\nFirst 4x4 detail:", file=P)
    print(f"{'row':>3} {'col':>3} {'MFMA':>12} {'Manual':>12} {'Err':>10} {'RelErr':>10}", file=P)
    for r in range(min(4, num_a_rows)):
        for c in range(min(4, num_b_cols)):
            mv = float(mfma_C[r, c])
            rv = float(ref_scaled[r, c])
            err = abs(mv - rv)
            rel = err / max(abs(rv), 1e-6)
            flag = " ***" if rel > 0.01 else ""
            print(f"{r:3d} {c:3d} {mv:12.4f} {rv:12.4f} {err:10.4f} {rel:10.6f}{flag}", file=P)

    # Scale values for first 4 rows and 4 cols
    print(f"\nA_scale (first 4 rows, first 2 blocks):", file=P)
    for r in range(min(4, num_a_rows)):
        vals = [int(A_scale_u8[r, b].item()) for b in range(min(2, A_scale_u8.shape[1]))]
        print(f"  row {r}: {vals}  (2^(v-127) = {[2.0**(v-127) for v in vals]})", file=P)

    print(f"\nB_scale (first 4 cols, first 2 blocks):", file=P)
    for c in range(min(4, num_b_cols)):
        vals = [int(B_scale_u8[c, b].item()) for b in range(min(2, B_scale_u8.shape[1]))]
        print(f"  col {c}: {vals}  (2^(v-127) = {[2.0**(v-127) for v in vals]})", file=P)

    # Also compare MFMA with gemm_afp4wfp4 reference on first 4x4
    print(f"\n{'=' * 70}", file=P)
    print("MFMA vs gemm_afp4wfp4 (Triton reference) for first 4x4:", file=P)
    print(f"{'=' * 70}", file=P)

    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    triton_ref = gemm_afp4wfp4(
        A_fp4_u8, B_q_u8, A_scale, B_scale_raw,
        dtype=torch.bfloat16
    )
    triton_cpu = triton_ref[:min(4, M), :min(4, N)].float().cpu()

    print(f"{'row':>3} {'col':>3} {'MFMA':>12} {'Triton':>12} {'Manual':>12} {'Tri-Man':>10}", file=P)
    for r in range(min(4, num_a_rows)):
        for c in range(min(4, num_b_cols)):
            mv = float(mfma_C[r, c])
            tv = float(triton_cpu[r, c])
            rv = float(ref_scaled[r, c])
            tri_man_err = abs(tv - rv) / max(abs(rv), 1e-6)
            print(f"{r:3d} {c:3d} {mv:12.4f} {tv:12.4f} {rv:12.4f} {tri_man_err:10.6f}", file=P)

    # Check if Triton matches manual dequant (it should, since both use same FP4 LUT)
    triton_vs_manual = 0
    for r in range(min(4, num_a_rows)):
        for c in range(min(4, num_b_cols)):
            tv = float(triton_cpu[r, c])
            rv = float(ref_scaled[r, c])
            if abs(tv - rv) / max(abs(rv), 1e-6) > 0.01:
                triton_vs_manual += 1
    print(f"\nTriton vs Manual: {triton_vs_manual}/16 elements >1% error", file=P)

    # Check if MFMA matches Triton (same kernel, same scales)
    mfma_vs_triton = 0
    for r in range(min(4, num_a_rows)):
        for c in range(min(4, num_b_cols)):
            mv = float(mfma_C[r, c])
            tv = float(triton_cpu[r, c])
            if abs(mv - tv) / max(abs(tv), 1e-6) > 0.01:
                mfma_vs_triton += 1
    print(f"MFMA vs Triton:  {mfma_vs_triton}/16 elements >1% error", file=P)

    # KEY INSIGHT: Check scale product vs what MFMA produces
    # If we compute MFMA_output / manual_ref, we get the effective scale ratio
    print(f"\n{'=' * 70}", file=P)
    print("RATIO ANALYSIS: MFMA / Manual_Ref (should be 1.0 if scales match)", file=P)
    print(f"{'=' * 70}", file=P)
    for r in range(min(4, num_a_rows)):
        ratios = []
        for c in range(min(4, num_b_cols)):
            mv = float(mfma_C[r, c])
            rv = float(ref_scaled[r, c])
            if abs(rv) > 1e-6:
                ratios.append(mv / rv)
            else:
                ratios.append(float('nan'))
        print(f"  row {r}: ratios = {['%.4f' % r for r in ratios]}", file=P)

    # Also print ratio for Triton/Manual
    print(f"\nRATIO: Triton / Manual_Ref:", file=P)
    for r in range(min(4, num_a_rows)):
        ratios = []
        for c in range(min(4, num_b_cols)):
            tv = float(triton_cpu[r, c])
            rv = float(ref_scaled[r, c])
            if abs(rv) > 1e-6:
                ratios.append(tv / rv)
            else:
                ratios.append(float('nan'))
        print(f"  row {r}: ratios = {['%.4f' % r for r in ratios]}", file=P)

    sys.stderr.flush()

    # =============================================
    # Return correct Triton result for test pass
    # =============================================
    return gemm_afp4wfp4(
        A_fp4_u8, B_q_u8, A_scale, B_scale_raw,
        dtype=torch.bfloat16
    )

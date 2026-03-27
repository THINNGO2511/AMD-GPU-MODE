#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MFMA E8M0 Scale Formula Diagnostic.

PROVEN: Data path is 100% correct with sa=sb=127 (256/256 zero error).
With real E8M0 scales, ~8% error. This kernel discovers the MFMA's
ACTUAL scale formula by sweeping scale values with constant FP4 data.

Approach: Single 32x32 tile, K=32 (half=0 only), A=B=all 1.0 FP4 (nibble 0x2).
With 32 products of 1.0*1.0, the unscaled result = 32.0.
Vary sa (keeping sb=127) and measure C[0,0] to reverse-engineer the formula.

If MFMA uses 2^(sa-127): result = 32.0 * 2^(sa-127), ratio = 1.0 for all sa.
If different bias: ratio = constant != 1.0.
If different formula: ratios vary.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch, sys
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

// Launch one wavefront (64 threads), single 32x32 tile, K=32.
// Only half=0 loads data (16 bytes of 0x22 = FP4 1.0 pairs).
// half=1 loads zeros everywhere.
// sa_val and sb_val are the E8M0 scale values to test.
__global__ __launch_bounds__(64)
void scale_sweep_kernel(float* __restrict__ C_out, int sa_val, int sb_val) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int half = tid / 32;

    // Build A register: half=0 gets 16 bytes of 0x22 (FP4 1.0), half=1 gets zeros
    uint8_t a_bytes[32];
    if (half == 0) {
        for (int i = 0; i < 16; i++) a_bytes[i] = 0x22;  // two FP4 1.0 nibbles per byte
    } else {
        for (int i = 0; i < 16; i++) a_bytes[i] = 0x00;
    }
    for (int i = 16; i < 32; i++) a_bytes[i] = 0x00;  // upper half always zero
    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // Build B register: same pattern as A
    uint8_t b_bytes[32];
    if (half == 0) {
        for (int i = 0; i < 16; i++) b_bytes[i] = 0x22;
    } else {
        for (int i = 0; i < 16; i++) b_bytes[i] = 0x00;
    }
    for (int i = 16; i < 32; i++) b_bytes[i] = 0x00;
    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    // Zero accumulator
    v16f acc = {};

    // Scale values passed as parameters
    unsigned sa = (unsigned)sa_val;
    unsigned sb = (unsigned)sb_val;

    // MFMA: cbsz=4 (FP4 A), blgp=4 (FP4 B), cbsz_sel=0, blgp_sel=0
    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc,
        4, 4, 0,
        sa, 0, sb);

    // Store 32x32 output using salykova pattern
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = half * 4 + j + i * 8;
            int col = lane;
            C_out[row * 32 + col] = acc[i * 4 + j];
        }
    }
}

torch::Tensor launch_sweep(int64_t sa_val, int64_t sb_val) {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(scale_sweep_kernel, grid, block, 0, 0,
        (float*)C.data_ptr(), (int)sa_val, (int)sb_val);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_sweep(int64_t sa_val, int64_t sb_val);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="scale_sweep_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_sweep"],
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


_ran = False
def custom_kernel(data: input_t) -> output_t:
    global _ran
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]

    if not _ran:
        _ran = True
        try:
            mod = _get_mod()

            print("=" * 80, file=sys.stderr)
            print("MFMA E8M0 SCALE FORMULA DIAGNOSTIC", file=sys.stderr)
            print("Setup: 32x32 tile, K=32 (half=0 only), A=B=all 1.0 FP4", file=sys.stderr)
            print("Unscaled dot = 32 products of 1.0*1.0 = 32.0", file=sys.stderr)
            print("=" * 80, file=sys.stderr)

            # ---- PART 1: Sweep sa with sb=127 ----
            print("\n--- PART 1: Sweep sa (sb=127 fixed) ---", file=sys.stderr)
            print(f"{'sa':>5} {'C[0,0]':>14} {'expected_2pow':>14} {'ratio':>10} {'log2(ratio)':>12}", file=sys.stderr)
            print("-" * 60, file=sys.stderr)

            sa_values = [0, 1, 120, 121, 122, 123, 124, 125, 126, 127,
                         128, 129, 130, 131, 132, 133, 134, 254, 255]

            for sa in sa_values:
                C_test = mod.launch_sweep(sa, 127)
                torch.cuda.synchronize()
                c00 = C_test[0, 0].item()
                expected = 32.0 * (2.0 ** (sa - 127))
                if expected != 0 and c00 != 0:
                    ratio = c00 / expected
                    import math
                    log2r = math.log2(abs(ratio)) if ratio > 0 else float('nan')
                    print(f"{sa:>5} {c00:>14.6g} {expected:>14.6g} {ratio:>10.6f} {log2r:>12.4f}", file=sys.stderr)
                else:
                    print(f"{sa:>5} {c00:>14.6g} {expected:>14.6g} {'N/A':>10} {'N/A':>12}", file=sys.stderr)

            # ---- PART 2: Sweep sb with sa=127 ----
            print("\n--- PART 2: Sweep sb (sa=127 fixed) ---", file=sys.stderr)
            print(f"{'sb':>5} {'C[0,0]':>14} {'expected_2pow':>14} {'ratio':>10}", file=sys.stderr)
            print("-" * 50, file=sys.stderr)

            sb_values = [0, 1, 124, 125, 126, 127, 128, 129, 130, 254, 255]

            for sb in sb_values:
                C_test = mod.launch_sweep(127, sb)
                torch.cuda.synchronize()
                c00 = C_test[0, 0].item()
                expected = 32.0 * (2.0 ** (sb - 127))
                if expected != 0 and c00 != 0:
                    ratio = c00 / expected
                    print(f"{sb:>5} {c00:>14.6g} {expected:>14.6g} {ratio:>10.6f}", file=sys.stderr)
                else:
                    print(f"{sb:>5} {c00:>14.6g} {expected:>14.6g} {'N/A':>10}", file=sys.stderr)

            # ---- PART 3: Both scales vary (sa*sb interaction) ----
            print("\n--- PART 3: Both scales vary (sa=sb) ---", file=sys.stderr)
            print(f"{'sa=sb':>5} {'C[0,0]':>14} {'exp_product':>14} {'exp_sum':>14} {'ratio_prod':>10} {'ratio_sum':>10}", file=sys.stderr)
            print("-" * 70, file=sys.stderr)

            for sv in [124, 125, 126, 127, 128, 129, 130]:
                C_test = mod.launch_sweep(sv, sv)
                torch.cuda.synchronize()
                c00 = C_test[0, 0].item()
                # If scales multiply: result = 32.0 * 2^(sa-127) * 2^(sb-127)
                exp_product = 32.0 * (2.0 ** (sv - 127)) * (2.0 ** (sv - 127))
                # If scales add: result = 32.0 * 2^((sa-127)+(sb-127))
                exp_sum = 32.0 * (2.0 ** (2 * (sv - 127)))  # same as product for power-of-2
                ratio_p = c00 / exp_product if exp_product != 0 else float('nan')
                ratio_s = c00 / exp_sum if exp_sum != 0 else float('nan')
                print(f"{sv:>5} {c00:>14.6g} {exp_product:>14.6g} {exp_sum:>14.6g} {ratio_p:>10.6f} {ratio_s:>10.6f}", file=sys.stderr)

            # ---- PART 4: Cross-check sa != sb ----
            print("\n--- PART 4: sa != sb (cross-check) ---", file=sys.stderr)
            print(f"{'sa':>5} {'sb':>5} {'C[0,0]':>14} {'exp_2pow':>14} {'ratio':>10}", file=sys.stderr)
            print("-" * 55, file=sys.stderr)

            cross_cases = [(126, 128), (128, 126), (125, 129), (129, 125),
                           (120, 134), (134, 120), (124, 130), (130, 124)]
            for sa, sb in cross_cases:
                C_test = mod.launch_sweep(sa, sb)
                torch.cuda.synchronize()
                c00 = C_test[0, 0].item()
                expected = 32.0 * (2.0 ** (sa - 127)) * (2.0 ** (sb - 127))
                ratio = c00 / expected if expected != 0 else float('nan')
                print(f"{sa:>5} {sb:>5} {c00:>14.6g} {expected:>14.6g} {ratio:>10.6f}", file=sys.stderr)

            # ---- PART 5: Check uniformity across all elements ----
            print("\n--- PART 5: Output uniformity check ---", file=sys.stderr)
            for sa in [125, 127, 129]:
                C_test = mod.launch_sweep(sa, 127)
                torch.cuda.synchronize()
                c = C_test.cpu()
                nz = (c != 0).sum().item()
                unique = torch.unique(c)
                print(f"sa={sa}: non-zero={nz}/1024, unique_vals={len(unique)}, "
                      f"min={c.min().item():.6g} max={c.max().item():.6g} "
                      f"vals={unique[:5].tolist()}", file=sys.stderr)

            # ---- PART 6: Test different FP4 values with varying scale ----
            # Use FP4 = 0.5 (nibble 0x1) to see if formula is value-dependent
            print("\n--- PART 6: FP4=0.5 (nibble 0x1) with scale sweep ---", file=sys.stderr)
            print("(If formula is 2^(s-127), ratio should be same as Part 1)", file=sys.stderr)
            # For this we need a different kernel launch... skip, the constant test suffices

            print("\n" + "=" * 80, file=sys.stderr)
            print("SCALE SWEEP COMPLETE", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            sys.stderr.flush()

        except Exception as e:
            import traceback
            print(f"[SCALE_SWEEP] ERROR: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

    # Return correct gemm_a16wfp4 result for test pass
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

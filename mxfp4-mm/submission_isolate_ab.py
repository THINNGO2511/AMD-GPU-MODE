#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Diagnostic: Isolate whether A or B operand has wrong register-to-matrix-position
mapping in the 32x32x64 MFMA FP4 instruction.

Three tests, all with trivial scale=127, half-fill pattern:
  Test 1: Varying A (row 0=2.0, rows 1-31=0.5), Constant B (all 1.0)
  Test 2: Constant A (all 1.0), Varying B (col 0=2.0, cols 1-31=0.5)
  Test 3: Per-row varying A (cycle 0.5,1.0,1.5,2.0), Constant B (all 1.0)

Output as float32 to avoid precision loss.
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

// FP4 E2M1 nibble values:
//   0x0 = 0.0
//   0x1 = 0.5
//   0x2 = 1.0
//   0x3 = 1.5
//   0x4 = 2.0
//   0x5 = 3.0
//   0x6 = 4.0
//   0x7 = 6.0

// Helper: fill 32 bytes with a single fp4x2 byte value (first 16 active, last 16 zero)
__device__ void fill_halfreg(uint8_t* buf, uint8_t fp4x2_val) {
    for (int i = 0; i < 16; i++) buf[i] = fp4x2_val;
    for (int i = 16; i < 32; i++) buf[i] = 0x00;
}

// =====================================================================
// TEST 1: Varying A, Constant B
// A: row 0 = all FP4 2.0 (nibble 0x4), rows 1-31 = all FP4 0.5 (0x1)
// B: all FP4 1.0 (nibble 0x2)
// Half-fill: bytes 0-15 = 32 FP4 active, bytes 16-31 = zero.
// Each half of wavefront contributes K=32 to separate K ranges => K_effective=64.
// (Confirmed: all-1.0 const test produces 64.0 with this pattern.)
// Expected: C[0,j] = 2.0*1.0*64 = 128.0, C[1..31,j] = 0.5*1.0*64 = 32.0
// =====================================================================
__global__ __launch_bounds__(64)
void test1_kernel(float* __restrict__ C_out) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int half = tid / 32;

    // A operand: lane determines the A row.
    // Row 0 (lane==0): all FP4 2.0 = nibble 0x4, packed pair = 0x44
    // Rows 1-31: all FP4 0.5 = nibble 0x1, packed pair = 0x11
    uint8_t a_bytes[32];
    if (lane == 0) {
        fill_halfreg(a_bytes, 0x44);  // 2.0 pairs
    } else {
        fill_halfreg(a_bytes, 0x11);  // 0.5 pairs
    }
    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // B operand: all FP4 1.0 = nibble 0x2, packed pair = 0x22
    uint8_t b_bytes[32];
    fill_halfreg(b_bytes, 0x22);  // 1.0 pairs
    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    v16f acc = {};
    unsigned scale_a = 127u;
    unsigned scale_b = 127u;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc, 4, 4, 0, scale_a, 0, scale_b);

    // Store output: 32x32 matrix
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = half * 4 + j + i * 8;
            int col = lane;
            C_out[row * 32 + col] = acc[i * 4 + j];
        }
    }
}

// =====================================================================
// TEST 2: Constant A, Varying B
// A: all FP4 1.0 (nibble 0x2)
// B: col 0 = all FP4 2.0 (nibble 0x4), cols 1-31 = all FP4 0.5 (0x1)
// Expected: C[i,0] = 1.0*2.0*64 = 128.0, C[i,1..31] = 1.0*0.5*64 = 32.0
// =====================================================================
__global__ __launch_bounds__(64)
void test2_kernel(float* __restrict__ C_out) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int half = tid / 32;

    // A operand: all FP4 1.0
    uint8_t a_bytes[32];
    fill_halfreg(a_bytes, 0x22);  // 1.0 pairs
    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // B operand: lane determines the B column.
    // Col 0 (lane==0): all FP4 2.0 = nibble 0x4, packed pair = 0x44
    // Cols 1-31: all FP4 0.5 = nibble 0x1, packed pair = 0x11
    uint8_t b_bytes[32];
    if (lane == 0) {
        fill_halfreg(b_bytes, 0x44);  // 2.0 pairs
    } else {
        fill_halfreg(b_bytes, 0x11);  // 0.5 pairs
    }
    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    v16f acc = {};
    unsigned scale_a = 127u;
    unsigned scale_b = 127u;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc, 4, 4, 0, scale_a, 0, scale_b);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = half * 4 + j + i * 8;
            int col = lane;
            C_out[row * 32 + col] = acc[i * 4 + j];
        }
    }
}

// =====================================================================
// TEST 3: Per-row varying A, constant B
// A: row i has FP4 value V(i) where V(i) = {0.5, 1.0, 1.5, 2.0} for i%4
// B: all FP4 1.0
// Expected: C[i,j] = V(i%4) * 1.0 * 64
//   i%4==0: 0.5*64=32.0
//   i%4==1: 1.0*64=64.0
//   i%4==2: 1.5*64=96.0
//   i%4==3: 2.0*64=128.0
// =====================================================================
__global__ __launch_bounds__(64)
void test3_kernel(float* __restrict__ C_out) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int half = tid / 32;

    // A operand: lane determines the A row.
    // V(lane%4): 0.5=0x1, 1.0=0x2, 1.5=0x3, 2.0=0x4
    uint8_t nibble;
    switch (lane % 4) {
        case 0: nibble = 0x1; break;  // 0.5
        case 1: nibble = 0x2; break;  // 1.0
        case 2: nibble = 0x3; break;  // 1.5
        case 3: nibble = 0x4; break;  // 2.0
        default: nibble = 0x0; break;
    }
    uint8_t fp4x2_val = (nibble << 4) | nibble;  // pack same value in both nibbles
    uint8_t a_bytes[32];
    fill_halfreg(a_bytes, fp4x2_val);
    v8i a_reg;
    __builtin_memcpy(&a_reg, a_bytes, 32);

    // B operand: all FP4 1.0
    uint8_t b_bytes[32];
    fill_halfreg(b_bytes, 0x22);  // 1.0 pairs
    v8i b_reg;
    __builtin_memcpy(&b_reg, b_bytes, 32);

    v16f acc = {};
    unsigned scale_a = 127u;
    unsigned scale_b = 127u;

    acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, acc, 4, 4, 0, scale_a, 0, scale_b);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            int row = half * 4 + j + i * 8;
            int col = lane;
            C_out[row * 32 + col] = acc[i * 4 + j];
        }
    }
}

torch::Tensor launch_test1() {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(test1_kernel, grid, block, 0, 0, (float*)C.data_ptr());
    return C;
}

torch::Tensor launch_test2() {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(test2_kernel, grid, block, 0, 0, (float*)C.data_ptr());
    return C;
}

torch::Tensor launch_test3() {
    auto C = torch::zeros({32, 32}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(test3_kernel, grid, block, 0, 0, (float*)C.data_ptr());
    return C;
}
"""

CPP_FWD = """
torch::Tensor launch_test1();
torch::Tensor launch_test2();
torch::Tensor launch_test3();
"""

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="mfma_isolate_ab_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_test1", "launch_test2", "launch_test3"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
    return _mod


def _check_test(name, C, expected_fn):
    """Check test results and print diagnostics."""
    c = C.cpu()
    print(f"\n=== {name} ===", file=sys.stderr)
    # Print key rows with expected values
    for row in [0, 1, 2, 3, 4, 8, 16, 31]:
        exp_vals = [expected_fn(row, col) for col in range(4)]
        print(f"C[{row:2d}, 0:4]  = {c[row, 0:4].tolist()}  (expected {exp_vals})", file=sys.stderr)
    # Also check column variation
    print(f"C[0, 28:32]= {c[0, 28:32].tolist()}", file=sys.stderr)
    print(f"Min={c.min().item():.4f}  Max={c.max().item():.4f}  Mean={c.mean().item():.4f}", file=sys.stderr)

    # Check expected values
    all_pass = True
    n_mismatch = 0
    first_mismatch = None
    for row in range(32):
        for col in range(32):
            exp = expected_fn(row, col)
            got = c[row, col].item()
            if abs(got - exp) > 0.5:
                n_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = (row, col, got, exp)
                all_pass = False

    if all_pass:
        print(f"RESULT: PASS -- A/B mapping CORRECT for this test", file=sys.stderr)
    else:
        r, co, got, exp = first_mismatch
        print(f"FIRST MISMATCH: C[{r},{co}] = {got:.2f}, expected {exp:.2f}", file=sys.stderr)
        print(f"RESULT: FAIL ({n_mismatch}/1024 mismatched)", file=sys.stderr)

        # Print full row 0, row 1, col 0 for analysis
        print(f"Full row 0:  {c[0, :].tolist()}", file=sys.stderr)
        print(f"Full row 1:  {c[1, :].tolist()}", file=sys.stderr)
        print(f"Full col 0:  {c[:, 0].tolist()}", file=sys.stderr)

        # Unique values analysis
        uniq = torch.unique(c).tolist()
        if len(uniq) <= 20:
            print(f"Unique values: {uniq}", file=sys.stderr)
        else:
            print(f"Unique values ({len(uniq)}): first 20 = {uniq[:20]}", file=sys.stderr)

        # Ratio analysis: actual/expected
        ratios = []
        for row in range(32):
            for col in range(32):
                exp = expected_fn(row, col)
                got = c[row, col].item()
                if exp != 0:
                    ratios.append(got / exp)
        if ratios:
            import statistics
            print(f"Actual/Expected ratio: min={min(ratios):.4f} max={max(ratios):.4f} "
                  f"mean={statistics.mean(ratios):.4f} stdev={statistics.stdev(ratios):.6f}", file=sys.stderr)

    sys.stderr.flush()
    return all_pass


_ran_diag = False

def custom_kernel(data: input_t) -> output_t:
    global _ran_diag
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]

    if not _ran_diag:
        _ran_diag = True
        try:
            mod = _get_mod()

            # --- Test 1: Varying A, Constant B ---
            C1 = mod.launch_test1()
            torch.cuda.synchronize()
            # Expected: C[0,j]=128.0 (2.0*1.0*64), C[1..31,j]=32.0 (0.5*1.0*64)
            _check_test("TEST 1: Varying A (row0=2.0, rest=0.5), Const B (1.0)",
                        C1, lambda r, c: 128.0 if r == 0 else 32.0)

            # --- Test 2: Constant A, Varying B ---
            C2 = mod.launch_test2()
            torch.cuda.synchronize()
            # Expected: C[i,0]=128.0 (1.0*2.0*64), C[i,1..31]=32.0 (1.0*0.5*64)
            _check_test("TEST 2: Const A (1.0), Varying B (col0=2.0, rest=0.5)",
                        C2, lambda r, c: 128.0 if c == 0 else 32.0)

            # --- Test 3: Per-row varying A, constant B ---
            C3 = mod.launch_test3()
            torch.cuda.synchronize()
            # Expected: C[i,j] = V(i%4) * 64
            # V = {0.5, 1.0, 1.5, 2.0} for i%4 = {0, 1, 2, 3}
            vals = {0: 0.5*64, 1: 1.0*64, 2: 1.5*64, 3: 2.0*64}
            _check_test("TEST 3: Per-row A (0.5/1.0/1.5/2.0 cycle), Const B (1.0)",
                        C3, lambda r, c: vals[r % 4])

            print("\n=== SUMMARY ===", file=sys.stderr)
            print("If Test1 FAILS but Test2 PASSES: A operand mapping is wrong", file=sys.stderr)
            print("If Test1 PASSES but Test2 FAILS: B operand mapping is wrong", file=sys.stderr)
            print("If BOTH FAIL: both mappings wrong OR scale/K-dim issue", file=sys.stderr)
            print("If BOTH PASS: mapping is correct, error source is elsewhere", file=sys.stderr)
            sys.stderr.flush()
        except Exception as e:
            import traceback
            print(f"DIAGNOSTIC ERROR: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()

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

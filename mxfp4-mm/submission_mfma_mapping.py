#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Probe MFMA thread-to-output mapping by using A=identity, B=identity patterns.
This tells us exactly where each thread's output goes in the 32x32 C matrix.
"""
import torch, os, subprocess, ctypes, hashlib
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x32_t = fp4x2_t __attribute__((ext_vector_type(32)));
typedef float fp32x16_t __attribute__((ext_vector_type(16)));

// Test 1: Set A row 0 = all 1.0, rest = 0. B col 0 = all 1.0, rest = 0.
// Output should be C[0,0] = K (sum of 1.0*1.0 over K=64 elements)
// This reveals which c_reg indices correspond to which C[row,col] positions.

// Test 2: For each thread, write tid and c_reg index to output.
// This directly maps (tid, reg_index) -> (row, col).

__global__ __launch_bounds__(64)
void probe_output_mapping(float* __restrict__ C_flat) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {}, b_reg = {};
    fp32x16_t c_reg = {};

    // Set all A = 1.0, B = 1.0 (fp4 nibble 0x2)
    for (int i = 0; i < 16; i++) {
        a_reg[i] = 0x22;
        b_reg[i] = 0x22;
    }

    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, 127, 0, 127);

    // Now c_reg has 16 values. All should be 64.0 (since all inputs are 1.0).
    // But WHERE in the 32x32 output do they go?

    // Write (tid * 100 + reg_index) to each output position.
    // Then read the 32x32 output to determine the mapping.
    // Use a second MFMA with identity pattern instead:

    // Actually, let's use a simpler approach:
    // Set A[row r] = delta(r, tid%32) for row selection
    // Set B[col c] = delta(c, some_pattern)
    // But fp4 doesn't have enough precision for identity...

    // Best approach: just write where each thread's output goes
    // Each thread writes its 16 values to a known location
    // Then we reconstruct the mapping.
    for (int i = 0; i < 16; i++) {
        // Encode: tid * 16 + i (unique identifier per output element)
        C_flat[tid * 16 + i] = c_reg[i];  // all 64.0
    }
}

// Test 3: Use specific patterns to determine row/col mapping
// Set A so that only row r has value 1.0, all other rows = 0
// Set B so that all columns have value 1.0
// Then C[r, :] should be non-zero, all other rows = 0
__global__ __launch_bounds__(64)
void probe_row_mapping(float* __restrict__ C_flat, int target_row) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {}, b_reg = {};
    fp32x16_t c_reg = {};

    // A: only the thread whose row matches target_row gets non-zero
    // Thread tid%32 corresponds to row tid%32 (from Salykova reference)
    if ((tid % 32) == target_row) {
        for (int i = 0; i < 16; i++) a_reg[i] = 0x22;  // all 1.0
    }

    // B: all 1.0
    for (int i = 0; i < 16; i++) b_reg[i] = 0x22;

    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, 127, 0, 127);

    for (int i = 0; i < 16; i++) {
        C_flat[tid * 16 + i] = c_reg[i];
    }
}

// Test 4: Determine which c_reg indices map to which (row, col)
// Use A=all-ones, B=column-indicator (only col c has non-zero)
__global__ __launch_bounds__(64)
void probe_col_mapping(float* __restrict__ C_flat, int target_col) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {}, b_reg = {};
    fp32x16_t c_reg = {};

    // A: all 1.0
    for (int i = 0; i < 16; i++) a_reg[i] = 0x22;

    // B: only threads that "own" column target_col get non-zero
    // From Salykova: B thread mapping is (tid%32)/2 for byte, tid%2 for nibble
    // So thread tid "owns" column: tid%32 (for tids 0-31 and 32-63)
    // Actually the column ownership is more subtle...
    // Let's try: set b_reg[0] to have a specific nibble pattern
    // For simplicity, just set all B to 1.0 for now
    for (int i = 0; i < 16; i++) b_reg[i] = 0x22;

    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, 127, 0, 127);

    for (int i = 0; i < 16; i++) {
        C_flat[tid * 16 + i] = c_reg[i];
    }
}

// Test 5: THE KEY TEST - use accumulator pattern to identify mapping
// Initialize C to identity-like pattern, then run MFMA with A=B=0
// C stays unchanged, revealing which (tid, reg_idx) maps to which (row, col)
__global__ __launch_bounds__(64)
void probe_c_identity(float* __restrict__ C_flat) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {}, b_reg = {};
    fp32x16_t c_reg = {};

    // Set c_reg to unique values: tid * 16 + i
    for (int i = 0; i < 16; i++) {
        c_reg[i] = (float)(tid * 16 + i);
    }

    // Run MFMA with A=B=0 (no change to C)
    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, 127, 0, 127);

    // Output: each thread writes its 16 values
    for (int i = 0; i < 16; i++) {
        C_flat[tid * 16 + i] = c_reg[i];
    }
}

// Test 6: Use the Salykova store pattern to write to a 32x32 matrix
// Then read back and verify
__global__ __launch_bounds__(64)
void probe_salykova_store(float* __restrict__ C_32x32) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {}, b_reg = {};
    fp32x16_t c_reg = {};

    // Set c_reg to unique values
    for (int i = 0; i < 16; i++) {
        c_reg[i] = (float)(tid * 16 + i);
    }

    // A=B=0, C unchanged
    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg, 4, 4, 0, 127, 0, 127);

    // Store using Salykova pattern
    for (int i = 0; i < 4; i++) {
        C_32x32[tid % 32 + (tid / 32) * 4 * 32 + i * 32 * 8] = c_reg[i * 4];
        C_32x32[tid % 32 + (tid / 32) * 4 * 32 + 32 * 1 + i * 32 * 8] = c_reg[i * 4 + 1];
        C_32x32[tid % 32 + (tid / 32) * 4 * 32 + 32 * 2 + i * 32 * 8] = c_reg[i * 4 + 2];
        C_32x32[tid % 32 + (tid / 32) * 4 * 32 + 32 * 3 + i * 32 * 8] = c_reg[i * 4 + 3];
    }
}

extern "C" {
    void run_probe_c_identity(void* C_flat) {
        hipLaunchKernelGGL(probe_c_identity, dim3(1), dim3(64), 0, 0, (float*)C_flat);
    }
    void run_probe_salykova(void* C_32x32) {
        hipLaunchKernelGGL(probe_salykova_store, dim3(1), dim3(64), 0, 0, (float*)C_32x32);
    }
    void run_probe_row(void* C_flat, int row) {
        hipLaunchKernelGGL(probe_row_mapping, dim3(1), dim3(64), 0, 0, (float*)C_flat, row);
    }
}
"""

_lib = None

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mfma_map_{h}.hip'
    so = f'/tmp/_mfma_map_{h}.so'
    with open(src, 'w') as f:
        f.write(HIP_SRC)
    hipcc = 'hipcc'
    for p in ['/opt/rocm/bin/hipcc', '/opt/rocm/hip/bin/hipcc']:
        if os.path.exists(p):
            hipcc = p
            break
    r = subprocess.run(
        [hipcc, '-shared', '-fPIC', '-O3', '--offload-arch=gfx950',
         '-std=c++17', '-o', so, src],
        capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"COMPILE FAILED:\n{r.stderr[:2000]}")
        return None
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.run_probe_c_identity.argtypes = [ctypes.c_void_p]
    _lib.run_probe_c_identity.restype = None
    _lib.run_probe_salykova.argtypes = [ctypes.c_void_p]
    _lib.run_probe_salykova.restype = None
    _lib.run_probe_row.argtypes = [ctypes.c_void_p, ctypes.c_int]
    _lib.run_probe_row.restype = None
    return _lib

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    lib = _compile()

    if lib is not None:
        # Test: C identity - reveals (tid, reg_idx) values after MFMA pass-through
        c_flat = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
        lib.run_probe_c_identity(c_flat.data_ptr())
        torch.cuda.synchronize()
        cf = c_flat.cpu()

        # Each element should be tid*16+i (unchanged since A=B=0)
        print("=== C Identity Test (A=B=0, C=tid*16+i) ===")
        expected = torch.arange(64*16, dtype=torch.float32)
        match = (cf == expected).all().item()
        print(f"Identity preserved: {match}")
        if not match:
            mismatches = (cf != expected).nonzero().flatten()[:10]
            for idx in mismatches:
                print(f"  [{idx.item()}]: expected={expected[idx.item()]:.0f} got={cf[idx.item()]:.0f}")

        # Salykova store pattern: write to 32x32 matrix
        c_32x32 = torch.zeros(32 * 32, dtype=torch.float32, device='cuda')
        lib.run_probe_salykova(c_32x32.data_ptr())
        torch.cuda.synchronize()
        cm = c_32x32.cpu().view(32, 32)

        print(f"\n=== Salykova Store Pattern (32x32 matrix) ===")
        # Each element should be tid*16+reg_idx. We can decode (row,col) -> (tid, reg_idx)
        print("First 8 rows, first 8 cols:")
        for r in range(8):
            vals = [f"{cm[r,c].item():5.0f}" for c in range(8)]
            print(f"  row {r}: {' '.join(vals)}")

        # Decode the mapping: for each (row, col), find tid and reg_idx
        print(f"\nMapping decode (row,col) -> (tid, reg_idx):")
        for r in range(4):
            for c in range(4):
                val = int(cm[r, c].item())
                tid = val // 16
                reg = val % 16
                print(f"  C[{r},{c}] = {val} -> tid={tid}, reg_idx={reg}")

        # Row mapping: A[row0]=1, rest=0
        print(f"\n=== Row Mapping (A[row0]=1, B=all-ones) ===")
        c_flat2 = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
        lib.run_probe_row(c_flat2.data_ptr(), 0)
        torch.cuda.synchronize()
        cf2 = c_flat2.cpu()
        # Find which (tid, reg_idx) pairs are non-zero
        nonzero = (cf2.abs() > 0.1).nonzero().flatten()
        print(f"Non-zero positions (row=0 active): {len(nonzero)} values")
        for idx in nonzero[:16]:
            tid = idx.item() // 16
            reg = idx.item() % 16
            print(f"  tid={tid}, reg={reg}, val={cf2[idx.item()]:.1f}")

    # Reference output
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

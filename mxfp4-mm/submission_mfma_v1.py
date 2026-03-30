#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MFMA FP4 kernel using __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4.
Key insight: the intrinsic name includes "scale" and "f8f6f4", with Atype=Btype=4 for FP4.
Requires: #include <hip/hip_ext_ocp.h>
"""
import torch, os, subprocess, ctypes, hashlib
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>

// Try to include the OCP header for MFMA fp4 types
#if __has_include(<hip/hip_ext_ocp.h>)
#include <hip/hip_ext_ocp.h>
#define HAS_OCP 1
#else
#define HAS_OCP 0
#endif

// Fallback type definitions if header not available
#ifndef HAS_OCP
typedef uint8_t fp4x2_t;
#else
using fp4x2_t = __amd_fp4x2_storage_t;
#endif

using fp4x32_t = fp4x2_t __attribute__((ext_vector_type(32)));
typedef float fp32x16_t __attribute__((ext_vector_type(16)));

// Simple test: compile and run the MFMA intrinsic with known pattern
__global__ __launch_bounds__(64)
void test_mfma_scaled(float* __restrict__ output, int* __restrict__ status) {
    int tid = threadIdx.x;

    fp4x32_t a_reg = {};
    fp4x32_t b_reg = {};
    fp32x16_t c_reg = {};

    // Fill A and B with fp4 value 1.0 (nibble=0x2, packed byte=0x22)
    for (int i = 0; i < 32; i++) {
        a_reg[i] = 0x22;  // two fp4 1.0 values per byte
        b_reg[i] = 0x22;
    }
    // But only first 16 bytes matter for fp4 (128 bits), rest is padding
    for (int i = 16; i < 32; i++) {
        a_reg[i] = 0;
        b_reg[i] = 0;
    }

    // Scale = 127 means 2^(127-127) = 1.0 (no scaling)
    uint8_t scale_a = 127, scale_b = 127;

    // Call the MFMA intrinsic
    // Atype=4 (FP4 E2M1), Btype=4 (FP4 E2M1)
    // OPSEL_A=0, OPSEL_B=0
    c_reg = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c_reg,
        4,        // Atype = FP4 E2M1
        4,        // Btype = FP4 E2M1
        0,        // OPSEL_A
        scale_a,  // scale for A
        0,        // OPSEL_B
        scale_b   // scale for B
    );

    if (tid == 0) *status = 1;

    // Write output: each thread writes 16 float values
    for (int i = 0; i < 16; i++) {
        output[tid * 16 + i] = c_reg[i];
    }
}

extern "C" {
    void run_test(void* output, void* status) {
        hipLaunchKernelGGL(test_mfma_scaled, dim3(1), dim3(64), 0, 0,
            (float*)output, (int*)status);
    }
}
"""

_lib = None

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mfma_v1_{h}.hip'
    so = f'/tmp/_mfma_v1_{h}.so'
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
        # Try to find what headers exist
        for hdr in ['hip/hip_ext_ocp.h', 'hip/hip_fp4.h', 'hip/amd_detail/amd_hip_fp4.h',
                     'hip/amd_detail/amd_hip_ext_ocp.h']:
            full = f'/opt/rocm/include/{hdr}'
            if os.path.exists(full):
                print(f"  EXISTS: {full}")
                # Show first 30 lines
                with open(full) as fh:
                    lines = fh.readlines()[:30]
                    for l in lines:
                        print(f"    {l.rstrip()}")
            else:
                print(f"  MISSING: {full}")
        # Also search for any fp4 or mfma_scale headers
        r2 = subprocess.run(['find', '/opt/rocm/include', '-name', '*.h', '-exec',
                            'grep', '-l', 'mfma_scale', '{}', ';'],
                           capture_output=True, text=True, timeout=30)
        if r2.stdout:
            print(f"\nHeaders with mfma_scale:\n{r2.stdout[:500]}")
        r3 = subprocess.run(['find', '/opt/rocm/include', '-name', '*ocp*', '-o', '-name', '*fp4*'],
                           capture_output=True, text=True, timeout=30)
        if r3.stdout:
            print(f"\nFP4/OCP headers:\n{r3.stdout[:500]}")
        return None

    print("MFMA COMPILATION SUCCESSFUL!")
    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.run_test.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.run_test.restype = None
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
        output = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
        status = torch.zeros(1, dtype=torch.int32, device='cuda')
        lib.run_test(output.data_ptr(), status.data_ptr())
        torch.cuda.synchronize()

        s = status.item()
        o = output.cpu().tolist()
        print(f"\n=== MFMA FP4 Test ===")
        print(f"Status: {s}")
        print(f"Thread 0 output[0:16]: {o[:16]}")
        print(f"Thread 1 output[0:16]: {o[16:32]}")
        nonzero = sum(1 for v in o if abs(v) > 1e-10)
        print(f"Non-zero values: {nonzero}/{len(o)}")
        # With all-1.0 fp4 inputs, 32x32 tile, K=64:
        # Each output C[i,j] = sum_k A[i,k]*B[k,j] = 64 * 1.0 * 1.0 = 64.0
        print(f"Expected (all 1.0 inputs, K=64): 64.0")
        print(f"Range: [{min(o):.2f}, {max(o):.2f}]")

    # Reference output
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe: Test MFMA fp4 intrinsics on gfx950. Find the right intrinsic and data layout."""
import torch, os, subprocess, ctypes, hashlib
from task import input_t, output_t

# Try multiple MFMA intrinsic names - gfx950 might use different naming
HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>

// Try to find available MFMA intrinsics by checking compilation
// On gfx950, the fp4 MFMA should be available

// Test 1: Basic MFMA availability check
// Use known gfx950 MFMA: v_mfma_f32_32x32x64 with fp4 inputs
// The intrinsic might be:
//   __builtin_amdgcn_mfma_f32_32x32x64_fp4
// Input: 2 x i64 for A, 2 x i64 for B, 16 x float for C
// Or it might need a different type signature

typedef float float16 __attribute__((ext_vector_type(16)));
typedef long long v2i64 __attribute__((ext_vector_type(2)));
typedef int v4i32 __attribute__((ext_vector_type(4)));

// Kernel 1: Test if MFMA compiles and run with identity-like pattern
__global__ __launch_bounds__(64)
void test_mfma(float* __restrict__ output, int* __restrict__ status) {
    int tid = threadIdx.x;  // 0-63 within wave64

    // Initialize output accumulator to zero
    float16 c = {0};

    // Create simple test pattern for A and B
    // Each thread provides 2 x i64 = 16 bytes = 32 fp4 values
    // For test: fill with known pattern
    v2i64 a, b;

    // Fill A: thread 0 gets (1,0,0,...), thread 1 gets (0,1,0,...), etc.
    // Actually, let's just fill with a simple incrementing pattern
    // fp4 value 0x22 = byte with two fp4 values of 1.0 each (index 2 = 1.0)
    // fp4 value 0x00 = two zeros
    // Let's use all-ones pattern: 0x22 repeated = all fp4 values are 1.0

    // Pack 16 bytes with fp4 value = 1.0 (nibble = 0x2)
    // 0x22 = two fp4 1.0 values per byte
    unsigned long long pattern = 0x2222222222222222ULL;
    a[0] = pattern;
    a[1] = pattern;
    b[0] = pattern;
    b[1] = pattern;

    // Try the MFMA intrinsic
    // Different possible names for gfx950:
#if defined(__gfx950__)
    c = __builtin_amdgcn_mfma_f32_32x32x64_fp4(a, b, c);
    if (tid == 0) *status = 1;  // success
#else
    // Try anyway - the arch flag might not define __gfx950__
    c = __builtin_amdgcn_mfma_f32_32x32x64_fp4(a, b, c);
    if (tid == 0) *status = 2;  // compiled without __gfx950__ define
#endif

    // Write output: each thread writes its 16 float values
    for (int i = 0; i < 16; i++) {
        output[tid * 16 + i] = c[i];
    }
}

// Kernel 2: Probe thread-to-matrix mapping
// Set A to identity-like pattern, B to all-ones
// This reveals which A elements map to which output positions
__global__ __launch_bounds__(64)
void probe_mapping(float* __restrict__ output, float* __restrict__ a_debug) {
    int tid = threadIdx.x;

    float16 c = {0};
    v2i64 a, b;

    // B = all fp4 1.0 values
    unsigned long long ones = 0x2222222222222222ULL;
    b[0] = ones;
    b[1] = ones;

    // A = fp4 1.0 only for this thread's first value, rest zero
    // This isolates which output elements this thread's A values contribute to
    if (tid == 0) {
        // Thread 0: first fp4 value = 1.0, rest = 0
        a[0] = 0x0000000000000002ULL;  // only lowest nibble = 1.0
        a[1] = 0ULL;
    } else if (tid == 1) {
        a[0] = 0x0000000000000002ULL;
        a[1] = 0ULL;
    } else {
        a[0] = 0ULL;
        a[1] = 0ULL;
    }

    c = __builtin_amdgcn_mfma_f32_32x32x64_fp4(a, b, c);

    for (int i = 0; i < 16; i++) {
        output[tid * 16 + i] = c[i];
    }

    // Also dump what A values each thread holds
    unsigned char* a_bytes = (unsigned char*)&a;
    for (int i = 0; i < 16; i++) {
        a_debug[tid * 16 + i] = (float)a_bytes[i];
    }
}

extern "C" {
    void run_test_mfma(void* output, void* status) {
        dim3 block(64);  // one wave64
        dim3 grid(1);
        hipLaunchKernelGGL(test_mfma, grid, block, 0, 0,
            (float*)output, (int*)status);
    }

    void run_probe_mapping(void* output, void* a_debug) {
        dim3 block(64);
        dim3 grid(1);
        hipLaunchKernelGGL(probe_mapping, grid, block, 0, 0,
            (float*)output, (float*)a_debug);
    }
}
"""

_lib = None

def _compile():
    global _lib
    if _lib is not None:
        return _lib
    h = hashlib.md5(HIP_SRC.encode()).hexdigest()[:8]
    src = f'/tmp/_mfma_probe_{h}.hip'
    so = f'/tmp/_mfma_probe_{h}.so'
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
        print(f"hipcc FAILED:\n{r.stderr[:3000]}")
        # Try alternative intrinsic names
        print(f"\nTrying to find available MFMA intrinsics...")
        # Search ROCm headers
        for hdr in ['/opt/rocm/include/hip/amd_detail/amd_hip_fp4.h',
                     '/opt/rocm/include/hip/amd_detail/amd_hip_unsafe_atomics.h',
                     '/opt/rocm/lib/llvm/lib/clang/19/include/__clang_hip_math.h']:
            if os.path.exists(hdr):
                content = open(hdr).read()
                if 'mfma' in content.lower():
                    print(f"  Found MFMA refs in {hdr}")
        # Search for intrinsic headers
        r2 = subprocess.run(['find', '/opt/rocm/include', '-name', '*.h', '-exec',
                            'grep', '-l', 'mfma_f32', '{}', ';'],
                           capture_output=True, text=True, timeout=30)
        if r2.stdout:
            print(f"  Headers with mfma_f32: {r2.stdout[:1000]}")

        # Also check what builtins are available
        test_src = """
#include <hip/hip_runtime.h>
__global__ void test() {
    // Try various intrinsic names
    typedef float float16 __attribute__((ext_vector_type(16)));
    typedef long long v2i64 __attribute__((ext_vector_type(2)));
    float16 c = {0};
    v2i64 a = {0}, b = {0};
    // This will fail with the specific error telling us the right name
    c = __builtin_amdgcn_mfma_f32_32x32x64_fp4(a, b, c);
}
"""
        test_path = '/tmp/_mfma_test.hip'
        with open(test_path, 'w') as tf:
            tf.write(test_src)
        r3 = subprocess.run([hipcc, '-c', '--offload-arch=gfx950', '-o', '/dev/null', test_path],
                           capture_output=True, text=True, timeout=30)
        print(f"\nDirect intrinsic test stderr:\n{r3.stderr[:2000]}")

        return None

    _lib = ctypes.cdll.LoadLibrary(so)
    _lib.run_test_mfma.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.run_test_mfma.restype = None
    _lib.run_probe_mapping.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    _lib.run_probe_mapping.restype = None
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
        # Test 1: Basic MFMA
        output = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
        status = torch.zeros(1, dtype=torch.int32, device='cuda')
        lib.run_test_mfma(output.data_ptr(), status.data_ptr())
        torch.cuda.synchronize()

        s = status.item()
        print(f"\n=== MFMA Test ===")
        print(f"Status: {s} (1=success with __gfx950__, 2=success without)")

        # Print output for thread 0
        o = output.cpu().tolist()
        print(f"Thread 0 output (16 values): {o[:16]}")
        print(f"Thread 1 output (16 values): {o[16:32]}")

        # Check if any non-zero output
        nonzero = sum(1 for v in o if abs(v) > 1e-10)
        print(f"Total non-zero values: {nonzero} / {len(o)}")

        # Expected: with all-ones A and B (fp4 1.0), each output element should be
        # sum of 64 products of 1.0*1.0 = 64.0
        print(f"Expected value (if all 1.0 inputs): 64.0")
        print(f"Actual range: [{min(o):.2f}, {max(o):.2f}]")

        # Test 2: Probe mapping (if test 1 works)
        if nonzero > 0:
            output2 = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
            a_dbg = torch.zeros(64 * 16, dtype=torch.float32, device='cuda')
            lib.run_probe_mapping(output2.data_ptr(), a_dbg.data_ptr())
            torch.cuda.synchronize()
            o2 = output2.cpu().tolist()
            print(f"\n=== Mapping Probe (thread 0 A=[1,0,...], B=all-ones) ===")
            for t in range(4):
                vals = o2[t*16:(t+1)*16]
                nonz = [i for i, v in enumerate(vals) if abs(v) > 1e-10]
                print(f"  Thread {t} output: non-zero at indices {nonz}")
                if nonz:
                    print(f"    values: {[f'{vals[i]:.2f}' for i in nonz]}")

    # Use reference for output
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

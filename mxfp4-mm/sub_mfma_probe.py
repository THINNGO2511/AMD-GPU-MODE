#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MFMA 16x16x128 FP4 data-mapping probe.

Three tests using __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4:
  TEST 0: All A=0x11 (FP4 1.0 pairs), All B=0x11, scales=127 (neutral)
          => Reveals baseline output for uniform inputs
  TEST 1: Only thread 0 has A=0x11, rest A=0x00. B=0x11 for all.
          => Reveals thread 0's A contribution to the output matrix
  TEST 2: Thread 0 K-group 0 (bytes 0-3) = 0x11, rest = 0x00. B=0x11 for all.
          => Reveals which rows/cols K-group 0 of lane 0 maps to

Each test: 64 threads x 4 output floats = 256 floats written to debug buffer.
Total debug buffer: 3 tests x 256 = 768 floats.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <stdint.h>

typedef int   __attribute__((ext_vector_type(8))) int8v;  // 256-bit for MFMA intrinsic
typedef int   __attribute__((ext_vector_type(4))) int4v;  // convenience
typedef float __attribute__((ext_vector_type(4))) float4v;

// 16x16x128 MFMA FP4: 9 args
// a(int4v=128bits=64FP4), b(int4v), c(float4v), cbsz_a, cbsz_b, cbsz_sel, scale_a, op_sel, scale_b
// cbsz_a=4 => FP4 for A, cbsz_b=4 => FP4 for B

__global__ __launch_bounds__(64, 1)
void mfma_probe_kernel(float* __restrict__ dbg)
{
    const int lane = threadIdx.x;  // 0..63

    // ===================== TEST 0: Uniform A=0x11, B=0x11, scale=127 =====================
    {
        int8v a_reg = {}, b_reg = {};  // 256-bit, upper half stays zero
        // Fill all 16 bytes (128 bits) with 0x11 = FP4 pair (1.0, 1.0)
        uint8_t* a_bytes = (uint8_t*)&a_reg;
        uint8_t* b_bytes = (uint8_t*)&b_reg;
        for (int i = 0; i < 16; i++) {
            a_bytes[i] = 0x11;
            b_bytes[i] = 0x11;
        }
        float4v acc = {};
        uint32_t sa = 127, sb = 127;

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

        dbg[0 * 256 + lane * 4 + 0] = acc[0];
        dbg[0 * 256 + lane * 4 + 1] = acc[1];
        dbg[0 * 256 + lane * 4 + 2] = acc[2];
        dbg[0 * 256 + lane * 4 + 3] = acc[3];
    }

    // ===================== TEST 1: Only thread 0 A=0x11, rest A=0x00 =====================
    {
        int8v a_reg = {}, b_reg = {};  // test 1/2 variant
        uint8_t* a_bytes = (uint8_t*)&a_reg;
        uint8_t* b_bytes = (uint8_t*)&b_reg;
        // A: only thread 0 gets nonzero
        if (lane == 0) {
            for (int i = 0; i < 16; i++) a_bytes[i] = 0x11;
        }
        // B: all threads get 0x11
        for (int i = 0; i < 16; i++) b_bytes[i] = 0x11;

        float4v acc = {};
        uint32_t sa = 127, sb = 127;

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

        dbg[1 * 256 + lane * 4 + 0] = acc[0];
        dbg[1 * 256 + lane * 4 + 1] = acc[1];
        dbg[1 * 256 + lane * 4 + 2] = acc[2];
        dbg[1 * 256 + lane * 4 + 3] = acc[3];
    }

    // ===================== TEST 2: Thread 0, bytes 0-3 only = 0x11 =====================
    {
        int8v a_reg = {}, b_reg = {};  // test 1/2 variant
        uint8_t* a_bytes = (uint8_t*)&a_reg;
        uint8_t* b_bytes = (uint8_t*)&b_reg;
        // A: only thread 0, only first 4 bytes (K-group 0 = 8 FP4 values)
        if (lane == 0) {
            a_bytes[0] = 0x11;
            a_bytes[1] = 0x11;
            a_bytes[2] = 0x11;
            a_bytes[3] = 0x11;
            // bytes 4..15 stay 0
        }
        // B: all threads get 0x11
        for (int i = 0; i < 16; i++) b_bytes[i] = 0x11;

        float4v acc = {};
        uint32_t sa = 127, sb = 127;

        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);

        dbg[2 * 256 + lane * 4 + 0] = acc[0];
        dbg[2 * 256 + lane * 4 + 1] = acc[1];
        dbg[2 * 256 + lane * 4 + 2] = acc[2];
        dbg[2 * 256 + lane * 4 + 3] = acc[3];
    }
}

torch::Tensor launch_probe()
{
    auto dbg = torch::zeros({768}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(mfma_probe_kernel, grid, block, 0, 0,
        (float*)dbg.data_ptr());
    return dbg;
}
"""

CPP_FWD = "torch::Tensor launch_probe();"

_mod = None

def _get_mod():
    global _mod
    if _mod is not None:
        return _mod
    from torch.utils.cpp_extension import load_inline
    _mod = load_inline(
        name="mfma_probe_v1",
        cpp_sources=CPP_FWD,
        cuda_sources=HIP_SRC,
        functions=["launch_probe"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
    )
    return _mod


def run_probe():
    mod = _get_mod()
    dbg = mod.launch_probe()
    torch.cuda.synchronize()
    dbg = dbg.cpu().numpy()

    test_names = [
        "TEST 0: Uniform A=0x11, B=0x11, scale=127",
        "TEST 1: Only thread0 A=0x11, rest A=0x00, B=0x11",
        "TEST 2: Thread0 bytes[0:4]=0x11 only, B=0x11",
    ]

    for t in range(3):
        base = t * 256
        vals = dbg[base:base + 256]
        print(f"\n{'='*72}")
        print(f"  {test_names[t]}")
        print(f"{'='*72}")

        # Print per-thread: 64 threads x 4 values
        for tid in range(64):
            v = vals[tid * 4: tid * 4 + 4]
            tag = ""
            if all(x == 0.0 for x in v):
                tag = " [ZERO]"
            print(f"  T{tid:02d}: [{v[0]:12.4f}, {v[1]:12.4f}, {v[2]:12.4f}, {v[3]:12.4f}]{tag}")

        # Summary: count unique nonzero patterns
        nonzero_threads = []
        patterns = {}
        for tid in range(64):
            v = tuple(vals[tid * 4: tid * 4 + 4])
            if any(x != 0.0 for x in v):
                nonzero_threads.append(tid)
                key = v
                if key not in patterns:
                    patterns[key] = []
                patterns[key].append(tid)

        print(f"\n  Summary: {len(nonzero_threads)}/64 threads have nonzero output")
        print(f"  Unique nonzero patterns: {len(patterns)}")
        for pat, tids in sorted(patterns.items(), key=lambda x: x[1][0]):
            tid_str = ",".join(str(t) for t in tids[:8])
            if len(tids) > 8:
                tid_str += f"...({len(tids)} total)"
            print(f"    [{pat[0]:12.4f},{pat[1]:12.4f},{pat[2]:12.4f},{pat[3]:12.4f}] -> T[{tid_str}]")


# Entry point for popcorn submission (wraps as custom_kernel)
def custom_kernel(A, B, B_q, B_shuffle, B_scale):
    """Run probe and return dummy output to satisfy eval harness."""
    run_probe()
    m, k = A.shape
    n = B.shape[0]
    return torch.zeros(m, n, dtype=torch.bfloat16, device=A.device)


if __name__ == "__main__":
    # Local test: just run the probe
    print("Building MFMA probe kernel...")
    run_probe()
    print("\nDone.")

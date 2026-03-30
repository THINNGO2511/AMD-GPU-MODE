#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Scale probe: test MFMA FP4 scale routing with known inputs.
A = all 1.0 FP4, B = all 1.0 FP4, vary scales to verify behavior.
"""
from task import input_t, output_t
import torch

_probed = False

def _run_scale_probe():
    global _probed
    if _probed:
        return
    _probed = True

    import os
    os.environ['PYTORCH_ROCM_ARCH'] = 'gfx950'

    from torch.utils.cpp_extension import load_inline

    cpp_sources = """
    torch::Tensor scale_probe(int test_id);
    """

    cuda_sources = r"""
    #include <torch/extension.h>
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
    #include <hip/hip_bfloat16.h>
    #include <cstdio>

    // FP4 E2M1 encoding: 1.0 = 0b0010 = 0x2
    // In a byte, two FP4 nibbles packed: low nibble + high nibble
    // All 1.0 => each byte = 0x22

    // MFMA f32_32x32x64_f8f6f4:
    //   A: 32 threads, each holds 16 bytes = 32 FP4 values (packed in 4 x i32)
    //   B: 32 threads, each holds 16 bytes = 32 FP4 values (packed in 4 x i32)
    //   C accumulator: 32x32 = 1024 floats, 16 per thread (in first wave)
    //   K dimension = 64 per instruction
    //   With 64 lanes and 32 FP4 values per lane: 64 lanes * 32 = 2048 values
    //   Actually: each lane provides 16 bytes = 32 nibbles, but K=64 means
    //   64 FP4 multiplies per output element.
    //
    // cbsz=4 means A is FP4, blgp=4 means B is FP4
    //
    // Scale args to __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4:
    //   (a_data, b_data, c_accum, cbsz, blgp,
    //    scale_a_sel, scale_a, scale_b_sel, scale_b)
    //
    // scale_a, scale_b are i32 containing E8M0 scale values
    // E8M0: bias-127 encoding. 127 = 2^0 = 1.0, 128 = 2^1 = 2.0

    __global__ void probe_kernel(float* out, int scale_a_val, int scale_b_val) {
        // Each thread in the wavefront contributes to the MFMA
        int lane = threadIdx.x;  // 0..63

        // Pack A data: all 1.0 FP4 = 0x22222222 per i32 (8 FP4 values per i32)
        // 4 x i32 = 16 bytes = 32 FP4 values per lane
        int a0 = 0x22222222;
        int a1 = 0x22222222;
        int a2 = 0x22222222;
        int a3 = 0x22222222;

        // Pack B data: same
        int b0 = 0x22222222;
        int b1 = 0x22222222;
        int b2 = 0x22222222;
        int b3 = 0x22222222;

        // Zero accumulator: 16 floats per thread
        float c[16];
        for (int i = 0; i < 16; i++) c[i] = 0.0f;

        // Build the 128-bit (4 x i32) operands via inline asm
        // Use the scale variant of MFMA
        // __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 expects:
        //   long a (8 bytes), long b (8 bytes), float[16] c,
        //   int cbsz, int blgp,
        //   int scale_a_sel, int scale_a, int scale_b_sel, int scale_b
        //
        // But the actual intrinsic takes v_pk operands.
        // Let's use inline ASM directly for the MFMA instruction.

        // Actually: the builtin takes __attribute__((ext_vector_type(8))) int for a,b
        // Let's try the asm approach which is more reliable.

        // v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[20:23], v[0:15],
        //    op_sel_hi:[0,0], cbsz:4, blgp:4, v24, v25

        // Use inline asm
        asm volatile(
            // Move scale values into sgpr/vgpr
            "v_mov_b32 v16, %[a0]\n"
            "v_mov_b32 v17, %[a1]\n"
            "v_mov_b32 v18, %[a2]\n"
            "v_mov_b32 v19, %[a3]\n"
            "v_mov_b32 v20, %[b0]\n"
            "v_mov_b32 v21, %[b1]\n"
            "v_mov_b32 v22, %[b2]\n"
            "v_mov_b32 v23, %[b3]\n"
            // Zero accum in v0..v15
            "v_mov_b32 v0,  %[c0]\n"
            "v_mov_b32 v1,  %[c1]\n"
            "v_mov_b32 v2,  %[c2]\n"
            "v_mov_b32 v3,  %[c3]\n"
            "v_mov_b32 v4,  %[c4]\n"
            "v_mov_b32 v5,  %[c5]\n"
            "v_mov_b32 v6,  %[c6]\n"
            "v_mov_b32 v7,  %[c7]\n"
            "v_mov_b32 v8,  %[c8]\n"
            "v_mov_b32 v9,  %[c9]\n"
            "v_mov_b32 v10, %[c10]\n"
            "v_mov_b32 v11, %[c11]\n"
            "v_mov_b32 v12, %[c12]\n"
            "v_mov_b32 v13, %[c13]\n"
            "v_mov_b32 v14, %[c14]\n"
            "v_mov_b32 v15, %[c15]\n"
            // Scale registers
            "v_mov_b32 v24, %[sa]\n"
            "v_mov_b32 v25, %[sb]\n"
            // The MFMA instruction
            "v_mfma_scale_f32_32x32x64_f8f6f4 v[0:15], v[16:19], v[20:23], v[0:15], v24, v25 op_sel_hi:[0,0] cbsz:4 blgp:4\n"
            // Read results back
            "v_mov_b32 %[c0],  v0\n"
            "v_mov_b32 %[c1],  v1\n"
            "v_mov_b32 %[c2],  v2\n"
            "v_mov_b32 %[c3],  v3\n"
            "v_mov_b32 %[c4],  v4\n"
            "v_mov_b32 %[c5],  v5\n"
            "v_mov_b32 %[c6],  v6\n"
            "v_mov_b32 %[c7],  v7\n"
            "v_mov_b32 %[c8],  v8\n"
            "v_mov_b32 %[c9],  v9\n"
            "v_mov_b32 %[c10], v10\n"
            "v_mov_b32 %[c11], v11\n"
            "v_mov_b32 %[c12], v12\n"
            "v_mov_b32 %[c13], v13\n"
            "v_mov_b32 %[c14], v14\n"
            "v_mov_b32 %[c15], v15\n"
            : [c0]"+v"(c[0]),  [c1]"+v"(c[1]),  [c2]"+v"(c[2]),  [c3]"+v"(c[3]),
              [c4]"+v"(c[4]),  [c5]"+v"(c[5]),  [c6]"+v"(c[6]),  [c7]"+v"(c[7]),
              [c8]"+v"(c[8]),  [c9]"+v"(c[9]),  [c10]"+v"(c[10]), [c11]"+v"(c[11]),
              [c12]"+v"(c[12]), [c13]"+v"(c[13]), [c14]"+v"(c[14]), [c15]"+v"(c[15])
            : [a0]"v"(a0), [a1]"v"(a1), [a2]"v"(a2), [a3]"v"(a3),
              [b0]"v"(b0), [b1]"v"(b1), [b2]"v"(b2), [b3]"v"(b3),
              [sa]"v"(scale_a_val), [sb]"v"(scale_b_val)
            : "v0","v1","v2","v3","v4","v5","v6","v7",
              "v8","v9","v10","v11","v12","v13","v14","v15",
              "v16","v17","v18","v19","v20","v21","v22","v23",
              "v24","v25"
        );

        // Write results: lane writes its 16 floats
        for (int i = 0; i < 16; i++) {
            out[lane * 16 + i] = c[i];
        }
    }

    torch::Tensor scale_probe(int test_id) {
        // Allocate output: 64 lanes * 16 floats = 1024 floats
        auto out = torch::zeros({1024}, torch::dtype(torch::kFloat32).device(torch::kCUDA));

        int scale_a, scale_b;
        const char* label;
        switch (test_id) {
            case 0: scale_a = 127; scale_b = 127; label = "sa=127(1.0), sb=127(1.0)"; break;
            case 1: scale_a = 128; scale_b = 127; label = "sa=128(2.0), sb=127(1.0)"; break;
            case 2: scale_a = 127; scale_b = 128; label = "sa=127(1.0), sb=128(2.0)"; break;
            case 3: scale_a = 128; scale_b = 128; label = "sa=128(2.0), sb=128(2.0)"; break;
            default: scale_a = 127; scale_b = 127; label = "default"; break;
        }

        printf("Test %d: %s\n", test_id, label);

        // Launch exactly 1 wavefront = 64 threads
        hipLaunchKernelGGL(probe_kernel, dim3(1), dim3(64), 0, 0,
                           out.data_ptr<float>(), scale_a, scale_b);
        hipDeviceSynchronize();

        return out;
    }
    """

    module = load_inline(
        name='scale_probe',
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=['scale_probe'],
        extra_cuda_cflags=['-mcpu=gfx950', '-O2'],
        with_cuda=True,
        is_python_module=True,
    )

    print("=" * 60)
    print("MFMA FP4 SCALE PROBE")
    print("=" * 60)
    print("FP4 E2M1: 1.0 = 0b0010 = nibble 0x2")
    print("E8M0 scale: 127 = 2^0 = 1.0, 128 = 2^1 = 2.0")
    print("MFMA 32x32x64: K=64 FP4 muls per output element")
    print("Expected: all 1.0 * all 1.0 * K=64 => C[i,j] = 64.0")
    print("=" * 60)

    expected = [64.0, 128.0, 128.0, 256.0]
    labels = [
        "sa=1.0, sb=1.0 => expect 64",
        "sa=2.0, sb=1.0 => expect 128",
        "sa=1.0, sb=2.0 => expect 128",
        "sa=2.0, sb=2.0 => expect 256",
    ]

    for test_id in range(4):
        result = module.scale_probe(test_id)
        # Reshape to 64 lanes x 16 elements
        r = result.cpu().reshape(64, 16)
        # The 32x32 output matrix is distributed across lanes
        # Print first few lanes' values
        vals = r[:4, :4]
        unique = torch.unique(result.cpu())
        print(f"\nTest {test_id}: {labels[test_id]}")
        print(f"  Unique values: {unique.tolist()}")
        print(f"  Lane 0 first 4: {r[0, :4].tolist()}")
        print(f"  Lane 1 first 4: {r[1, :4].tolist()}")
        if len(unique) == 1 and unique[0].item() == expected[test_id]:
            print(f"  PASS: all elements = {expected[test_id]}")
        elif len(unique) == 1:
            print(f"  WRONG: all elements = {unique[0].item()}, expected {expected[test_id]}")
            print(f"  Ratio to expected: {unique[0].item() / expected[test_id]:.4f}")
        else:
            print(f"  MIXED values: min={result.min().item()}, max={result.max().item()}, mean={result.mean().item():.2f}")

    # Also test: what if scale is packed differently?
    # E8M0 scale might be per-row or per-column, let's check with different per-lane scales
    print("\n" + "=" * 60)
    print("EXTRA: Testing scale=0 (2^-127) and scale=254 (2^127)")
    print("=" * 60)

    for test_id, (sa, sb, desc) in enumerate([
        (0, 127, "sa=0 (2^-127), sb=1.0"),
        (254, 127, "sa=254 (2^127), sb=1.0"),
    ]):
        # Recompile not needed, just call with custom values
        # But our function only takes test_id... let's just report what we got
        pass

    print("\nScale probe complete.")


def custom_kernel(data: input_t) -> output_t:
    _run_scale_probe()

    # Run baseline for correctness check
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )

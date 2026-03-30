"""MFMA FP4 32x32x64 Register Mapping Probe for gfx950"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <cstdint>

__global__ void mfma_probe_kernel(float* C_out, int probe_mode) {
    int lane = threadIdx.x;
    
    // FP4 E2M1 nibble for 1.0 = 0b0010 = 2
    // Packed byte of two 1.0s = 0x22
    // 4 x uint32 of all 1.0s = 0x22222222
    
    // Each thread encodes its lane ID:
    // FP4 nibble = lane & 7 (encodes 0-7 as fp4 values 0,0.5,1,1.5,2,3,4,6)
    uint8_t nib = (uint8_t)(lane & 7);
    uint8_t byt = (nib << 4) | nib;
    uint32_t w = (uint32_t)byt | ((uint32_t)byt << 8) | ((uint32_t)byt << 16) | ((uint32_t)byt << 24);
    
    typedef int int8v __attribute__((ext_vector_type(8)));
    typedef float float16v __attribute__((ext_vector_type(16)));

    int8v a_reg, b_reg;
    uint32_t sa, sb;

    if (probe_mode == 0) {
        // A = all 1.0 with scale=1, B = lane-unique with scale = 2^(lane/8)
        a_reg = (int8v){(int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u,
                        (int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u};
        b_reg = (int8v){(int)w, (int)w, (int)w, (int)w, (int)w, (int)w, (int)w, (int)w};
        sa = 127;  // 2^0 = 1.0
        sb = (uint32_t)(127 + (lane >> 3));
    } else if (probe_mode == 1) {
        // A = lane-unique with scale = 2^(lane/8), B = all 1.0
        a_reg = (int8v){(int)w, (int)w, (int)w, (int)w, (int)w, (int)w, (int)w, (int)w};
        b_reg = (int8v){(int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u,
                        (int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u};
        sa = (uint32_t)(127 + (lane >> 3));
        sb = 127;
    } else if (probe_mode == 2) {
        // PROBE 2: A=1.0, B has DIFFERENT value per int32 slot
        // Slot 0: all 0.5 (nib=1), Slot 1: all 1.0 (nib=2), ...
        // This reveals which int32 maps to which K-positions
        // FP4 values: 0=0, 1=0.5, 2=1.0, 3=1.5, 4=2.0, 5=3.0, 6=4.0, 7=6.0
        a_reg = (int8v){(int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u,
                        (int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u};
        // Each int32 has a different nibble (but same nibble for all 8 positions within the int32)
        b_reg = (int8v){
            (int)0x11111111u,  // slot 0: nib=1 (0.5) x8 nibbles
            (int)0x22222222u,  // slot 1: nib=2 (1.0) x8
            (int)0x33333333u,  // slot 2: nib=3 (1.5) x8
            (int)0x44444444u,  // slot 3: nib=4 (2.0) x8
            (int)0x55555555u,  // slot 4: nib=5 (3.0) x8
            (int)0x66666666u,  // slot 5: nib=6 (4.0) x8
            (int)0x77777777u,  // slot 6: nib=7 (6.0) x8
            (int)0x11111111u   // slot 7: nib=1 (0.5) x8 (repeat to distinguish from slot 0)
        };
        sa = 127;
        sb = 127;
    } else {
        // PROBE 3: A has DIFFERENT value per int32 slot, B=1.0
        a_reg = (int8v){
            (int)0x11111111u, (int)0x22222222u, (int)0x33333333u, (int)0x44444444u,
            (int)0x55555555u, (int)0x66666666u, (int)0x77777777u, (int)0x11111111u
        };
        b_reg = (int8v){(int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u,
                        (int)0x22222222u, (int)0x22222222u, (int)0x22222222u, (int)0x22222222u};
        sa = 127;
        sb = 127;
    }

    float16v c = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

    c = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
        a_reg, b_reg, c, 4, 4, 0, sa, 0, sb);
    
    for (int v = 0; v < 16; v++)
        C_out[lane * 16 + v] = c[v];
}

torch::Tensor run_mfma_probe(int probe_mode) {
    auto C = torch::zeros({64 * 16}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    dim3 grid(1);
    dim3 block(64);
    hipLaunchKernelGGL(mfma_probe_kernel, grid, block, 0, 0, (float*)C.data_ptr(), probe_mode);
    return C;
}
"""

CPP_FWD = "torch::Tensor run_mfma_probe(int probe_mode);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(
            name="mfma_probe_v4",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["run_mfma_probe"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
    return _mod

def custom_kernel(data):
    A, B, B_q, B_shuffle, B_scale_sh = data
    mod = _get_mod()
    
    r0 = mod.run_mfma_probe(0).cpu()
    r1 = mod.run_mfma_probe(1).cpu()
    
    print("=== PROBE 0: A=1.0, B=lane-encoded (key threads only) ===")
    for t in [0, 1, 2, 7, 8, 9, 31, 32, 33]:
        vals = [r0[t*16+v].item() for v in range(16)]
        print(f"T{t:02d}: " + " ".join(f"{v:8.2f}" for v in vals))

    print("=== PROBE 1: A=lane-encoded, B=1.0 (key threads only) ===")
    for t in [0, 1, 2, 7, 31, 32, 33, 63]:
        vals = [r1[t*16+v].item() for v in range(16)]
        print(f"T{t:02d}: " + " ".join(f"{v:8.2f}" for v in vals))

    r2 = mod.run_mfma_probe(2).cpu()
    print("=== PROBE 2: A=1.0, B=per-slot-unique (K-dim mapping) ===")
    for t in [0, 1, 2, 31, 32, 33]:
        vals = [r2[t*16+v].item() for v in range(16)]
        print(f"T{t:02d}: " + " ".join(f"{v:8.2f}" for v in vals))

    r3 = mod.run_mfma_probe(3).cpu()
    print("=== PROBE 3: A=per-slot-unique, B=1.0 (A K-dim mapping) ===")
    for t in [0, 1, 2, 31, 32, 33]:
        vals = [r3[t*16+v].item() for v in range(16)]
        print(f"T{t:02d}: " + " ".join(f"{v:8.2f}" for v in vals))

    M, K = A.shape
    N = B.shape[0]
    return torch.zeros(M, N, dtype=torch.bfloat16, device=A.device)

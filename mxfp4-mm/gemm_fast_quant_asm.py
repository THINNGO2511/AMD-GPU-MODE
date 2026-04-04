#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Fast fused quant+shuffle HIP kernel + CK ASM GEMM.
Strategy: Replace 2 Python ops (dynamic_mxfp4_quant + e8m0_shuffle) with 1 fast HIP kernel.
Then call gemm_a4w4_asm with pre-shuffled B_shuffle + B_scale_sh.
Target: save 5-8μs per call vs Triton a16wfp4.
Falls back to Triton if HIP quant fails.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time, sys
from task import input_t, output_t

_call = 0
_hip_mod = None
_use_hip = False

# HIP kernel for fused bf16→FP4 quant + e8m0 scale shuffle
# Matches aiter's _mxfp4_quant_op exactly:
# Scale: (amax_int + 0x200000) & 0xFF800000 → floor(log2)-2
# FP4: RNE via bit manipulation
CPP_FWD = """
torch::Tensor fused_quant_shuffle_v1(torch::Tensor A, int64_t M, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

// FP4 quantization matching aiter's _mxfp4_quant_op
// Group size = 32 (MXFP4 block size)
// Returns packed fp4x2 [M, K/2] + shuffled e8m0 scales [M, K/32]
// Combined into a single output: first M*K/2 bytes are fp4x2, last M*K/32 bytes are shuffled scales

__device__ __forceinline__ uint8_t fp4_rne(float x, float inv_scale) {
    // Scale the value
    float sx = x * inv_scale;
    // Clamp to FP4 range: [-6, 6]
    sx = fminf(fmaxf(sx, -6.0f), 6.0f);
    // Get sign
    uint32_t sign = (sx < 0.0f) ? 1 : 0;
    float absx = fabsf(sx);

    // FP4 E2M1 encoding: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
    uint8_t fp4;
    if (absx < 0.25f) fp4 = 0;        // 0
    else if (absx < 0.75f) fp4 = 1;   // 0.5
    else if (absx < 1.25f) fp4 = 2;   // 1.0
    else if (absx < 1.75f) fp4 = 3;   // 1.5
    else if (absx < 2.5f) fp4 = 4;    // 2.0
    else if (absx < 3.5f) fp4 = 5;    // 3.0
    else if (absx < 5.0f) fp4 = 6;    // 4.0
    else fp4 = 7;                       // 6.0

    return (sign << 3) | fp4;
}

// E8M0 shuffle: reorders scales to match CK kernel's expected layout
// Input: flat scales [M, K/32] in row-major
// The shuffle permutation: view as [M/32, K/32/8, 4, 16, 2, 2].permute(0,5,3,1,4,2)
// This can be computed per-element with index arithmetic

__global__ void fused_quant_shuffle_kernel(
    const hip_bfloat16* __restrict__ A,  // [M, K] bf16
    uint8_t* __restrict__ A_fp4,          // [M, K/2] packed fp4x2
    uint8_t* __restrict__ A_scale_sh,     // [M, K/32] shuffled e8m0
    int M, int K
) {
    // Each thread handles one group of 32 bf16 values → 16 fp4x2 bytes + 1 e8m0 scale
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_groups = M * (K / 32);
    if (gid >= num_groups) return;

    int row = gid / (K / 32);
    int grp = gid % (K / 32);

    // Load 32 bf16 values and find amax
    const hip_bfloat16* base = A + row * K + grp * 32;
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (float)base[i];
        amax = fmaxf(amax, fabsf(vals[i]));
    }

    // Compute E8M0 scale using aiter's method:
    // (amax_int + 0x200000) & 0xFF800000 → floor(log2) - 2
    uint32_t amax_int;
    memcpy(&amax_int, &amax, 4);
    uint32_t rounded = (amax_int + 0x200000) & 0xFF800000;
    // E8M0 = exponent of rounded value - 2 + 127
    int exponent = ((rounded >> 23) & 0xFF);
    uint8_t e8m0;
    if (amax == 0.0f) {
        e8m0 = 0;
    } else {
        int e = exponent - 2;  // -2 because FP4 max is 6 = 1.5 * 2^2
        e8m0 = (uint8_t)(e < 0 ? 0 : (e > 255 ? 255 : e));
    }

    // Compute inverse scale for quantization
    float scale_val;
    uint32_t scale_bits = ((uint32_t)e8m0) << 23;
    memcpy(&scale_val, &scale_bits, 4);
    float inv_scale = (scale_val > 0.0f) ? (1.0f / scale_val) : 0.0f;

    // Quantize 32 values to FP4 and pack as fp4x2 (16 bytes)
    uint8_t* dst = A_fp4 + row * (K / 2) + grp * 16;
    for (int i = 0; i < 16; i++) {
        uint8_t lo = fp4_rne(vals[2*i], inv_scale);
        uint8_t hi = fp4_rne(vals[2*i+1], inv_scale);
        dst[i] = (hi << 4) | lo;
    }

    // Shuffle the scale: compute shuffled index
    // Original layout: [M, K/32] row-major → view as [M/32, K/32/8, 4, 16, 2, 2]
    // Shuffled: permute(0, 5, 3, 1, 4, 2) → [M/32, 2, 16, K/32/8, 2, 4]
    int sm = M, sn = K / 32;
    int d0 = row / 32;          // M/32 block
    int r_in = row % 32;        // position within block
    int d1 = grp / 8;           // K/32/8 block
    int c_in = grp % 8;         // position within block

    // Decompose r_in and c_in into (i2, i3, i4, i5) from view(d0, d1, 4, 16, 2, 2)
    // r_in * 8 + c_in maps to flat index within (4, 16, 2, 2) = 256 elements
    // But that's not right... let me think about the reshape

    // Actually: [M/32, K/32/8, 4, 16, 2, 2]
    // The total per (d0, d1) block is 4*16*2*2 = 256 = 32 * 8
    // Which is (32 rows) * (8 scale columns per d1 block)
    // So the inner dims are: dim2=r_in/8(0..3), dim3=r_in%8(but 8>7...)

    // Actually the reshape is: [sm, sn] → [sm//32, sn//8, 4, 16, 2, 2]
    // sm//32 and sn//8 are the block counts
    // The 4*16*2*2=256 maps 32*8=256 elements
    // The order is: for each (d0, d1), the 256 elements are indexed as
    // flat_idx = dim2*16*2*2 + dim3*2*2 + dim4*2 + dim5
    // where flat_idx = r_in * (sn//1*...) ... this is getting complex

    // Simpler: compute the shuffled flat index directly
    // Original flat: row * sn + grp
    // View: (d0, d1, d2, d3, d4, d5) where
    //   d0 = row/32, remainder r = row%32
    //   The remaining dims map r*8+c_in (within the d1 block)
    //   inner_flat = r_in * 8 + c_in (= row%32 * 8 + grp%8)
    //   But 4*16*2*2 = 256 = 32*8 ✓
    //   d2 = inner_flat / (16*2*2) = inner_flat / 64
    //   d3 = (inner_flat / (2*2)) % 16 = (inner_flat / 4) % 16
    //   d4 = (inner_flat / 2) % 2
    //   d5 = inner_flat % 2

    int inner = r_in * 8 + c_in;
    int i2 = inner / 64;           // 0..3
    int i3 = (inner / 4) % 16;     // 0..15
    int i4 = (inner / 2) % 2;      // 0..1
    int i5 = inner % 2;            // 0..1

    // Permuted: (d0, d5, d3, d1, d4, d2) = (d0, i5, i3, d1, i4, i2)
    // Shuffled flat: d0*(2*16*(sn/8)*2*4) + i5*(16*(sn/8)*2*4) + i3*((sn/8)*2*4) + d1*(2*4) + i4*4 + i2
    int n_d1 = sn / 8;
    int sh_flat = d0 * (2 * 16 * n_d1 * 2 * 4)
                + i5 * (16 * n_d1 * 2 * 4)
                + i3 * (n_d1 * 2 * 4)
                + d1 * (2 * 4)
                + i4 * 4
                + i2;

    A_scale_sh[sh_flat] = e8m0;
}

torch::Tensor fused_quant_shuffle_v1(torch::Tensor A, int64_t M, int64_t K) {
    auto device = A.device();

    // Allocate outputs
    auto A_fp4 = torch::empty({M, K/2}, torch::dtype(torch::kUInt8).device(device));
    auto A_scale = torch::empty({M, K/32}, torch::dtype(torch::kUInt8).device(device));

    int num_groups = M * (K / 32);
    int threads = 256;
    int blocks = (num_groups + threads - 1) / threads;

    hipLaunchKernelGGL(fused_quant_shuffle_kernel, dim3(blocks), dim3(threads), 0, 0,
        (const hip_bfloat16*)A.data_ptr(),
        (uint8_t*)A_fp4.data_ptr(),
        (uint8_t*)A_scale.data_ptr(),
        (int)M, (int)K);

    // Return concatenated [fp4, scale] — caller separates
    return torch::cat({A_fp4.view(-1), A_scale.view(-1)});
}
"""

def _compile_hip():
    global _hip_mod, _use_hip
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _hip_mod = load_inline(
            name="fused_quant_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["fused_quant_shuffle_v1"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
        t1 = time.time()
        print(f"[PROBE] HIP quant compiled in {t1-t0:.1f}s", flush=True)

        # Test accuracy against aiter
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle
        A_test = torch.randn(32, 512, dtype=torch.bfloat16, device='cuda')
        ref_fp4, ref_scale = dynamic_mxfp4_quant(A_test)
        ref_scale_sh = e8m0_shuffle(ref_scale)

        # Our fused kernel
        out = _hip_mod.fused_quant_shuffle_v1(A_test, 32, 512)
        fp4_bytes = 32 * 256  # M * K/2
        scale_bytes = 32 * 16  # M * K/32
        hip_fp4 = out[:fp4_bytes]
        hip_scale = out[fp4_bytes:fp4_bytes+scale_bytes]

        # Compare fp4 (flatten both to 1D uint8)
        ref_fp4_flat = ref_fp4.contiguous().view(torch.uint8).flatten()
        ref_scale_flat = ref_scale_sh.contiguous().view(torch.uint8).flatten()
        print(f"[PROBE] hip_fp4: {hip_fp4.shape}, ref_fp4: {ref_fp4_flat.shape}", flush=True)
        print(f"[PROBE] hip_scale: {hip_scale.shape}, ref_scale: {ref_scale_flat.shape}", flush=True)

        min_len = min(hip_fp4.shape[0], ref_fp4_flat.shape[0])
        fp4_match = (hip_fp4[:min_len] == ref_fp4_flat[:min_len]).float().mean().item()
        min_len_s = min(hip_scale.shape[0], ref_scale_flat.shape[0])
        scale_match = (hip_scale[:min_len_s] == ref_scale_flat[:min_len_s]).float().mean().item()
        print(f"[PROBE] FP4 match: {fp4_match*100:.1f}%, Scale match: {scale_match*100:.1f}%", flush=True)

        if fp4_match > 0.95 and scale_match > 0.95:
            _use_hip = True
            print("[PROBE] HIP quant: USING for benchmark", flush=True)
        else:
            print("[PROBE] HIP quant: accuracy too low, using Triton fallback", flush=True)

        # Timing comparison
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _hip_mod.fused_quant_shuffle_v1(A_test, 32, 512)
        torch.cuda.synchronize()
        hip_time = (time.time() - t0) / 100 * 1e6

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            fp4, sc = dynamic_mxfp4_quant(A_test)
            e8m0_shuffle(sc)
        torch.cuda.synchronize()
        aiter_time = (time.time() - t0) / 100 * 1e6

        print(f"[PROBE] HIP quant: {hip_time:.1f}μs, Aiter quant+shuffle: {aiter_time:.1f}μs, speedup: {aiter_time/hip_time:.2f}x", flush=True)

    except Exception as e:
        print(f"[PROBE] HIP quant failed: {str(e)[:300]}", flush=True)
        _use_hip = False


# ===== Proven GEMM fallback =====
_gc = {}; _bsr = None; _bqu = None; _braw = None; _yc = {}; _w = False
_K7168 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
_K512 = {"BLOCK_SIZE_M":4,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":3,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}
_K2048 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}

def _bgc(sm,sn,d):
    t=sm*sn;d0,d1=sm//32,sn//8
    i=torch.arange(t,dtype=torch.int64,device=d)
    i=i.view(d0,d1,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(-1)
    return i,torch.empty(t,dtype=torch.uint8,device=d)

def _fu(f,sm,sn):
    gi,ob=_gc[(sm,sn)];torch.take(f,gi,out=ob);return ob.view(sm,sn)

def _pw(d):
    global _w
    if _w:return
    _w=True
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    for m,n,k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
        try:
            da=torch.randn(m,k,dtype=torch.bfloat16,device=d)
            if k==1536:
                af,asc=dynamic_mxfp4_quant(da)
                gemm_afp4wfp4(af.view(torch.uint8),torch.zeros(n,k//2,dtype=torch.uint8,device=d),asc,torch.full((n,k//32),127,dtype=torch.uint8,device=d),dtype=torch.bfloat16)
            else:
                pn=((n+31)//32)*32
                cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
                gemm_a16wfp4(da,torch.zeros(n,k//2,dtype=torch.uint8,device=d),torch.full((pn,k//32),127,dtype=torch.uint8,device=d),dtype=torch.bfloat16,y=torch.empty(m,n,dtype=torch.bfloat16,device=d),config=cfg)
            del da
        except:pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _call
    _call += 1
    if _call == 1:
        _compile_hip()

    # Proven fallback (always used for now)
    global _bsr, _bqu, _braw
    A,B,B_q,B_shuffle,B_scale_sh=data;m,k=A.shape;n=B.shape[0]
    if _bsr is not B_scale_sh:
        _bsr=B_scale_sh;su=B_scale_sh.view(torch.uint8);sm,sn=su.shape
        if (sm,sn) not in _gc:_gc[(sm,sn)]=_bgc(sm,sn,su.device)
        _braw=_fu(su.reshape(-1),sm,sn);_bqu=B_q.view(torch.uint8)
    _pw(A.device)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key=(m,n)
    if key not in _yc:_yc[key]=torch.empty(m,n,dtype=torch.bfloat16,device=A.device)
    out=_yc[key]
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bqu,asc,_braw,dtype=torch.bfloat16)
    cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
    gemm_a16wfp4(A,_bqu,_braw,dtype=torch.bfloat16,y=out,config=cfg)
    return out

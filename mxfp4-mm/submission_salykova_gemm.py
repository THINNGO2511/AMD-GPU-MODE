"""GEMM HIP kernel using EXACT salykova FP4 MFMA register layout"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch
from task import input_t, output_t

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

typedef uint8_t fp4x2_t;
typedef fp4x2_t __attribute__((ext_vector_type(16))) fp4x64_t;  // 16 bytes = 32 FP4 values... wait, 64?
// Actually fp4x64_t = 16 x fp4x2 = 16 bytes = 32 FP4 values
// But the MFMA needs 32 FP4 per thread = 16 bytes = 16 x uint8
// The intrinsic takes int __attribute__((ext_vector_type(8))) = 8 x int32 = 32 bytes = 64 FP4
// Salykova uses fp4x2_t[16] = 16 bytes... 
// BUT the builtin signature says V8Zi = 8 x int32 = 32 bytes
// The blog uses fp4x64_t which is 16 x uint8 = 16 bytes
// This works because for FP4, only 4 of 8 int32 regs carry data (others zero)
// Actually re-reading: fp4x64_t is 64 FP4 = 32 bytes = 8 x int32. Let me check.
// fp4x2_t __attribute__((ext_vector_type(16))) = 16 x uint8 = 16 bytes = 32 FP4
// But the name says fp4x64... confusing. The intrinsic needs 8 x int32.
// Let me just use the types that match the intrinsic.

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(16))) float16v;

// FP4 nibble extraction helper
__device__ __forceinline__ uint8_t extract_fp4(uint8_t packed, int idx) {
    return (idx == 0) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
}

__device__ __forceinline__ uint8_t create_fp4x2(uint8_t lo, uint8_t hi) {
    return (hi << 4) | (lo & 0x0F);
}

// bf16 <-> float
__device__ __forceinline__ float bf16_to_float(hip_bfloat16 v) {
    return (float)v;
}

// Simple FP4 E2M1 quantization for A
__device__ __forceinline__ void quant_bf16_to_fp4(
    const hip_bfloat16* src, uint8_t* dst_packed, uint8_t* scale_out,
    int count)
{
    // Find amax for this group of 32
    float amax = 0.0f;
    for (int i = 0; i < count; i++) {
        float v = fabsf((float)src[i]);
        if (v > amax) amax = v;
    }
    
    // E8M0 scale = 2^floor(log2(amax/6.0))  
    // FP4 max value is 6.0, so scale = amax/6.0 rounded to power of 2
    uint8_t scale;
    if (amax == 0.0f) {
        scale = 0;
    } else {
        // Extract exponent from amax/6.0
        float ratio = amax / 6.0f;
        int exp_bits;
        memcpy(&exp_bits, &ratio, 4);
        exp_bits = ((exp_bits >> 23) & 0xFF);
        if (exp_bits < 1) exp_bits = 1;
        if (exp_bits > 254) exp_bits = 254;
        scale = (uint8_t)exp_bits;
    }
    *scale_out = scale;
    
    float inv_scale = (scale == 0) ? 0.0f : 1.0f / ldexpf(1.0f, (int)scale - 127);
    
    // Quantize each value to FP4 E2M1
    for (int i = 0; i < count; i += 2) {
        float v0 = (i < count) ? (float)src[i] * inv_scale : 0.0f;
        float v1 = (i+1 < count) ? (float)src[i+1] * inv_scale : 0.0f;
        
        // Clamp and round to nearest E2M1
        auto to_fp4 = [](float x) -> uint8_t {
            int sign = (x < 0) ? 8 : 0;
            float ax = fabsf(x);
            uint8_t nib;
            if (ax < 0.25f) nib = 0;
            else if (ax < 0.75f) nib = 1;
            else if (ax < 1.25f) nib = 2;
            else if (ax < 1.75f) nib = 3;
            else if (ax < 2.5f) nib = 4;
            else if (ax < 3.5f) nib = 5;
            else if (ax < 5.0f) nib = 6;
            else nib = 7;
            return sign | nib;
        };
        
        dst_packed[i/2] = create_fp4x2(to_fp4(v0), to_fp4(v1));
    }
}

__global__ void gemm_fp4_kernel(
    const hip_bfloat16* __restrict__ A,   // [M, K] row-major
    const uint8_t* __restrict__ B,         // [N, K/2] row-major (fp4x2 packed)
    const uint8_t* __restrict__ B_scale,   // [N, K/32] row-major (e8m0)
    hip_bfloat16* __restrict__ C,          // [M, N] row-major
    int M, int N, int K)
{
    int tile_n = blockIdx.x;  // N-tile
    int tile_m = blockIdx.y;  // M-tile
    int lane = threadIdx.x;   // 0..63
    
    int m_base = tile_m * 32;
    int n_base = tile_n * 32;
    
    float16v acc = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
    int K_packed = K / 2;  // bytes per row in B
    int K_groups = K / 32; // scale groups per row
    
    // K-loop: step by 64 FP4 elements = 32 packed bytes
    for (int k = 0; k < K; k += 64) {
        // ---- Load A (salykova pattern) ----
        // A is [M, K] bf16, row-major
        // Thread t loads A[m_base + t%32, k + (t/32)*32 : k + (t/32)*32 + 32]
        // = 32 bf16 values -> quantize to 16 packed FP4 bytes
        int a_row = m_base + (lane % 32);
        int a_k_start = k + (lane / 32) * 32;
        
        uint8_t a_packed[16];  // 32 FP4 values
        uint8_t a_scale;
        
        if (a_row < M && a_k_start + 31 < K) {
            hip_bfloat16 a_vals[32];
            for (int i = 0; i < 32; i++) {
                a_vals[i] = A[(long long)a_row * K + a_k_start + i];
            }
            quant_bf16_to_fp4(a_vals, a_packed, &a_scale, 32);
        } else {
            for (int i = 0; i < 16; i++) a_packed[i] = 0;
            a_scale = 127;  // scale = 1.0
        }
        
        // Pack into int8v (8 x int32 = 32 bytes)
        // But we only have 16 bytes of data. Pad with zeros.
        int8v a_reg;
        // Copy 16 bytes into first 4 int32s, zero rest
        uint8_t a_buf[32] = {0};
        for (int i = 0; i < 16; i++) a_buf[i] = a_packed[i];
        memcpy(&a_reg, a_buf, 32);
        
        // ---- Load B (salykova transposed pattern) ----
        // B is [N, K/2] row-major, fp4x2 packed
        // ldg_b = B + (lane%32)/2 + (K/2) * (n_base + ...) + 16*32*(lane/32)
        // Actually salykova assumes B is [K/2, N] column-major style
        // Our B is [N, K/2] row-major. Need to adapt.
        //
        // Salykova: B + (threadIdx.x % 32) / 2 + 16 * 32 * (threadIdx.x / 32)
        // This accesses B as if it's 32 columns x K/2 rows
        // For our [N, K/2] layout:
        //   B[n, k_byte] = B + n * K_packed + k_byte
        //
        // Thread t needs column n_base + (t%32) of B
        // But salykova loads B transposed: 2 threads share a byte
        //
        // For our layout, thread t reading column n_base + col:
        //   col = lane % 32
        //   B_col_ptr = B + (n_base + col) * K_packed + k/2
        //   Load 32 bytes (64 FP4) from B_col_ptr
        
        int b_col = n_base + (lane % 32);
        int b_k_byte = k / 2;
        
        uint8_t b_buf[32] = {0};
        if (b_col < N && b_k_byte + 31 < K_packed) {
            const uint8_t* b_ptr = B + (long long)b_col * K_packed + b_k_byte;
            for (int i = 0; i < 32; i++) {
                b_buf[i] = b_ptr[i];
            }
        }
        
        // But wait — salykova's B loading is NOT a simple contiguous read.
        // It uses extract_fp4 with stride-16 access.
        // This is because B needs to be in TRANSPOSED register layout.
        //
        // For now: try simple contiguous load and see if accuracy improves.
        // If not, implement the salykova transposed pattern.
        
        int8v b_reg;
        memcpy(&b_reg, b_buf, 32);
        
        // ---- Load scales ----
        int a_group = (a_k_start / 32);  // which 32-element group
        // a_scale already computed above
        
        int b_group0 = (k / 32);       // scale for K[k:k+31]
        int b_group1 = (k / 32) + 1;   // scale for K[k+32:k+63]
        uint8_t bs0 = 127, bs1 = 127;
        if (b_col < N) {
            if (b_group0 < K_groups) bs0 = B_scale[(long long)b_col * K_groups + b_group0];
            if (b_group1 < K_groups) bs1 = B_scale[(long long)b_col * K_groups + b_group1];
        }
        
        unsigned int sa_packed = (unsigned int)a_scale;
        unsigned int sb_packed = (unsigned int)bs0 | ((unsigned int)bs1 << 8);
        
        // ---- MFMA ----
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa_packed, 0, sb_packed);
    }
    
    // ---- Store output (salykova pattern) ----
    for (int i = 0; i < 4; i++) {
        int row = m_base + (lane / 32) * 4 + i * 8;
        int col = n_base + (lane % 32);
        if (row < M && col < N) {
            C[(long long)row * N + col]     = (hip_bfloat16)acc[i * 4];
            if (row + 1 < M) C[(long long)(row+1) * N + col] = (hip_bfloat16)acc[i * 4 + 1];
            if (row + 2 < M) C[(long long)(row+2) * N + col] = (hip_bfloat16)acc[i * 4 + 2];
            if (row + 3 < M) C[(long long)(row+3) * N + col] = (hip_bfloat16)acc[i * 4 + 3];
        }
    }
}

torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B_q, torch::Tensor B_scale,
                           int64_t M, int64_t N, int64_t K) {
    auto C = torch::empty({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(64);
    
    hipLaunchKernelGGL(gemm_fp4_kernel, grid, block, 0, 0,
        (const hip_bfloat16*)A.data_ptr(),
        (const uint8_t*)B_q.data_ptr(),
        (const uint8_t*)B_scale.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),
        (int)M, (int)N, (int)K);
    
    return C;
}
"""

CPP_FWD = "torch::Tensor launch_gemm(torch::Tensor A, torch::Tensor B_q, torch::Tensor B_scale, int64_t M, int64_t N, int64_t K);"

_mod = None
def _get_mod():
    global _mod
    if _mod is not None:
        return _mod
    from torch.utils.cpp_extension import load_inline
    try:
        _mod = load_inline(
            name="salykova_gemm_v2",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["launch_gemm"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[WARN] HIP compilation failed: {e}")
        _mod = None
    return _mod

# Unshuffle B_scale
_sc = {}
def _unshuffle(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _sc: return _sc[key]
    n_sc = K // 32
    sm = ((N+255)//256)*256; sn = ((n_sc+7)//8)*8
    s = B_scale_sh.view(torch.uint8)
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N,:n_sc] = s[:N,:n_sc]
    r = p.view(sm//32, sn//8, 4, 16, 2, 2)
    u = r.permute(0,5,3,1,4,2).contiguous()
    result = u.view(sm, sn)[:N,:n_sc]
    _sc[key] = result
    return result

# Fallback
def _fallback(A, B_q, B_shuffle, B_scale_sh):
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_ss = e8m0_shuffle(A_s)
    return gemm_afp4wfp4(A_q, B_shuffle, A_ss, B_scale_sh)

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    
    mod = _get_mod()
    if mod is not None:
        B_scale = _unshuffle(B_scale_sh, N, K)
        B_q_u8 = B_q.view(torch.uint8)
        try:
            return mod.launch_gemm(A, B_q_u8, B_scale, M, N, K)
        except Exception as e:
        import traceback; traceback.print_exc()
            print(f"[WARN] HIP kernel failed: {e}, using fallback")
    
    return _fallback(A, B_q, B_shuffle, B_scale_sh)

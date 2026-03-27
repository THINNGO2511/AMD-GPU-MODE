
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from torch.utils.cpp_extension import load_inline

SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>

// ============================================================================
// FP4 GEMM Kernel for MI355X (gfx950) using MFMA 32x32x64 FP4
// ============================================================================
//
// Computes C[M,N] = A[M,K] (bf16) × B_shuffled[N,K/2] (fp4 packed) 
// with per-block scaling.
//
// Each block = 1 wave (64 threads) computes a 32×32 output tile.
// K dimension processed in chunks of 64 (64 FP4 values per MFMA call).
//
// B is pre-shuffled: B_sh[col][k_byte] where col = thread's column.
// B_scale_sh is pre-shuffled to match: one scale per 64-element K-block per column.
//
// A is bf16 input that we quantize on-the-fly to FP4.
// ============================================================================

// --------------------------------------------------------------------------
// BF16 → FP4 quantization helpers
// --------------------------------------------------------------------------

// FP4 E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
// We quantize bf16 to the nearest FP4 value and return the 4-bit code.

__device__ __forceinline__ uint8_t bf16_to_fp4(hip_bfloat16 val) {
    float f = __bfloat162float(val);
    uint8_t sign = (f < 0.0f) ? 1 : 0;
    float af = fabsf(f);
    
    // FP4 E2M1 magnitude encoding:
    // 0b000 = 0.0    0b001 = 0.5    0b010 = 1.0    0b011 = 1.5
    // 0b100 = 2.0    0b101 = 3.0    0b110 = 4.0    0b111 = 6.0
    uint8_t code;
    if      (af < 0.25f)  code = 0;  // 0.0
    else if (af < 0.75f)  code = 1;  // 0.5
    else if (af < 1.25f)  code = 2;  // 1.0
    else if (af < 1.75f)  code = 3;  // 1.5
    else if (af < 2.5f)   code = 4;  // 2.0
    else if (af < 3.5f)   code = 5;  // 3.0
    else if (af < 5.0f)   code = 6;  // 4.0
    else                   code = 7;  // 6.0
    
    return (sign << 3) | code;
}

// Compute block scale: max_abs / 6.0 (FP4 max magnitude)
__device__ __forceinline__ float compute_block_scale(
    const hip_bfloat16* row_ptr, int k_start, int k_end, int K) 
{
    float max_val = 0.0f;
    for (int i = k_start; i < k_end && i < K; i++) {
        float v = fabsf(__bfloat162float(row_ptr[i]));
        if (v > max_val) max_val = v;
    }
    if (max_val == 0.0f) return 0.0f;
    return max_val / 6.0f;  // FP4 E2M1 max representable = 6.0
}

// --------------------------------------------------------------------------
// Main GEMM kernel
// --------------------------------------------------------------------------
__global__ void gemm_fp4_kernel(
    const hip_bfloat16* __restrict__ A,       // [M, K] row-major bf16
    const uint8_t*      __restrict__ B_sh,    // [N, K/2] shuffled FP4 packed (2 per byte)
    const uint8_t*      __restrict__ B_scale_sh, // [N, K/64 * 4] shuffled scales (float32)
    hip_bfloat16*       __restrict__ C,       // [M, N] row-major bf16 output
    int M, int N, int K)
{
    const int lane = threadIdx.x;          // 0..63
    const int tile_n = blockIdx.x;         // N-tile index
    const int tile_m = blockIdx.y;         // M-tile index
    const int n_base = tile_n * 32;
    const int m_base = tile_m * 32;
    
    // Each thread in the wave handles:
    //   A: row = m_base + lane%32  (row for this thread's A operand)
    //   B: col = n_base + lane%32  (column for this thread's B operand)
    const int my_row = m_base + (lane % 32);
    const int my_col = n_base + (lane % 32);
    
    typedef int int8v __attribute__((ext_vector_type(8)));
    typedef float float16v __attribute__((ext_vector_type(16)));
    
    float16v acc;
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;
    
    // Number of K-blocks (each block = 64 FP4 values)
    const int k_blocks = (K + 63) / 64;
    
    // Pointer to B scales for this column: stored as float32 per k-block
    // B_scale_sh layout: [col][k_block] as float32 = 4 bytes each
    const float* b_scales = (const float*)(B_scale_sh + my_col * k_blocks * 4);
    
    // Pointer to B data for this column: [col][K/2] bytes (2 FP4 per byte)
    const uint8_t* b_col_ptr = B_sh + (size_t)my_col * (K / 2);
    
    for (int kb = 0; kb < k_blocks; kb++) {
        int k_start = kb * 64;
        
        // ---- Load and quantize A ----
        // Thread loads A[my_row, k_start : k_start+64] as bf16, quantizes to FP4
        int8v a_reg;
        uint32_t scale_a_bits;
        
        if (my_row < M) {
            const hip_bfloat16* a_row = A + (size_t)my_row * K;
            
            // Compute per-block scale for A
            float a_scale = compute_block_scale(a_row, k_start, k_start + 64, K);
            float a_inv_scale = (a_scale > 0.0f) ? (1.0f / a_scale) : 0.0f;
            
            // Pack scale as bf16 in upper 16 bits of uint32 for MFMA scale arg
            hip_bfloat16 a_scale_bf16 = __float2bfloat16(a_scale);
            uint16_t a_scale_u16;
            __builtin_memcpy(&a_scale_u16, &a_scale_bf16, 2);
            scale_a_bits = (uint32_t)a_scale_u16;
            
            // Quantize 64 bf16 values → 64 FP4 → pack into 8 int32 (8 FP4 per int32)
            uint32_t packed[8];
            for (int g = 0; g < 8; g++) {
                uint32_t word = 0;
                for (int p = 0; p < 8; p++) {
                    int idx = k_start + g * 8 + p;
                    hip_bfloat16 val = (idx < K) ? a_row[idx] : hip_bfloat16(0);
                    float scaled = __bfloat162float(val) * a_inv_scale;
                    uint8_t fp4 = bf16_to_fp4(__float2bfloat16(scaled));
                    word |= ((uint32_t)(fp4 & 0xF)) << (p * 4);
                }
                packed[g] = word;
            }
            __builtin_memcpy(&a_reg, packed, 32);
        } else {
            // Out-of-bounds row: zero
            uint32_t z[8] = {0,0,0,0,0,0,0,0};
            __builtin_memcpy(&a_reg, z, 32);
            scale_a_bits = 0;
        }
        
        // ---- Load B (pre-shuffled FP4) ----
        // B_sh[my_col][k_start/2 .. k_start/2+32] = 32 bytes = 64 FP4
        int8v b_reg;
        uint32_t scale_b_bits;
        
        if (my_col < N) {
            const uint8_t* b_ptr = b_col_ptr + k_start / 2;  // 2 FP4 per byte
            uint32_t b_packed[8];
            
            // Load 32 bytes into 8 int32
            // Each byte has 2 FP4 values (lo nibble, hi nibble)
            // We need to repack: 8 FP4 per int32 (4-bit each)
            for (int g = 0; g < 8; g++) {
                uint32_t word = 0;
                for (int p = 0; p < 4; p++) {
                    uint8_t byte_val = b_ptr[g * 4 + p];
                    // Each byte has 2 FP4: lo nibble = even index, hi nibble = odd index
                    uint8_t lo = byte_val & 0xF;
                    uint8_t hi = (byte_val >> 4) & 0xF;
                    word |= ((uint32_t)lo) << (p * 8);
                    word |= ((uint32_t)hi) << (p * 8 + 4);
                }
                b_packed[g] = word;
            }
            __builtin_memcpy(&b_reg, b_packed, 32);
            
            // Load B scale
            float b_scale = b_scales[kb];
            hip_bfloat16 b_scale_bf16 = __float2bfloat16(b_scale);
            uint16_t b_scale_u16;
            __builtin_memcpy(&b_scale_u16, &b_scale_bf16, 2);
            scale_b_bits = (uint32_t)b_scale_u16;
        } else {
            uint32_t z[8] = {0,0,0,0,0,0,0,0};
            __builtin_memcpy(&b_reg, z, 32);
            scale_b_bits = 0;
        }
        
        // ---- MFMA FP4 32×32×64 ----
        // 9 args: (a, b, c, cbsz=4, blgp=4, 0, scale_a, 0, scale_b)
        // cbsz=4 → FP4, blgp=4 → FP4
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 
            4,     // cbsz = 4 (FP4 for A)
            4,     // blgp = 4 (FP4 for B)  
            0,     // unused
            scale_a_bits,
            0,     // unused
            scale_b_bits
        );
    }
    
    // ---- Store C output ----
    // Output mapping (confirmed):
    //   For register v (0..15):
    //     row = (lane/32)*4 + (v%4) + (v/4)*8
    //     col = lane % 32
    // This gives 1024 unique (row,col) positions covering the 32×32 tile.
    
    for (int v = 0; v < 16; v++) {
        int row = m_base + (lane / 32) * 4 + (v % 4) + (v / 4) * 8;
        int col = n_base + (lane % 32);
        
        if (row < M && col < N) {
            C[(size_t)row * N + col] = __float2bfloat16(acc[v]);
        }
    }
}


// ============================================================================
// Host wrapper
// ============================================================================
torch::Tensor fn(
    torch::Tensor A,           // [M, K] bf16
    torch::Tensor B_shuffled,  // [N, K/2] uint8 (pre-shuffled FP4)
    torch::Tensor B_scale_sh,  // [N, num_k_blocks * 4] uint8 (pre-shuffled scales as float32 bytes)
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    dim3 block(64);  // 1 wave = 64 threads
    
    hipLaunchKernelGGL(
        gemm_fp4_kernel, grid, block, 0, 0,
        (const hip_bfloat16*)A.data_ptr(),
        (const uint8_t*)B_shuffled.data_ptr(),
        (const uint8_t*)B_scale_sh.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),
        (int)M, (int)N, (int)K
    );
    
    return C;
}
"""

print("GEMM HIP FP4 kernel source ready.")
print(f"Source length: {len(SRC)} chars")

# To compile (on MI355X):
# mod = load_inline(
#     name="gemm_fp4_v2",
#     cpp_sources='torch::Tensor fn(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t);',
#     cuda_sources=SRC,
#     functions=["fn"],
#     extra_cuda_cflags=["-O3", "--offload-arch=gfx950"]
# )

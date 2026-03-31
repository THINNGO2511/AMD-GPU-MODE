#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP GEMM v7 FIXED for MXFP4 on MI355X (gfx950).

KEY FIX: Uses aiter dynamic_mxfp4_quant for A (proven correct), then a
clean HIP kernel for the MFMA GEMM. This eliminates custom quantization
as an error source.

Uses 16x16x128 MFMA FP4 (__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4).
From CK WarpGemmAttributeMfmaImpl_f32_16x16x128_f8f6f4:
  kM=16, kN=16, kK=128
  kAMLane=16, kBNLane=16, kABKLane=4, kABKPerLane=32
  kCMLane=4, kCNLane=16, kCM0PerLane=1, kCM1PerLane=4

Thread decomposition (64 threads per wavefront):
  lr = lane % 16  -> spatial index (A row / B col within 16x16 tile)
  lg = lane / 16  -> K-group index (0..3)

Data per thread:
  A: 32 FP4 values (16 fp4x2 bytes) for row lr, K-group lg
      = A_fp4[row, (k_iter/2 + lg*16) .. (k_iter/2 + lg*16 + 15)]
  B: 32 FP4 values (16 fp4x2 bytes) for col lr, K-group lg
      = B_q[col, (k_iter/2 + lg*16) .. (k_iter/2 + lg*16 + 15)]
  Stored in LOWER 128 bits of 256-bit register (int32 slots [0..3]).
  UPPER 128 bits MUST be zero (slots [4..7]).

Scale per thread:
  uint32 with 4 E8M0 bytes: byte[g] = scale for K-group g (g=0..3)
  sa = A_scale[row, k_iter/32+0] | (A_scale[row, k_iter/32+1] << 8) | ...
  sb = B_scale[col, k_iter/32+0] | (B_scale[col, k_iter/32+1] << 8) | ...
  Hardware selects byte[lg] for this thread.

Output: 4 fp32 per thread
  C[lg*4 + i, lr] for i=0..3

Tile: 32x32 output via 4 warps (256 threads).
  warp_id/2 = wm (M-half: 0 or 1), warp_id%2 = wn (N-half: 0 or 1)
  Each warp handles one 16x16 sub-tile via one MFMA call.

Falls back to proven Triton paths for shapes where HIP isn't faster.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP kernel source -- 16x16x128 MFMA FP4 GEMM
# ---------------------------------------------------------------------------
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

// 256-bit MFMA operand = 8 x int32
typedef int __attribute__((ext_vector_type(8))) int8v;
// 4 fp32 output per thread for 16x16 MFMA
typedef float __attribute__((ext_vector_type(4))) float4v;

#define TM 32
#define TN 32
#define TK 128
#define NTHREADS 256
#define WARP_SZ 64

// bf16 -> f32
__device__ __forceinline__ float bf16_to_f32(unsigned short x) {
    union { unsigned int u; float f; } v;
    v.u = (unsigned int)x << 16;
    return v.f;
}

// f32 -> bf16 (RNE)
__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    union { unsigned int u; float f; } v;
    v.f = f;
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (unsigned short)(v.u >> 16);
}

// -----------------------------------------------------------------------
// Main kernel: C[M,N] = A_fp4[M,K/2] * B_q[N,K/2]^T
//   A_fp4: [M, K/2] uint8 (fp4x2, pre-quantized by aiter)
//   A_sc:  [M, K/32] uint8 (E8M0 scales, unshuffled)
//   B_q:   [N, K/2] uint8 (fp4x2)
//   B_sc:  [N, K/32] uint8 (E8M0 scales, unshuffled)
// -----------------------------------------------------------------------
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_v7(
    const unsigned char* __restrict__ A_fp4,
    const unsigned char* __restrict__ A_sc,
    const unsigned char* __restrict__ B_q,
    const unsigned char* __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;

    // Warp decomposition: 4 warps cover 2x2 grid of 16x16 sub-tiles
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int wm = warp_id >> 1;    // 0 or 1: which M-half of 32x32 tile
    const int wn = warp_id & 1;     // 0 or 1: which N-half of 32x32 tile

    // 16x16x128 MFMA thread decomposition within wavefront
    const int lr = lane & 15;       // 0..15: spatial index (row for A, col for B)
    const int lg = lane >> 4;       // 0..3: K-group index

    // Global tile base
    const int m_base = bm * TM;
    const int n_base = bn * TN;

    // Strides
    const int Khalf = K >> 1;       // K/2: bytes per row in fp4x2 tensors
    const int Kblocks = K >> 5;     // K/32: scale blocks per row

    // The A row and B column this thread is responsible for
    const int a_row = m_base + wm * 16 + lr;
    const int b_col = n_base + wn * 16 + lr;

    // Accumulator
    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    // K loop: 128 elements per MFMA iteration
    for (int k_iter = 0; k_iter < K; k_iter += TK) {

        // === Load A operand: 16 fp4x2 bytes for (a_row, K-group lg) ===
        int8v a_op = {};  // zero-init: upper 128 bits stay zero
        if (a_row < M) {
            // K byte offset for this K-group: lg * 16 bytes = lg * 32 FP4 values
            int a_byte_off = (k_iter >> 1) + lg * 16;
            if (a_byte_off + 16 <= Khalf) {
                const int* src = (const int*)(A_fp4 + (long long)a_row * Khalf + a_byte_off);
                a_op[0] = src[0];
                a_op[1] = src[1];
                a_op[2] = src[2];
                a_op[3] = src[3];
            }
        }

        // === Load B operand: 16 fp4x2 bytes for (b_col, K-group lg) ===
        int8v b_op = {};  // zero-init
        if (b_col < N) {
            int b_byte_off = (k_iter >> 1) + lg * 16;
            if (b_byte_off + 16 <= Khalf) {
                const int* src = (const int*)(B_q + (long long)b_col * Khalf + b_byte_off);
                b_op[0] = src[0];
                b_op[1] = src[1];
                b_op[2] = src[2];
                b_op[3] = src[3];
            }
        }

        // === Pack scales: 4 E8M0 bytes into uint32 ===
        // Byte[g] = scale for K-group g. Hardware selects byte[lg].
        unsigned int sa;
        if (a_row < M) {
            int kb = k_iter >> 5;  // k_iter / 32
            const unsigned char* sp = A_sc + (long long)a_row * Kblocks + kb;
            unsigned char s0 = (kb     < Kblocks) ? sp[0] : 127;
            unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
            unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
            unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
            sa = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                 ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
        } else {
            sa = 0x7F7F7F7Fu;  // neutral scale
        }

        unsigned int sb;
        if (b_col < N) {
            int kb = k_iter >> 5;
            const unsigned char* sp = B_sc + (long long)b_col * Kblocks + kb;
            unsigned char s0 = (kb     < Kblocks) ? sp[0] : 127;
            unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
            unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
            unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
            sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                 ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
        } else {
            sb = 0x7F7F7F7Fu;
        }

        // === MFMA 16x16x128 FP4 ===
        // cbsz=4 (FP4 A), blgp=4 (FP4 B), opsel=0 for both
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    // === Store output ===
    // Output mapping: col = lr, row = lg*4 + i (i=0..3)
    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}

torch::Tensor launch_hip_gemm_v7(
    torch::Tensor A_fp4,
    torch::Tensor A_sc,
    torch::Tensor B_q,
    torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A_fp4.device()));

    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);

    hipLaunchKernelGGL(hip_mxfp4_gemm_v7,
        grid, block, 0, 0,
        (const unsigned char*)A_fp4.data_ptr(),
        (const unsigned char*)A_sc.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K);

    return C;
}
"""

CPP_FORWARD = """
torch::Tensor launch_hip_gemm_v7(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
"""

# ---------------------------------------------------------------------------
# Module build
# ---------------------------------------------------------------------------
_module = None
_build_failed = False

def _get_module():
    global _module, _build_failed
    if _build_failed:
        return None
    if _module is not None:
        return _module
    try:
        from torch.utils.cpp_extension import load_inline
        _module = load_inline(
            name="hip_gemm_v7",
            cpp_sources=CPP_FORWARD,
            cuda_sources=HIP_SOURCE,
            functions=["launch_hip_gemm_v7"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
        return _module
    except Exception as e:
        print(f"[HIP BUILD FAILED] {e}")
        _build_failed = True
        return None

# ---------------------------------------------------------------------------
# Unshuffle E8M0 scales (undo e8m0_shuffle applied during B quantization)
# ---------------------------------------------------------------------------
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_b_cache_key = None
_b_q_u8 = None
_b_sc_raw = None
_y_cache = {}
_warmed = False
_hip_tested = False
_hip_works = False

_K7168_CFG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# ---------------------------------------------------------------------------
# Test HIP kernel accuracy on first call
# ---------------------------------------------------------------------------
def _test_hip_accuracy(mod, A, B_q_u8, B_sc_raw, A_fp4_u8, A_scale_u8, m, n, k):
    """Compare HIP kernel output against Triton reference. Return True if passes."""
    global _hip_tested, _hip_works
    _hip_tested = True

    try:
        # HIP kernel -- ensure all tensors have correct strides
        b_sc_slice = B_sc_raw[:n, :k // 32].contiguous()
        a_sc_slice = A_scale_u8[:m, :k // 32].contiguous()
        a_fp4_slice = A_fp4_u8[:m, :k // 2].contiguous()
        b_q_slice = B_q_u8[:n, :k // 2].contiguous()
        C_hip = mod.launch_hip_gemm_v7(a_fp4_slice, a_sc_slice, b_q_slice, b_sc_slice, m, n, k)
        torch.cuda.synchronize()

        # Triton reference
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        C_ref = gemm_afp4wfp4(A_fp4_u8, B_q_u8, A_scale_u8, B_sc_raw,
                               dtype=torch.bfloat16)
        torch.cuda.synchronize()

        # Compare
        diff = (C_hip.float() - C_ref.float()).abs()
        ref_abs = C_ref.float().abs().clamp(min=1e-8)
        rel = diff / ref_abs
        mean_abs = diff.mean().item()
        mean_rel = rel.mean().item()
        max_abs = diff.max().item()

        print(f"[HIP v7 TEST] M={m} N={n} K={k}: "
              f"mean_abs={mean_abs:.4f} max_abs={max_abs:.4f} mean_rel={mean_rel:.4%}")

        # Check tolerance: rtol=1e-2, atol=1e-2
        within = (diff <= 1e-2 + 1e-2 * ref_abs).float().mean().item()
        print(f"[HIP v7 TEST] Within tol: {within:.2%}")

        # Sample values
        h = C_hip.float().flatten()[:6]
        r = C_ref.float().flatten()[:6]
        print(f"[HIP v7 TEST] HIP[:6] = {[f'{x:.3f}' for x in h.tolist()]}")
        print(f"[HIP v7 TEST] REF[:6] = {[f'{x:.3f}' for x in r.tolist()]}")

        # Accept if >90% within tolerance
        _hip_works = within > 0.90
        if _hip_works:
            print("[HIP v7 TEST] PASSED - will use HIP for small M")
        else:
            print("[HIP v7 TEST] FAILED - falling back to Triton")

    except Exception as e:
        print(f"[HIP v7 TEST] ERROR: {e}")
        _hip_works = False

    return _hip_works

# ---------------------------------------------------------------------------
# Prewarm Triton JIT
# ---------------------------------------------------------------------------
def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except Exception:
            pass
    torch.cuda.synchronize()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_q_u8, _b_sc_raw

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B preprocessing
    bkey = B_scale_sh.data_ptr()
    if bkey != _b_cache_key:
        _b_cache_key = bkey
        _b_q_u8 = B_q.view(torch.uint8)
        _b_sc_raw = _unshuffle_e8m0(B_scale_sh)
        _prewarm()

    # Try HIP kernel for small M (where fused bf16->fp4 quant + MFMA could win)
    mod = _get_module()
    if mod is not None and m <= 64:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        # Quantize A using aiter (proven correct)
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        A_fp4_u8 = A_fp4.view(torch.uint8).contiguous()
        A_scale_u8 = A_scale.view(torch.uint8).contiguous()

        # Test accuracy on first call
        if not _hip_tested:
            _test_hip_accuracy(mod, A, _b_q_u8, _b_sc_raw,
                             A_fp4_u8, A_scale_u8, m, n, k)

        if _hip_works:
            b_sc_slice = _b_sc_raw[:n, :k // 32].contiguous()
            a_sc_slice = A_scale_u8[:m, :k // 32].contiguous()
            a_fp4_slice = A_fp4_u8[:m, :k // 2].contiguous()
            b_q_slice = _b_q_u8[:n, :k // 2].contiguous()
            return mod.launch_hip_gemm_v7(a_fp4_slice, a_sc_slice,
                                          b_q_slice, b_sc_slice, m, n, k)

    # Fallback: proven Triton paths
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _b_q_u8, A_scale, _b_sc_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _K7168_CFG if k == 7168 else None

    return gemm_a16wfp4(A, _b_q_u8, _b_sc_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)

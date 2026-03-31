#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
HIP GEMM v6 Scale Test — isolates whether ~13% error is from:
  A) Data layout (FP4 operand byte mapping) — would show error even with scale=127
  B) Scale packing (E8M0 bytes in uint32) — would show error ONLY with real scales

Two kernels compiled from SAME source with a preprocessor flag:
  - hip_mxfp4_gemm_normal: uses real computed/loaded scales (same as v4)
  - hip_mxfp4_gemm_force127: overrides sa=sb=0x7F7F7F7F (all scales=1.0)

If force127 error ~0% -> data layout correct, bug is in scale packing.
If force127 error still ~13% -> data layout itself is wrong.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"

import torch
from task import input_t, output_t

# ---------------------------------------------------------------------------
# HIP kernel source — TWO variants via FORCE_SCALE_127 preprocessor define
# ---------------------------------------------------------------------------
HIP_SOURCE = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <torch/extension.h>

typedef int __attribute__((ext_vector_type(8))) int8v;
typedef float __attribute__((ext_vector_type(4))) float4v;

#define TM 32
#define TN 32
#define TK 128
#define NTHREADS 256
#define WARP_SZ 64

#define LDS_A_FP4   (TM * (TK / 2))
#define LDS_A_SCALE (TM * (TK / 32))
#define LDS_B_FP4   (TN * (TK / 2))
#define LDS_TOTAL   (LDS_A_FP4 + LDS_A_SCALE + LDS_B_FP4)

#define OFF_A_FP4    0
#define OFF_A_SCALE  LDS_A_FP4
#define OFF_B_FP4   (LDS_A_FP4 + LDS_A_SCALE)

__device__ __forceinline__ float bf16_to_f32(unsigned short x) {
    union { unsigned int u; float f; } v;
    v.u = (unsigned int)x << 16;
    return v.f;
}

__device__ __forceinline__ unsigned short f32_to_bf16(float f) {
    union { unsigned int u; float f; } v;
    v.f = f;
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (unsigned short)(v.u >> 16);
}

__device__ void quant_bf16x32_to_fp4(
    const unsigned short* __restrict__ src,
    unsigned char* __restrict__ dst_fp4,
    unsigned char& scale_out,
    int valid)
{
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        if (i < valid) {
            vals[i] = bf16_to_f32(src[i]);
            amax = fmaxf(amax, fabsf(vals[i]));
        } else {
            vals[i] = 0.0f;
        }
    }

    union { float f; unsigned int u; } au;
    au.f = amax;
    unsigned int ai = au.u;
    ai = (ai + 0x200000u) & 0xFF800000u;
    int biased_exp = (int)((ai >> 23) & 0xFF);
    int su = biased_exp - 127 - 2;
    if (su < -127) su = -127;
    if (su > 127) su = 127;
    scale_out = (unsigned char)(su + 127);

    float inv_scale;
    {
        int qs = (-su) + 127;
        if (qs < 1) qs = 1;
        if (qs > 254) qs = 254;
        union { unsigned int u; float f; } sv;
        sv.u = (unsigned int)qs << 23;
        inv_scale = sv.f;
    }

    for (int i = 0; i < 16; i++) {
        unsigned char nib[2];
        for (int j = 0; j < 2; j++) {
            float qx = vals[2 * i + j] * inv_scale;
            union { float f; unsigned int u; } qu;
            qu.f = qx;
            unsigned int qx_u = qu.u;
            unsigned int sign = qx_u & 0x80000000u;
            qx_u ^= sign;
            qu.u = qx_u;
            float qx_abs = qu.f;

            unsigned char e2m1;
            if (qx_abs >= 6.0f) {
                e2m1 = 0x7;
            } else if (qx_abs < 1.0f) {
                unsigned int dm = ((127u - 1u) + (23u - 1u) + 1u) << 23;
                union { unsigned int u; float f; } dv;
                dv.u = dm;
                float dn = qx_abs + dv.f;
                union { float f; unsigned int u; } dr;
                dr.f = dn;
                e2m1 = (unsigned char)(dr.u - dm);
            } else {
                unsigned int mant_odd = (qx_u >> 22) & 1;
                unsigned int val_add = ((1u - 127u) << 23) + (1 << 21) - 1;
                unsigned int nx = qx_u + val_add + mant_odd;
                e2m1 = (unsigned char)(nx >> 22);
            }

            unsigned char sign_bit = (unsigned char)(sign >> 28);
            nib[j] = (e2m1 & 0x7) | sign_bit;
        }
        dst_fp4[i] = nib[0] | (nib[1] << 4);
    }
}

// ---- Templated kernel: force_scale_127 parameter ----
template <int FORCE_SCALE_127>
__global__ __attribute__((amdgpu_flat_work_group_size(256, 256)))
void hip_mxfp4_gemm_impl(
    const unsigned short* __restrict__ A,
    const unsigned char*  __restrict__ B_q,
    const unsigned char*  __restrict__ B_sc,
    unsigned short*       __restrict__ C,
    int M, int N, int K)
{
    const int bn = blockIdx.x;
    const int bm = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SZ;
    const int lane = tid % WARP_SZ;
    const int lr = lane & 15;
    const int lg = lane >> 4;

    const int wm = warp_id >> 1;
    const int wn = warp_id & 1;

    const int m_base = bm * TM;
    const int n_base = bn * TN;
    const int Khalf = K / 2;
    const int Kblocks = K / 32;

    extern __shared__ unsigned char smem[];
    unsigned char* lds_a_fp4   = smem + OFF_A_FP4;
    unsigned char* lds_a_scale = smem + OFF_A_SCALE;
    unsigned char* lds_b_fp4   = smem + OFF_B_FP4;

    float4v acc = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int k_iter = 0; k_iter < K; k_iter += TK) {
        __syncthreads();

        // Phase 1: Load + quantize A
        if (tid < 128) {
            int a_row_local = tid >> 2;
            int a_kblock    = tid & 3;
            int a_row_global = m_base + a_row_local;
            int k_global = k_iter + a_kblock * 32;

            unsigned char* fp4_dst = lds_a_fp4 + a_row_local * (TK / 2) + a_kblock * 16;
            unsigned char sc = 127;

            if (a_row_global < M && k_global < K) {
                int valid = K - k_global;
                if (valid > 32) valid = 32;
                const unsigned short* a_ptr = A + (long long)a_row_global * K + k_global;
                quant_bf16x32_to_fp4(a_ptr, fp4_dst, sc, valid);
            } else {
                for (int x = 0; x < 16; x++) fp4_dst[x] = 0;
            }

            lds_a_scale[a_row_local * 4 + a_kblock] = sc;
        }

        // Phase 2: Load B tile
        {
            int byte_off = tid * 8;
            int b_row_local = byte_off / (TK / 2);
            int b_col = byte_off % (TK / 2);
            int b_row_global = n_base + b_row_local;

            unsigned char* dst = lds_b_fp4 + b_row_local * (TK / 2) + b_col;
            if (b_row_global < N && b_row_local < TN) {
                const unsigned char* gsrc = B_q + (long long)b_row_global * Khalf + (k_iter / 2) + b_col;
                *(unsigned long long*)dst = *(const unsigned long long*)gsrc;
            } else if (b_row_local < TN) {
                *(unsigned long long*)dst = 0ULL;
            }
        }

        __syncthreads();

        // Phase 3: Load MFMA operands from LDS
        int8v a_op = {};
        {
            int a_row = wm * 16 + lr;
            int a_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_a_fp4 + a_row * (TK / 2) + a_byte_off);
            a_op[0] = src_ptr[0];
            a_op[1] = src_ptr[1];
            a_op[2] = src_ptr[2];
            a_op[3] = src_ptr[3];
        }

        int8v b_op = {};
        {
            int b_row = wn * 16 + lr;
            int b_byte_off = lg * 16;
            const int* src_ptr = (const int*)(lds_b_fp4 + b_row * (TK / 2) + b_byte_off);
            b_op[0] = src_ptr[0];
            b_op[1] = src_ptr[1];
            b_op[2] = src_ptr[2];
            b_op[3] = src_ptr[3];
        }

        // Phase 4: Pack scales
        unsigned int sa;
        unsigned int sb;

        if (FORCE_SCALE_127) {
            // OVERRIDE: all scales = 127 = E8M0 exponent 0 = multiply by 1.0
            sa = 0x7F7F7F7Fu;
            sb = 0x7F7F7F7Fu;
        } else {
            // Normal: load real scales (identical to v4)
            {
                int a_row = wm * 16 + lr;
                const unsigned char* sp = lds_a_scale + a_row * 4;
                sa = (unsigned int)sp[0] | ((unsigned int)sp[1] << 8) |
                     ((unsigned int)sp[2] << 16) | ((unsigned int)sp[3] << 24);
            }
            {
                int b_row_global = n_base + wn * 16 + lr;
                int kb = k_iter / 32;
                if (b_row_global < N) {
                    const unsigned char* sp = B_sc + (long long)b_row_global * Kblocks + kb;
                    unsigned char s0 = sp[0];
                    unsigned char s1 = (kb + 1 < Kblocks) ? sp[1] : 127;
                    unsigned char s2 = (kb + 2 < Kblocks) ? sp[2] : 127;
                    unsigned char s3 = (kb + 3 < Kblocks) ? sp[3] : 127;
                    sb = (unsigned int)s0 | ((unsigned int)s1 << 8) |
                         ((unsigned int)s2 << 16) | ((unsigned int)s3 << 24);
                } else {
                    sb = 0x7F7F7F7Fu;
                }
            }
        }

        // Phase 5: MFMA
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            a_op, b_op, acc,
            4, 4, 0,
            sa, 0, sb);
    }

    // Store output
    for (int i = 0; i < 4; i++) {
        int r = m_base + wm * 16 + lg * 4 + i;
        int c = n_base + wn * 16 + lr;
        if (r < M && c < N) {
            C[(long long)r * N + c] = f32_to_bf16(acc[i]);
        }
    }
}

// ---- Two explicit instantiations ----
torch::Tensor launch_hip_gemm_normal(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(
        (hip_mxfp4_gemm_impl<0>),
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K);
    return C;
}

torch::Tensor launch_hip_gemm_force127(
    torch::Tensor A, torch::Tensor B_q, torch::Tensor B_sc,
    int64_t M, int64_t N, int64_t K)
{
    auto C = torch::empty({M, N},
        torch::TensorOptions().dtype(torch::kBFloat16).device(A.device()));
    dim3 grid((int)((N + TN - 1) / TN), (int)((M + TM - 1) / TM));
    dim3 block(NTHREADS);
    hipLaunchKernelGGL(
        (hip_mxfp4_gemm_impl<1>),
        grid, block, LDS_TOTAL, 0,
        (const unsigned short*)A.data_ptr(),
        (const unsigned char*)B_q.data_ptr(),
        (const unsigned char*)B_sc.data_ptr(),
        (unsigned short*)C.data_ptr(),
        (int)M, (int)N, (int)K);
    return C;
}
"""

CPP_FORWARD = """
torch::Tensor launch_hip_gemm_normal(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
torch::Tensor launch_hip_gemm_force127(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
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
            name="hip_gemm_v6_scaletest",
            cpp_sources=CPP_FORWARD,
            cuda_sources=HIP_SOURCE,
            functions=["launch_hip_gemm_normal", "launch_hip_gemm_force127"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
            verbose=False,
        )
        return _module
    except Exception as e:
        print(f"[HIP BUILD FAILED] {e}")
        _build_failed = True
        return None

# ---------------------------------------------------------------------------
# Unshuffle E8M0
# ---------------------------------------------------------------------------
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

# ---------------------------------------------------------------------------
# Compute reference using aiter (proven correct path)
# ---------------------------------------------------------------------------
def _compute_reference(A, B_q_u8, B_sc_raw, m, n, k):
    """Compute reference output using aiter gemm_a16wfp4 (known correct)."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, B_q_u8, B_sc_raw, dtype=torch.bfloat16)

# ---------------------------------------------------------------------------
# Error analysis helper
# ---------------------------------------------------------------------------
def _analyze_error(name, output, reference):
    """Print detailed error statistics between output and reference."""
    out_f = output.float()
    ref_f = reference.float()
    diff = (out_f - ref_f).abs()
    ref_abs = ref_f.abs().clamp(min=1e-8)
    rel_err = diff / ref_abs

    print(f"\n=== {name} ===")
    print(f"  Shape: {output.shape}")
    print(f"  Output range: [{out_f.min().item():.4f}, {out_f.max().item():.4f}]")
    print(f"  Reference range: [{ref_f.min().item():.4f}, {ref_f.max().item():.4f}]")
    print(f"  Abs error: mean={diff.mean().item():.6f}, max={diff.max().item():.6f}")
    print(f"  Rel error: mean={rel_err.mean().item():.4%}, max={rel_err.max().item():.4%}")
    print(f"  RMSE: {(diff**2).mean().sqrt().item():.6f}")

    # Check tolerance: rtol=1e-2, atol=1e-2
    within_tol = (diff <= 1e-2 + 1e-2 * ref_abs).float().mean().item()
    print(f"  Within rtol=1e-2, atol=1e-2: {within_tol:.2%}")

    # Sample first 8 elements
    flat_out = out_f.flatten()[:8]
    flat_ref = ref_f.flatten()[:8]
    flat_diff = diff.flatten()[:8]
    print(f"  First 8 output:    {[f'{x:.4f}' for x in flat_out.tolist()]}")
    print(f"  First 8 reference: {[f'{x:.4f}' for x in flat_ref.tolist()]}")
    print(f"  First 8 abs_diff:  {[f'{x:.4f}' for x in flat_diff.tolist()]}")

    # Also check ratio (output/reference) for systematic bias
    ratio = out_f / ref_f.clamp(min=1e-8)
    valid_mask = ref_f.abs() > 0.01
    if valid_mask.any():
        valid_ratios = ratio[valid_mask]
        print(f"  Ratio (out/ref) where |ref|>0.01: mean={valid_ratios.mean().item():.6f}, std={valid_ratios.std().item():.6f}")

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_b_cache_key = None
_b_q_u8 = None
_b_sc_raw = None
_test_done = False

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def custom_kernel(data: input_t) -> output_t:
    global _b_cache_key, _b_q_u8, _b_sc_raw, _test_done

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Cache B preprocessing
    bkey = B_scale_sh.data_ptr()
    if bkey != _b_cache_key:
        _b_cache_key = bkey
        _b_q_u8 = B_q.view(torch.uint8)
        _b_sc_raw = _unshuffle_e8m0(B_scale_sh)

    mod = _get_module()
    if mod is None:
        print("[ERROR] HIP module failed to build, returning zeros")
        return torch.zeros(m, n, dtype=torch.bfloat16, device=A.device())

    # Only run diagnostic on first call (avoid spamming)
    if not _test_done:
        _test_done = True
        print(f"\n{'='*70}")
        print(f"HIP GEMM v6 SCALE TEST — M={m}, N={n}, K={k}")
        print(f"{'='*70}")

        b_sc_slice = _b_sc_raw[:n, :k // 32].contiguous()

        # 1. Run NORMAL kernel (same as v4)
        C_normal = mod.launch_hip_gemm_normal(A, _b_q_u8, b_sc_slice, m, n, k)
        torch.cuda.synchronize()

        # 2. Run FORCE127 kernel (all scales = 1.0)
        C_force127 = mod.launch_hip_gemm_force127(A, _b_q_u8, b_sc_slice, m, n, k)
        torch.cuda.synchronize()

        # 3. Compute reference (aiter, known correct)
        C_ref = _compute_reference(A, _b_q_u8, _b_sc_raw, m, n, k)
        torch.cuda.synchronize()

        # 4. Also compute a reference with scales=127 for fair comparison
        # For force127 comparison, we need: ref_force127 = A_fp4 @ B_fp4^T
        # where both use scale=1.0. We can approximate by computing:
        # quantize A with normal scales, then the MFMA with scale=127 means
        # the FP4 values are NOT rescaled. So output = sum(fp4_a * fp4_b)
        # without any scale correction.
        #
        # The "correct" output for force127 would be the same MFMA with
        # scale=127, but we don't have a reference for that. Instead:
        # - Compare NORMAL vs reference -> should show ~13% error (v4 baseline)
        # - Compare FORCE127 vs reference -> will show DIFFERENT error pattern
        # - Compare FORCE127 vs NORMAL -> shows how much scale changes things
        #
        # KEY INSIGHT: if NORMAL and FORCE127 give very similar outputs,
        # then the scales aren't being applied at all (possible HW behavior).

        print(f"\n--- Comparison 1: NORMAL (real scales) vs REFERENCE ---")
        _analyze_error("NORMAL vs REFERENCE", C_normal, C_ref)

        print(f"\n--- Comparison 2: FORCE127 (all scale=1) vs REFERENCE ---")
        _analyze_error("FORCE127 vs REFERENCE", C_force127, C_ref)

        print(f"\n--- Comparison 3: NORMAL vs FORCE127 ---")
        _analyze_error("NORMAL vs FORCE127", C_normal, C_force127)

        # Print scale statistics for context
        print(f"\n--- Scale Statistics ---")
        b_scales = b_sc_slice.view(-1).float()
        print(f"  B scale range: [{b_scales.min().item():.0f}, {b_scales.max().item():.0f}]")
        print(f"  B scale mean: {b_scales.mean().item():.1f}")
        print(f"  B scale std: {b_scales.std().item():.1f}")
        print(f"  B scales == 127: {(b_scales == 127).float().mean().item():.2%}")

        # A scales are computed dynamically, sample a few
        # We can read them from LDS indirectly by looking at the quant output
        print(f"\n  A bf16 amax (first row): {A[0].float().abs().max().item():.4f}")
        print(f"  A bf16 range: [{A.float().min().item():.4f}, {A.float().max().item():.4f}]")

        print(f"\n{'='*70}")
        print("INTERPRETATION:")
        print("  If NORMAL error >> FORCE127 error -> Scale packing is WRONG")
        print("  If NORMAL error ~ FORCE127 error -> Data layout is WRONG")
        print("  If NORMAL ~ FORCE127 outputs    -> Scales not being applied at all")
        print(f"{'='*70}\n")

    # Return the normal kernel output for actual evaluation
    b_sc_slice = _b_sc_raw[:n, :k // 32].contiguous()
    return mod.launch_hip_gemm_normal(A, _b_q_u8, b_sc_slice, m, n, k)

#!/usr/bin/env python3
"""GEMM Scale Debug: Isolate whether ~10% error is from data loading or scale routing.

Test 1: All scales=127 (trivial 1.0) - verifies data loading + MFMA math
Test 2: Two MFMA calls per K-step - manual per-group scaling
Test 3: Different cbsz_sel/op_sel values
Test 4: B_shuffle (pre-shuffled) variants
Test 5: B_shuffle + two-MFMA

If Test 1 passes: problem is ONLY scale routing. Use two-MFMA approach.
If Test 1 fails: data loading or register mapping is wrong.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch, sys

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v16f __attribute__((ext_vector_type(16)));

extern "C" __device__ v16f __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
    v8i, v8i, v16f, int, int, int, int, int, int) __asm("llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4");

__device__ __forceinline__ float bf2f(hip_bfloat16 v) {
    union { unsigned u; float f; } x;
    x.u = ((unsigned)(*reinterpret_cast<unsigned short*>(&v))) << 16;
    return x.f;
}
__device__ __forceinline__ hip_bfloat16 f2bf(float v) {
    union { float f; unsigned u; } x; x.f = v;
    unsigned short s = (unsigned short)(x.u >> 16);
    return *reinterpret_cast<hip_bfloat16*>(&s);
}
__device__ __forceinline__ unsigned char to_fp4(float val, float inv_s) {
    float sc = val * inv_s;
    float a = fabsf(sc);
    unsigned char sgn = (sc < 0.0f) ? 0x8 : 0x0;
    unsigned char e;
    if      (a >= 4.0f)   e = 0x7;
    else if (a >= 2.5f)   e = 0x6;
    else if (a >= 1.75f)  e = 0x5;
    else if (a >= 1.25f)  e = 0x4;
    else if (a >= 0.875f) e = 0x3;
    else if (a >= 0.625f) e = 0x2;
    else if (a >= 0.25f)  e = 0x1;
    else                  e = 0x0;
    return sgn | e;
}

__device__ void quant_group_32(const hip_bfloat16* src, int valid, unsigned int* dst, unsigned char* out_scale) {
    float vals[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; i++) {
        vals[i] = (i < valid) ? bf2f(src[i]) : 0.0f;
        amax = fmaxf(amax, fabsf(vals[i]));
    }
    unsigned char e8m0 = 0;
    float inv_s = 0.0f;
    if (amax > 0.0f) {
        union { float f; unsigned u; } x; x.f = amax;
        e8m0 = (unsigned char)((x.u >> 23) & 0xFF);
        x.u = ((unsigned int)e8m0) << 23;
        inv_s = 1.0f / x.f;
    }
    *out_scale = e8m0;
    for (int w = 0; w < 4; w++) {
        unsigned int pk = 0;
        for (int b = 0; b < 8; b++) {
            unsigned char fp4 = to_fp4(vals[w*8+b], inv_s);
            pk |= ((unsigned int)(fp4 & 0xF)) << (b * 4);
        }
        dst[w] = pk;
    }
}

__device__ void load_b_32bytes(const unsigned char* src, unsigned int* dst) {
    for (int w = 0; w < 8; w++) {
        unsigned int pk = 0;
        for (int n = 0; n < 8; n++) {
            int nidx = w * 8 + n;
            int bidx = nidx / 2;
            int is_hi = nidx % 2;
            unsigned char bv = src[bidx];
            unsigned char nib = is_hi ? ((bv >> 4) & 0xF) : (bv & 0xF);
            pk |= ((unsigned int)nib) << (n * 4);
        }
        dst[w] = pk;
    }
}

__device__ void load_b_16bytes(const unsigned char* src, unsigned int* dst) {
    for (int w = 0; w < 4; w++) {
        unsigned int pk = 0;
        for (int n = 0; n < 8; n++) {
            int nidx = w * 8 + n;
            int bidx = nidx / 2;
            int is_hi = nidx % 2;
            unsigned char bv = src[bidx];
            unsigned char nib = is_hi ? ((bv >> 4) & 0xF) : (bv & 0xF);
            pk |= ((unsigned int)nib) << (n * 4);
        }
        dst[w] = pk;
    }
    dst[4] = dst[5] = dst[6] = dst[7] = 0;
}

// MODE 0: Trivial scales (all 127 = 1.0)
__global__ void gemm_trivial(
    const hip_bfloat16* A, const unsigned char* B_q, hip_bfloat16* C,
    int M, int N, int K)
{
    int lane = threadIdx.x;
    int m_base = blockIdx.y * 32;
    int n_base = blockIdx.x * 32;
    int my_row = m_base + (lane % 32);
    int my_col = n_base + (lane % 32);
    int half = lane / 32;
    v16f acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ks = 0; ks < K; ks += 64) {
        v8i a_reg, b_reg;
        unsigned int* ap = (unsigned int*)&a_reg;
        unsigned int* bp = (unsigned int*)&b_reg;
        if (my_row < M) {
            const hip_bfloat16* abase = A + my_row * K + ks;
            for (int g = 0; g < 2; g++)
                for (int w = 0; w < 4; w++) {
                    unsigned int pk = 0;
                    for (int b = 0; b < 8; b++) {
                        int idx = g*32 + w*8 + b;
                        float val = (ks + idx < K) ? bf2f(abase[idx]) : 0.0f;
                        pk |= ((unsigned int)(to_fp4(val, 1.0f) & 0xF)) << (b * 4);
                    }
                    ap[g*4+w] = pk;
                }
        } else { for (int i = 0; i < 8; i++) ap[i] = 0; }
        if (my_col < N)
            load_b_32bytes(B_q + (size_t)my_col * (K/2) + ks/2, bp);
        else { for (int i = 0; i < 8; i++) bp[i] = 0; }
        unsigned int sa = 127u | (127u << 8), sb = 127u | (127u << 8);
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }
    for (int v = 0; v < 16; v++) {
        int row = m_base + half*4 + (v%4) + (v/4)*8;
        int col = n_base + (lane%32);
        if (row < M && col < N) C[row*N + col] = f2bf(acc[v]);
    }
}

// MODE 1: Two MFMA calls per K-step (32 deep each)
__global__ void gemm_two_mfma(
    const hip_bfloat16* A, const unsigned char* B_q, const unsigned char* B_scale,
    hip_bfloat16* C, int M, int N, int K)
{
    int lane = threadIdx.x;
    int m_base = blockIdx.y * 32; int n_base = blockIdx.x * 32;
    int my_row = m_base + (lane%32); int my_col = n_base + (lane%32);
    int half = lane / 32; int sg = K / 32;
    v16f acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ks = 0; ks < K; ks += 64) {
        for (int sub = 0; sub < 2; sub++) {
            int ko = ks + sub * 32;
            v8i a_reg, b_reg;
            unsigned int* ap = (unsigned int*)&a_reg;
            unsigned int* bp = (unsigned int*)&b_reg;
            unsigned char asc = 0, bsc = 127;
            for (int i = 0; i < 8; i++) { ap[i] = 0; bp[i] = 0; }
            if (my_row < M)
                quant_group_32(A + my_row*K + ko, min(32, K-ko), ap, &asc);
            if (my_col < N) {
                load_b_16bytes(B_q + (size_t)my_col*(K/2) + ko/2, bp);
                bsc = B_scale[my_col * sg + ko/32];
            }
            unsigned int sa = (unsigned int)asc, sb = (unsigned int)bsc;
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
        }
    }
    for (int v = 0; v < 16; v++) {
        int row = m_base + half*4 + (v%4) + (v/4)*8;
        int col = n_base + (lane%32);
        if (row < M && col < N) C[row*N + col] = f2bf(acc[v]);
    }
}

// MODE 2: Full K=64 with packed scales, variable cbsz_sel/op_sel
__global__ void gemm_full64(
    const hip_bfloat16* A, const unsigned char* B_q, const unsigned char* B_scale,
    hip_bfloat16* C, int M, int N, int K, int variant)
{
    int lane = threadIdx.x;
    int m_base = blockIdx.y * 32; int n_base = blockIdx.x * 32;
    int my_row = m_base + (lane%32); int my_col = n_base + (lane%32);
    int half = lane / 32; int sg = K / 32;
    v16f acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ks = 0; ks < K; ks += 64) {
        v8i a_reg, b_reg;
        unsigned int* ap = (unsigned int*)&a_reg;
        unsigned int* bp = (unsigned int*)&b_reg;
        unsigned char as0=127, as1=127, bs0=127, bs1=127;
        if (my_row < M) {
            quant_group_32(A + my_row*K + ks,     min(32, K-ks),    ap,   &as0);
            quant_group_32(A + my_row*K + ks + 32, min(32, K-ks-32), ap+4, &as1);
        } else { for (int i = 0; i < 8; i++) ap[i] = 0; }
        if (my_col < N) {
            load_b_32bytes(B_q + (size_t)my_col*(K/2) + ks/2, bp);
            bs0 = B_scale[my_col*sg + ks/32];
            bs1 = B_scale[my_col*sg + ks/32 + 1];
        } else { for (int i = 0; i < 8; i++) bp[i] = 0; }
        unsigned int sa = (unsigned int)as0 | ((unsigned int)as1 << 8);
        unsigned int sb = (unsigned int)bs0 | ((unsigned int)bs1 << 8);
        if (variant == 0)
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
        else if (variant == 1)
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 1, sa, 1, sb);
        else if (variant == 2)
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 1, sb);
        else
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 1, sa, 0, sb);
    }
    for (int v = 0; v < 16; v++) {
        int row = m_base + half*4 + (v%4) + (v/4)*8;
        int col = n_base + (lane%32);
        if (row < M && col < N) C[row*N + col] = f2bf(acc[v]);
    }
}

// MODE 3: B_shuffle direct load
__global__ void gemm_bshuffle(
    const hip_bfloat16* A, const unsigned char* B_sh, const unsigned char* B_sc,
    hip_bfloat16* C, int M, int N, int K, int use_trivial)
{
    int lane = threadIdx.x;
    int m_base = blockIdx.y * 32; int n_base = blockIdx.x * 32;
    int my_row = m_base + (lane%32); int my_col = n_base + (lane%32);
    int half = lane / 32; int sg = K / 32;
    v16f acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ks = 0; ks < K; ks += 64) {
        v8i a_reg, b_reg;
        unsigned int* ap = (unsigned int*)&a_reg;
        unsigned int* bp = (unsigned int*)&b_reg;
        unsigned char as0=127, as1=127, bs0=127, bs1=127;
        if (my_row < M) {
            if (use_trivial) {
                for (int g = 0; g < 2; g++)
                    for (int w = 0; w < 4; w++) {
                        unsigned int pk = 0;
                        for (int b = 0; b < 8; b++) {
                            int idx = g*32 + w*8 + b;
                            float val = (ks+idx < K) ? bf2f(A[my_row*K+ks+idx]) : 0.0f;
                            pk |= ((unsigned int)(to_fp4(val, 1.0f) & 0xF)) << (b*4);
                        }
                        ap[g*4+w] = pk;
                    }
            } else {
                quant_group_32(A+my_row*K+ks,     min(32,K-ks),    ap,   &as0);
                quant_group_32(A+my_row*K+ks+32,   min(32,K-ks-32), ap+4, &as1);
            }
        } else { for (int i = 0; i < 8; i++) ap[i] = 0; }
        if (my_col < N) {
            const unsigned int* bsrc = (const unsigned int*)(B_sh + (size_t)my_col*(K/2) + ks/2);
            for (int i = 0; i < 8; i++) bp[i] = bsrc[i];
            if (!use_trivial) {
                bs0 = B_sc[my_col*sg + ks/32];
                bs1 = B_sc[my_col*sg + ks/32 + 1];
            }
        } else { for (int i = 0; i < 8; i++) bp[i] = 0; }
        unsigned int sa = use_trivial ? (127u|(127u<<8)) : ((unsigned int)as0|((unsigned int)as1<<8));
        unsigned int sb = use_trivial ? (127u|(127u<<8)) : ((unsigned int)bs0|((unsigned int)bs1<<8));
        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }
    for (int v = 0; v < 16; v++) {
        int row = m_base + half*4 + (v%4) + (v/4)*8;
        int col = n_base + (lane%32);
        if (row < M && col < N) C[row*N + col] = f2bf(acc[v]);
    }
}

// MODE 4: Two-MFMA with B_shuffle
__global__ void gemm_bsh_two(
    const hip_bfloat16* A, const unsigned char* B_sh, const unsigned char* B_sc,
    hip_bfloat16* C, int M, int N, int K)
{
    int lane = threadIdx.x;
    int m_base = blockIdx.y * 32; int n_base = blockIdx.x * 32;
    int my_row = m_base + (lane%32); int my_col = n_base + (lane%32);
    int half = lane / 32; int sg = K / 32;
    v16f acc = {}; for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int ks = 0; ks < K; ks += 64) {
        for (int sub = 0; sub < 2; sub++) {
            int ko = ks + sub*32;
            v8i a_reg, b_reg;
            unsigned int* ap = (unsigned int*)&a_reg;
            unsigned int* bp = (unsigned int*)&b_reg;
            unsigned char asc = 0, bsc = 127;
            for (int i = 0; i < 8; i++) { ap[i] = 0; bp[i] = 0; }
            if (my_row < M)
                quant_group_32(A + my_row*K + ko, min(32, K-ko), ap, &asc);
            if (my_col < N) {
                const unsigned int* bsrc = (const unsigned int*)(B_sh + (size_t)my_col*(K/2) + ko/2);
                for (int i = 0; i < 4; i++) bp[i] = bsrc[i];
                bsc = B_sc[my_col*sg + ko/32];
            }
            unsigned int sa = (unsigned int)asc, sb = (unsigned int)bsc;
            acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
        }
    }
    for (int v = 0; v < 16; v++) {
        int row = m_base + half*4 + (v%4) + (v/4)*8;
        int col = n_base + (lane%32);
        if (row < M && col < N) C[row*N + col] = f2bf(acc[v]);
    }
}

// Launch wrappers
torch::Tensor launch_trivial(torch::Tensor A, torch::Tensor Bq, int64_t M, int64_t N, int64_t K) {
    auto C = torch::zeros({M,N}, A.options());
    dim3 g((N+31)/32, (M+31)/32); dim3 b(64);
    hipLaunchKernelGGL(gemm_trivial, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
        (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
torch::Tensor launch_two_mfma(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs,
                               int64_t M, int64_t N, int64_t K) {
    auto C = torch::zeros({M,N}, A.options());
    dim3 g((N+31)/32, (M+31)/32); dim3 b(64);
    hipLaunchKernelGGL(gemm_two_mfma, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
        (const unsigned char*)Bs.data_ptr(), (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
torch::Tensor launch_full64(torch::Tensor A, torch::Tensor Bq, torch::Tensor Bs,
                             int64_t M, int64_t N, int64_t K, int64_t variant) {
    auto C = torch::zeros({M,N}, A.options());
    dim3 g((N+31)/32, (M+31)/32); dim3 b(64);
    hipLaunchKernelGGL(gemm_full64, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const unsigned char*)Bq.data_ptr(),
        (const unsigned char*)Bs.data_ptr(), (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K, (int)variant);
    return C;
}
torch::Tensor launch_bshuffle(torch::Tensor A, torch::Tensor Bsh, torch::Tensor Bsc,
                               int64_t M, int64_t N, int64_t K, int64_t trivial) {
    auto C = torch::zeros({M,N}, A.options());
    dim3 g((N+31)/32, (M+31)/32); dim3 b(64);
    hipLaunchKernelGGL(gemm_bshuffle, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const unsigned char*)Bsh.data_ptr(),
        (const unsigned char*)Bsc.data_ptr(), (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K, (int)trivial);
    return C;
}
torch::Tensor launch_bsh_two(torch::Tensor A, torch::Tensor Bsh, torch::Tensor Bsc,
                              int64_t M, int64_t N, int64_t K) {
    auto C = torch::zeros({M,N}, A.options());
    dim3 g((N+31)/32, (M+31)/32); dim3 b(64);
    hipLaunchKernelGGL(gemm_bsh_two, g, b, 0, 0,
        (const hip_bfloat16*)A.data_ptr(), (const unsigned char*)Bsh.data_ptr(),
        (const unsigned char*)Bsc.data_ptr(), (hip_bfloat16*)C.data_ptr(), (int)M, (int)N, (int)K);
    return C;
}
"""

FWD_DECL = """
torch::Tensor launch_trivial(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
torch::Tensor launch_two_mfma(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
torch::Tensor launch_full64(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);
torch::Tensor launch_bshuffle(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);
torch::Tensor launch_bsh_two(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
"""

from torch.utils.cpp_extension import load_inline

print("Compiling 5 HIP kernel modes...")
try:
    mod = load_inline(
        name="gemm_scale_debug",
        cpp_sources=FWD_DECL,
        cuda_sources=HIP_SRC,
        functions=["launch_trivial", "launch_two_mfma", "launch_full64",
                   "launch_bshuffle", "launch_bsh_two"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950"],
        verbose=False,
    )
    print("Compilation SUCCESS")
except Exception as e:
    print(f"Compilation FAILED: {e}")
    sys.exit(1)


# ============================================================================
# Helpers
# ============================================================================
def unshuffle_scale(B_scale_sh, N, K):
    n_groups = K // 32
    n_pad = (N + 31) // 32 * 32
    s = B_scale_sh.view(n_pad, n_groups)
    if n_groups >= 4:
        s = s.view(n_pad, n_groups // 4, 4)
        s_new = torch.empty_like(s)
        s_new[:, :, 0] = s[:, :, 0]
        s_new[:, :, 1] = s[:, :, 2]
        s_new[:, :, 2] = s[:, :, 1]
        s_new[:, :, 3] = s[:, :, 3]
        s = s_new.reshape(n_pad, n_groups)
    return s[:N]

def check(C_test, C_ref, label, rtol=1e-2, atol=1e-2):
    diff = (C_test.float() - C_ref.float()).abs()
    close = torch.isclose(C_test.float(), C_ref.float(), rtol=rtol, atol=atol)
    n_total = close.numel()
    n_match = close.sum().item()
    n_mis = n_total - n_match
    mx = diff.max().item()
    mn = diff.mean().item()
    rel = (diff / (C_ref.float().abs() + 1e-8)).mean().item()
    status = "PASS" if n_mis == 0 else "FAIL"
    print(f"  [{status}] {label}: {n_match}/{n_total} match "
          f"({n_mis} mis={100*n_mis/max(n_total,1):.1f}%), "
          f"max={mx:.4f}, mean={mn:.6f}, rel={rel:.4f}")
    if n_mis > 0 and n_total <= 20000:
        idx = torch.where(~close)
        for i in range(min(5, len(idx[0]))):
            r, c = idx[0][i].item(), idx[1][i].item()
            print(f"    [{r},{c}] got={C_test[r,c].item():.4f} ref={C_ref[r,c].item():.4f}")
    return n_mis == 0

FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

def quant_fp4_s1(vals, device):
    flat = vals.float()
    abs_v = flat.abs()
    enc = torch.zeros_like(abs_v, dtype=torch.uint8)
    enc[abs_v >= 4.0] = 7
    enc[(abs_v >= 2.5) & (abs_v < 4.0)] = 6
    enc[(abs_v >= 1.75) & (abs_v < 2.5)] = 5
    enc[(abs_v >= 1.25) & (abs_v < 1.75)] = 4
    enc[(abs_v >= 0.875) & (abs_v < 1.25)] = 3
    enc[(abs_v >= 0.625) & (abs_v < 0.875)] = 2
    enc[(abs_v >= 0.25) & (abs_v < 0.625)] = 1
    sign = (flat < 0).to(torch.uint8) << 3
    fp4 = sign | enc
    lut = torch.tensor(FP4_LUT, device=device)
    mag = lut[(fp4 & 0x7).long()]
    neg = ((fp4 >> 3) & 1).float()
    dequant = mag * (1 - 2 * neg)
    even = fp4[:, 0::2]; odd = fp4[:, 1::2]
    packed = (even | (odd << 4)).to(torch.uint8)
    return packed, dequant

def quant_fp4_real(tensor):
    flat = tensor.float()
    Nr, Kc = flat.shape
    ng = Kc // 32
    groups = flat.reshape(Nr, ng, 32)
    amax = groups.abs().amax(dim=-1)
    amax_bits = amax.view(torch.int32)
    e8m0 = ((amax_bits >> 23) & 0xFF).clamp(0, 255).to(torch.uint8)
    scale_f = torch.zeros_like(amax)
    nz = e8m0 > 0
    scale_f[nz] = (e8m0[nz].int() << 23).view(torch.float32)
    inv_s = torch.where(scale_f > 0, 1.0 / scale_f, torch.zeros_like(scale_f))
    scaled = groups * inv_s.unsqueeze(-1)
    abs_s = scaled.abs()
    enc = torch.zeros_like(abs_s, dtype=torch.uint8)
    enc[abs_s >= 4.0] = 7
    enc[(abs_s >= 2.5) & (abs_s < 4.0)] = 6
    enc[(abs_s >= 1.75) & (abs_s < 2.5)] = 5
    enc[(abs_s >= 1.25) & (abs_s < 1.75)] = 4
    enc[(abs_s >= 0.875) & (abs_s < 1.25)] = 3
    enc[(abs_s >= 0.625) & (abs_s < 0.875)] = 2
    enc[(abs_s >= 0.25) & (abs_s < 0.625)] = 1
    sign = (scaled < 0).to(torch.uint8) << 3
    fp4 = (sign | enc).reshape(Nr, Kc)
    even = fp4[:, 0::2]; odd = fp4[:, 1::2]
    packed = (even | (odd << 4)).to(torch.uint8)
    return packed, e8m0

# ============================================================================
# Diagnostic tests
# ============================================================================
def run_diagnostics():
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    # Small shape: M=32, N=32, K=64 (single MFMA call)
    M, N, K = 32, 32, 64
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC: M={M}, N={N}, K={K} (single MFMA)")
    print(f"{'='*70}")

    A = torch.clamp(torch.randn(M, K, dtype=torch.bfloat16, device=device), -5, 5)
    B = torch.clamp(torch.randn(N, K, dtype=torch.bfloat16, device=device), -5, 5)

    # Quantize with scale=1.0 for trivial test
    Bq_s1, Bdq_s1 = quant_fp4_s1(B, device)
    _, Adq_s1 = quant_fp4_s1(A, device)
    C_ref_trivial = (Adq_s1 @ Bdq_s1.T).bfloat16()

    print("\n--- TEST 1: Trivial scales=127 (B_q row-major) ---")
    C1 = mod.launch_trivial(A, Bq_s1, M, N, K)
    torch.cuda.synchronize()
    t1_pass = check(C1, C_ref_trivial, "trivial_scales B_q[N,K/2]")

    # Zero input sanity
    Az = torch.zeros(M, K, dtype=torch.bfloat16, device=device)
    Cz = mod.launch_trivial(Az, Bq_s1, M, N, K)
    torch.cuda.synchronize()
    check(Cz, torch.zeros(M,N,dtype=torch.bfloat16,device=device), "zero_A_input")

    print("\n--- TEST 2: Two-MFMA (trivial scales) ---")
    Bs_triv = torch.full((N, K//32), 127, dtype=torch.uint8, device=device)
    C2t = mod.launch_two_mfma(A, Bq_s1, Bs_triv, M, N, K)
    torch.cuda.synchronize()
    t2t_pass = check(C2t, C_ref_trivial, "two_mfma_trivial")

    print("\n--- TEST 2b: Two-MFMA (real scales) ---")
    Bq_real, Bs_real = quant_fp4_real(B)
    C2r = mod.launch_two_mfma(A, Bq_real, Bs_real, M, N, K)
    torch.cuda.synchronize()
    C_bf16_ref = (A.float() @ B.float().T).bfloat16()
    check(C2r, C_bf16_ref, "two_mfma_real_scales (vs bf16)")

    print("\n--- TEST 3: Full64 cbsz_sel/op_sel variants (trivial) ---")
    names = ["(0,0)", "(1,1)", "(0,1)", "(1,0)"]
    for v in range(4):
        Cv = mod.launch_full64(A, Bq_s1, Bs_triv, M, N, K, v)
        torch.cuda.synchronize()
        check(Cv, C_ref_trivial, f"full64 {names[v]} trivial")

    print("\n--- TEST 3b: Full64 variants (real scales) ---")
    for v in range(4):
        Cv = mod.launch_full64(A, Bq_real, Bs_real, M, N, K, v)
        torch.cuda.synchronize()
        check(Cv, C_bf16_ref, f"full64 {names[v]} real (vs bf16)")

    # Larger shape
    M2, N2, K2 = 32, 64, 512
    print(f"\n{'='*70}")
    print(f"LARGER: M={M2}, N={N2}, K={K2}")
    print(f"{'='*70}")
    A2 = torch.randn(M2, K2, dtype=torch.bfloat16, device=device)
    B2 = torch.randn(N2, K2, dtype=torch.bfloat16, device=device)
    Bq2_s1, Bdq2_s1 = quant_fp4_s1(B2, device)
    _, Adq2_s1 = quant_fp4_s1(A2, device)
    Cref2_t = (Adq2_s1 @ Bdq2_s1.T).bfloat16()
    Cref2_bf = (A2.float() @ B2.float().T).bfloat16()
    Bq2_r, Bs2_r = quant_fp4_real(B2)

    print("\n--- Trivial scale ---")
    C2a = mod.launch_trivial(A2, Bq2_s1, M2, N2, K2)
    torch.cuda.synchronize()
    check(C2a, Cref2_t, f"trivial {M2}x{N2}x{K2}")

    print("\n--- Two-MFMA real ---")
    C2b = mod.launch_two_mfma(A2, Bq2_r, Bs2_r, M2, N2, K2)
    torch.cuda.synchronize()
    check(C2b, Cref2_bf, f"two_mfma {M2}x{N2}x{K2} (vs bf16)")

    print("\n--- Full64 variants ---")
    for v in range(4):
        Cv = mod.launch_full64(A2, Bq2_r, Bs2_r, M2, N2, K2, v)
        torch.cuda.synchronize()
        check(Cv, Cref2_bf, f"full64 {names[v]} {M2}x{N2}x{K2}")

    # ================================================================
    # DIAGNOSIS
    # ================================================================
    print(f"\n{'='*70}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*70}")
    if t1_pass:
        print("Trivial scales PASSED -> Data loading + MFMA math CORRECT")
        print("  -> The bug is in SCALE ROUTING")
        print("  -> Fix: use two-MFMA approach (Test 2) or find correct cbsz_sel/op_sel")
    else:
        print("Trivial scales FAILED -> Data loading or register mapping is WRONG")
        print("  -> Focus on B loading pattern (byte-to-nibble repacking)")
        print("  -> Try B_shuffle instead of B_q")

from task import input_t, output_t

_first_call = True

def custom_kernel(data: input_t) -> output_t:
    global _first_call
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]

    if _first_call:
        _first_call = False
        try:
            run_diagnostics()
        except Exception as e:
            print(f"Diagnostics error: {e}", flush=True)

    # Return correct result via Triton
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    Bs = unshuffle_scale(B_scale_sh, N, K)
    return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q.view(torch.uint8),
                         A_scale.view(torch.uint8), Bs, dtype=torch.bfloat16)

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM HIP v7: Test cbsz/blgp variations + K=32 with divide-by-2.
Two approaches tested (diagnostic only, returns Triton result):
A) cbsz=1, blgp=1 — might enable per-half scale routing
B) K=32 with both halves same data, divide by 2 at end
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
import torch, sys
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

HIP_SRC = r"""
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_ext_ocp.h>
#include <cstdint>

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

// MODE A: cbsz=1, blgp=1 with K=64 per call, per-half scales
__global__ void gemm_cbsz1(
    const uint8_t* A, const uint8_t* As, const uint8_t* B, const uint8_t* Bs,
    float* C, int M, int N, int K)
{
    int Kp=K/2, Kg=K/32, lane=threadIdx.x%32, half=threadIdx.x/32;
    int mb=blockIdx.y*32, nb=blockIdx.x*32;
    fp32x16_t acc={};
    for (int k=0; k<K; k+=64) {
        fp4x64_t a_reg={}, b_reg={};
        int ar=mb+lane;
        if (ar<M) { const uint8_t* s=A+ar*Kp+k/2+half*16; for(int i=0;i<16;i++) a_reg[i]=(fp4x2_t)s[i]; }
        int bc=nb+lane;
        if (bc<N) { const uint8_t* s=B+bc*Kp+k/2+half*16; for(int i=0;i<16;i++) b_reg[i]=(fp4x2_t)s[i]; }
        uint8_t sa=127, sb=127;
        if(ar<M && k/32+half<Kg) sa=As[ar*Kg+k/32+half];
        if(bc<N && k/32+half<Kg) sb=Bs[bc*Kg+k/32+half];
        // TRY: cbsz=1, blgp=1
        acc=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg,b_reg,acc,4,4,1,(unsigned)sa,1,(unsigned)sb);
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int r=mb+half*4+j+i*8, c=nb+lane;
        if(r<M&&c<N) C[(long long)r*N+c]=acc[i*4+j];
    }
}

// MODE B: K=32 per call, both halves same data, divide by 2
__global__ void gemm_k32_div2(
    const uint8_t* A, const uint8_t* As, const uint8_t* B, const uint8_t* Bs,
    float* C, int M, int N, int K)
{
    int Kp=K/2, Kg=K/32, lane=threadIdx.x%32, half=threadIdx.x/32;
    int mb=blockIdx.y*32, nb=blockIdx.x*32;
    fp32x16_t acc={};
    for (int k=0; k<K; k+=32) {
        fp4x64_t a_reg={}, b_reg={};
        int ar=mb+lane;
        // Both halves load SAME K block (k..k+31) in first 16 bytes
        if (ar<M) { const uint8_t* s=A+ar*Kp+k/2; for(int i=0;i<16&&k/2+i<Kp;i++) a_reg[i]=(fp4x2_t)s[i]; }
        int bc=nb+lane;
        if (bc<N) { const uint8_t* s=B+bc*Kp+k/2; for(int i=0;i<16&&k/2+i<Kp;i++) b_reg[i]=(fp4x2_t)s[i]; }
        uint8_t sa=127, sb=127;
        if(ar<M && k/32<Kg) sa=As[ar*Kg+k/32];
        if(bc<N && k/32<Kg) sb=Bs[bc*Kg+k/32];
        acc=__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a_reg,b_reg,acc,4,4,0,(unsigned)sa,0,(unsigned)sb);
    }
    // NO divide by 2: MFMA reads scale from byte0 for half=0, byte1(=0) for half=1
    // Half=1's contribution is ~zero due to 2^(0-127) scale, so no doubling occurs
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int r=mb+half*4+j+i*8, c=nb+lane;
        if(r<M&&c<N) C[(long long)r*N+c]=acc[i*4+j];
    }
}

torch::Tensor launch_cbsz1(torch::Tensor Af, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, int64_t M, int64_t N, int64_t K) {
    auto C=torch::zeros({M,N},torch::TensorOptions().dtype(torch::kFloat32).device(Af.device()));
    dim3 g((N+31)/32,(M+31)/32),b(64);
    hipLaunchKernelGGL(gemm_cbsz1,g,b,0,0,(const uint8_t*)Af.data_ptr(),(const uint8_t*)As.data_ptr(),(const uint8_t*)B.data_ptr(),(const uint8_t*)Bs.data_ptr(),(float*)C.data_ptr(),(int)M,(int)N,(int)K);
    return C;
}

torch::Tensor launch_k32(torch::Tensor Af, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, int64_t M, int64_t N, int64_t K) {
    auto C=torch::zeros({M,N},torch::TensorOptions().dtype(torch::kFloat32).device(Af.device()));
    dim3 g((N+31)/32,(M+31)/32),b(64);
    hipLaunchKernelGGL(gemm_k32_div2,g,b,0,0,(const uint8_t*)Af.data_ptr(),(const uint8_t*)As.data_ptr(),(const uint8_t*)B.data_ptr(),(const uint8_t*)Bs.data_ptr(),(float*)C.data_ptr(),(int)M,(int)N,(int)K);
    return C;
}
"""

CPP_FWD = """
torch::Tensor launch_cbsz1(torch::Tensor Af, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, int64_t M, int64_t N, int64_t K);
torch::Tensor launch_k32(torch::Tensor Af, torch::Tensor As, torch::Tensor B, torch::Tensor Bs, int64_t M, int64_t N, int64_t K);
"""

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(name="v7_cbsz", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                           functions=["launch_cbsz1","launch_k32"],
                           extra_cuda_cflags=["-O3","--offload-arch=gfx950"], verbose=True)
    return _mod

_sc = {}
def _unshuffle(s, N, K):
    key = id(s)
    if key in _sc: return _sc[key]
    n = K//32; sm=((N+255)//256)*256; sn=((n+7)//8)*8
    p = torch.zeros(sm,sn,dtype=torch.uint8,device=s.device)
    p[:N,:n] = s.view(torch.uint8)[:N,:n]
    r = p.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous()
    result = r.view(sm,sn)[:N,:n]
    _sc[key] = result
    return result

_first = True
def custom_kernel(data: input_t) -> output_t:
    global _first
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    mod = _get_mod()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    Bs = _unshuffle(B_scale_sh, N, K)
    Au, Asu, Bu = A_fp4.view(torch.uint8), A_scale.view(torch.uint8), B_q.view(torch.uint8)

    ref = gemm_afp4wfp4(Au, Bu, Asu, Bs, dtype=torch.bfloat16)

    if _first:
        _first = False
        # Test both modes
        cA = mod.launch_cbsz1(Au, Asu, Bu, Bs, M, N, K)
        cB = mod.launch_k32(Au, Asu, Bu, Bs, M, N, K)
        torch.cuda.synchronize()
        ref_f = ref.float()
        errA = (cA - ref_f).abs()
        errB = (cB - ref_f).abs()
        relA = errA / (ref_f.abs() + 1e-6)
        relB = errB / (ref_f.abs() + 1e-6)
        print(f"[V7] M={M} N={N} K={K}", file=sys.stderr)
        print(f"[CBSZ1] max_err={errA.max():.2f} mean_rel={relA.mean():.4f} ref[0,:4]={ref_f[0,:4].tolist()} out[0,:4]={cA[0,:4].tolist()}", file=sys.stderr)
        print(f"[K32D2] max_err={errB.max():.2f} mean_rel={relB.mean():.4f} out[0,:4]={cB[0,:4].tolist()}", file=sys.stderr)
        sys.stderr.flush()

    return ref

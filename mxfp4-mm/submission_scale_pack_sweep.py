#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Scale Packing Sweep: Try all possible scale VGPR formats.
The MFMA 32x32x64 processes K=64 with 2 scale blocks (K=0..31, K=32..63).
The scale VGPR must encode both scales. Try:
  Pack A: sa = sa0 | (sa1 << 8)    Pack B: sb = sb0 | (sb1 << 8)   [original Navi v3]
  Pack C: sa = sa0 | (sa1 << 16)   Pack D: sb = sb0 | (sb1 << 16)
  Pack E: sa = sa0,  sb = sb0 | (sb1 << 8)  [only pack B]
  Pack F: sa = sa0 | (sa1 << 8), sb = sb0   [only pack A]
Half-fill B (16 bytes + 16 zeros), full A (16 bytes + 16 zeros).
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

using fp4x2_t = __amd_fp4x2_storage_t;
using fp4x64_t = fp4x2_t __attribute__((ext_vector_type(32)));
using fp32x16_t = __attribute__((vector_size(16 * sizeof(float)))) float;

// mode: 0=both_byte, 1=both_word, 2=pack_b_only, 3=pack_a_only
__global__ void gemm_kernel(
    const uint8_t* A, const uint8_t* As, const uint8_t* B, const uint8_t* Bs,
    hip_bfloat16* C, int M, int N, int K, int pack_mode)
{
    int Kp=K/2, Kg=K/32, lane=threadIdx.x%32, half=threadIdx.x/32;
    int mb=blockIdx.y*32, nb=blockIdx.x*32;
    fp32x16_t acc={};

    for (int k=0; k<K; k+=64) {
        // Half-fill: each thread loads only its K half (16 bytes)
        fp4x64_t a_reg={}, b_reg={};
        int ar=mb+lane, bc=nb+lane;

        if (ar<M) {
            const uint8_t* s=A+(long long)ar*Kp+k/2+half*16;
            for(int i=0;i<16&&(k/2+half*16+i)<Kp;i++) a_reg[i]=(fp4x2_t)s[i];
        }
        if (bc<N) {
            const uint8_t* s=B+(long long)bc*Kp+k/2+half*16;
            for(int i=0;i<16&&(k/2+half*16+i)<Kp;i++) b_reg[i]=(fp4x2_t)s[i];
        }

        uint8_t sa0=127, sa1=127, sb0=127, sb1=127;
        if(ar<M) {
            int g=k/32; if(g<Kg) sa0=As[(long long)ar*Kg+g]; if(g+1<Kg) sa1=As[(long long)ar*Kg+g+1];
        }
        if(bc<N) {
            int g=k/32; if(g<Kg) sb0=Bs[(long long)bc*Kg+g]; if(g+1<Kg) sb1=Bs[(long long)bc*Kg+g+1];
        }

        unsigned sa, sb;
        if (pack_mode == 0) {
            // Pack both in bytes: sa = sa0|sa1<<8, sb = sb0|sb1<<8
            sa = (unsigned)sa0 | ((unsigned)sa1 << 8);
            sb = (unsigned)sb0 | ((unsigned)sb1 << 8);
        } else if (pack_mode == 1) {
            // Pack both in words: sa = sa0|sa1<<16, sb = sb0|sb1<<16
            sa = (unsigned)sa0 | ((unsigned)sa1 << 16);
            sb = (unsigned)sb0 | ((unsigned)sb1 << 16);
        } else if (pack_mode == 2) {
            // Single A scale (this half), pack B
            sa = (half == 0) ? (unsigned)sa0 : (unsigned)sa1;
            sb = (unsigned)sb0 | ((unsigned)sb1 << 8);
        } else {
            // Pack A, single B scale
            sa = (unsigned)sa0 | ((unsigned)sa1 << 8);
            sb = (half == 0) ? (unsigned)sb0 : (unsigned)sb1;
        }

        acc = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(
            a_reg, b_reg, acc, 4, 4, 0, sa, 0, sb);
    }

    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int r=mb+half*4+j+i*8, c=nb+lane;
        if(r<M&&c<N) C[(long long)r*N+c]=(hip_bfloat16)acc[i*4+j];
    }
}

torch::Tensor launch(torch::Tensor Af, torch::Tensor As, torch::Tensor Bq, torch::Tensor Bs,
                     int64_t M, int64_t N, int64_t K, int64_t mode) {
    auto C=torch::empty({M,N},torch::TensorOptions().dtype(torch::kBFloat16).device(Af.device()));
    dim3 g((N+31)/32,(M+31)/32), b(64);
    hipLaunchKernelGGL(gemm_kernel,g,b,0,0,
        (const uint8_t*)Af.data_ptr(),(const uint8_t*)As.data_ptr(),
        (const uint8_t*)Bq.data_ptr(),(const uint8_t*)Bs.data_ptr(),
        (hip_bfloat16*)C.data_ptr(),(int)M,(int)N,(int)K,(int)mode);
    return C;
}
"""

CPP_FWD = "torch::Tensor launch(torch::Tensor Af, torch::Tensor As, torch::Tensor Bq, torch::Tensor Bs, int64_t M, int64_t N, int64_t K, int64_t mode);"

_mod = None
def _get_mod():
    global _mod
    if _mod is None:
        _mod = load_inline(name="scale_pack_sweep", cpp_sources=CPP_FWD, cuda_sources=HIP_SRC,
                           functions=["launch"], extra_cuda_cflags=["-O3","--offload-arch=gfx950"], verbose=True)
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
    A, B, B_q, _, B_scale_sh = data
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
        labels = ["byte_pack", "word_pack", "pack_b_only", "pack_a_only"]
        for mode in range(4):
            c = mod.launch(Au, Asu, Bu, Bs, M, N, K, mode)
            torch.cuda.synchronize()
            err = (c.float() - ref.float()).abs()
            rel = err / (ref.float().abs() + 1e-6)
            print(f"[{labels[mode]}] max_err={err.max():.2f} mean_rel={rel.mean():.4f} "
                  f"out[0,:4]={c[0,:4].tolist()}", file=sys.stderr)
        print(f"[REF] ref[0,:4]={ref[0,:4].tolist()}", file=sys.stderr)
        sys.stderr.flush()
    return ref

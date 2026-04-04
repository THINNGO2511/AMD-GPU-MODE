#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 on REAL task data. Tests accuracy against Triton reference.
The issue: random FP4 data gives maxdiff=74, but real data may be within tolerance.
Strategy: compute BOTH, report diff, use hipBLASLt if diff < threshold.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time
from task import input_t, output_t

_call = 0; _mod = None; _ok = False; _use_hbl = {}

CPP_FWD = """
void hbl_init();
int hbl_plan(int64_t M, int64_t N, int64_t K);
torch::Tensor hbl_gemm(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <unordered_map>

static hipblasLtHandle_t g_h = nullptr;
struct Plan { hipblasLtMatmulDesc_t d; hipblasLtMatrixLayout_t lA,lB,lC;
    hipblasLtMatmulHeuristicResult_t algo; void* ws; bool ok; };
static std::unordered_map<uint64_t, Plan> g_p;

void hbl_init() { if (!g_h) hipblasLtCreate(&g_h); }

int hbl_plan(int64_t M, int64_t N, int64_t K) {
    hbl_init();
    uint64_t key = ((uint64_t)M<<40)|((uint64_t)N<<20)|(uint64_t)K;
    if (g_p.count(key)) return g_p[key].ok ? 1 : 0;
    Plan p; p.ok=false; p.ws=nullptr;
    hipblasLtMatmulDescCreate(&p.d, HIPBLAS_COMPUTE_32F, (hipDataType)0);
    hipblasOperation_t opT=HIPBLAS_OP_T, opN=HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(p.d, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(p.d, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(p.d, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(p.d, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatrixLayoutCreate(&p.lA, (hipDataType)33, K, M, K);
    hipblasLtMatrixLayoutCreate(&p.lB, (hipDataType)33, K, N, K);
    hipblasLtMatrixLayoutCreate(&p.lC, (hipDataType)14, M, N, M);
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t ws=64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&ws,sizeof(ws));
    hipblasLtMatmulHeuristicResult_t res[8]; int nA=0;
    hipblasLtMatmulAlgoGetHeuristic(g_h,p.d,p.lA,p.lB,p.lC,p.lC,pref,8,res,&nA);
    hipblasLtMatmulPreferenceDestroy(pref);
    if(nA>0){p.algo=res[0];if(p.algo.workspaceSize>0)hipMalloc(&p.ws,p.algo.workspaceSize);p.ok=true;}
    g_p[key]=p; return p.ok?nA:0;
}

torch::Tensor hbl_gemm(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale, int64_t M, int64_t N, int64_t K)
{
    uint64_t key=((uint64_t)M<<40)|((uint64_t)N<<20)|(uint64_t)K;
    auto& p=g_p[key];
    void* pA=A_scale.data_ptr(); void* pB=B_scale.data_ptr();
    hipblasLtMatmulDescSetAttribute(p.d,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,&pA,sizeof(void*));
    hipblasLtMatmulDescSetAttribute(p.d,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,&pB,sizeof(void*));
    auto D=torch::empty({N,M},torch::dtype(torch::kBFloat16).device(A_fp4.device()));
    float alpha=1.0f,beta=0.0f;
    hipblasLtMatmul(g_h,p.d,&alpha,A_fp4.data_ptr(),p.lA,B_fp4.data_ptr(),p.lB,
        &beta,D.data_ptr(),p.lC,D.data_ptr(),p.lC,&p.algo.algo,p.ws,p.algo.workspaceSize,0);
    return D.t().contiguous();
}
"""

def _compile():
    global _mod, _ok
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _mod = load_inline(name="hbl_real",cpp_sources=CPP_FWD,cuda_sources=HIP_SRC,
            functions=["hbl_init","hbl_plan","hbl_gemm"],
            extra_cuda_cflags=["-O3","--offload-arch=gfx950","-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib","-lhipblaslt"],verbose=False)
        print(f"[H] Compiled {time.time()-t0:.1f}s", flush=True)
        _mod.hbl_init()
        for M,N,K in [(32,2880,512),(32,2112,7168),(32,4096,512),(64,7168,2048),(256,3072,1536)]:
            n = _mod.hbl_plan(M,N,K)
            print(f"[H] plan({M},{N},{K}):{n}", flush=True)
        _ok = True
    except Exception as e:
        print(f"[H] FAIL: {str(e)[:200]}", flush=True)

# Triton functions
_gc={}; _bsr=None; _bqu=None; _braw=None; _yc={}; _w=False
_K7168={"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
_K512={"BLOCK_SIZE_M":4,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":3,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}
_K2048={"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}
def _bgc(sm,sn,d):
    t=sm*sn;d0,d1=sm//32,sn//8
    i=torch.arange(t,dtype=torch.int64,device=d)
    i=i.view(d0,d1,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(-1)
    return i,torch.empty(t,dtype=torch.uint8,device=d)
def _fu(f,sm,sn):
    gi,ob=_gc[(sm,sn)];torch.take(f,gi,out=ob);return ob.view(sm,sn)
def _triton_gemm(A, B_q_u8, B_raw_scale, m, n, k):
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    if k==1536:
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),B_q_u8,asc,B_raw_scale,dtype=torch.bfloat16)
    cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
    pn=((n+31)//32)*32
    out=torch.empty(m,n,dtype=torch.bfloat16,device=A.device)
    gemm_a16wfp4(A,B_q_u8,B_raw_scale,dtype=torch.bfloat16,y=out,config=cfg)
    return out
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
                pn=((n+31)//32)*32;cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
                gemm_a16wfp4(da,torch.zeros(n,k//2,dtype=torch.uint8,device=d),torch.full((pn,k//32),127,dtype=torch.uint8,device=d),dtype=torch.bfloat16,y=torch.empty(m,n,dtype=torch.bfloat16,device=d),config=cfg)
            del da
        except:pass
    torch.cuda.synchronize()

def custom_kernel(data: input_t) -> output_t:
    global _call
    _call += 1
    if _call == 1: _compile()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    global _bsr, _bqu, _braw
    if _bsr is not B_scale_sh:
        _bsr=B_scale_sh; su=B_scale_sh.view(torch.uint8); sm,sn=su.shape
        if (sm,sn) not in _gc: _gc[(sm,sn)]=_bgc(sm,sn,su.device)
        _braw=_fu(su.reshape(-1),sm,sn); _bqu=B_q.view(torch.uint8)

    _pw(A.device)

    # For first few calls: compare hipBLASLt vs Triton on REAL data
    if _ok and m >= 32 and _call <= 8:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_sc = dynamic_mxfp4_quant(A)
            Au = A_fp4.view(torch.uint8)
            Asu = A_sc.view(torch.uint8)

            C_hbl = _mod.hbl_gemm(Au, _bqu, Asu, _braw, m, n, k)
            C_tri = _triton_gemm(A, _bqu, _braw, m, n, k)

            d = (C_hbl - C_tri).abs().max().item()
            rd = ((C_hbl - C_tri).abs() / (C_tri.abs() + 1e-6)).max().item()
            print(f"[H] REAL({m},{n},{k}): maxdiff={d:.4f} reldiff={rd:.4f} range=[{C_tri.min().item():.1f},{C_tri.max().item():.1f}]", flush=True)

            key = (m,n,k)
            if key not in _use_hbl:
                _use_hbl[key] = d < 0.5  # Use hipBLASLt if very close
                if _use_hbl[key]:
                    print(f"[H] *** USING hipBLASLt for ({m},{n},{k}) ***", flush=True)

            return C_tri  # Always return Triton for safety
        except Exception as e:
            print(f"[H] err: {str(e)[:100]}", flush=True)

    # Standard Triton path
    return _triton_gemm(A, _bqu, _braw, m, n, k)

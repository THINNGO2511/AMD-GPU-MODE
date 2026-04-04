#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 with BYTE TRANSPOSITION fix.
Problem: row-major [M, K/2] FP4 bytes != col-major [K/2, M] FP4 bytes.
Fix: A_fp4.view(M, K//2).t().contiguous() → [K/2, M]
Tests: transposed FP4 + {raw, transposed, shuffled} scales.
Also compares vs gemm_a4w4_asm reference.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time
from task import input_t, output_t

_call = 0; _mod = None; _ok = False; _best_config = None

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
    global _mod, _ok, _best_config
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _mod = load_inline(name="hbl_t",cpp_sources=CPP_FWD,cuda_sources=HIP_SRC,
            functions=["hbl_init","hbl_plan","hbl_gemm"],
            extra_cuda_cflags=["-O3","--offload-arch=gfx950","-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib","-lhipblaslt"],verbose=False)
        print(f"[T] Compiled {time.time()-t0:.1f}s", flush=True)
        _mod.hbl_init()

        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        M, N, K = 32, 64, 512
        _mod.hbl_plan(M, N, K)

        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
        Au = A_fp4.view(torch.uint8)  # [M, K/2]
        Asu = A_sc.view(torch.uint8)  # [M, K/32]
        Bu = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
        Bsu = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
        C_ref = gemm_afp4wfp4(Au, Bu, Asu, Bsu, dtype=torch.bfloat16)

        print(f"[T] Au={Au.shape} Asu={Asu.shape} Bu={Bu.shape}", flush=True)

        # === Test all combinations of FP4 transpose + scale variants ===
        best_d = 999; best_name = ""

        # FP4 byte transpositions
        Au_notrans = Au  # [M, K/2] as-is
        Au_trans = Au.view(M, K//2).t().contiguous()  # [K/2, M]
        Bu_notrans = Bu  # [N, K/2]
        Bu_trans = Bu.view(N, K//2).t().contiguous()  # [K/2, N]

        # Scale variants
        Asu_notrans = Asu  # [M, K/32]
        Asu_trans = Asu.view(M, K//32).t().contiguous()  # [K/32, M]
        Bsu_notrans = Bsu  # [N, K/32]
        Bsu_trans = Bsu.view(N, K//32).t().contiguous()  # [K/32, N]

        configs = [
            # (name, A_fp4, B_fp4, A_scale, B_scale)
            ("noT_noT", Au_notrans, Bu_notrans, Asu_notrans, Bsu_notrans),
            ("AT_noT", Au_trans, Bu_notrans, Asu_notrans, Bsu_notrans),
            ("noT_BT", Au_notrans, Bu_trans, Asu_notrans, Bsu_notrans),
            ("AT_BT", Au_trans, Bu_trans, Asu_notrans, Bsu_notrans),
            ("AT_BT_sAT", Au_trans, Bu_trans, Asu_trans, Bsu_notrans),
            ("AT_BT_sBT", Au_trans, Bu_trans, Asu_notrans, Bsu_trans),
            ("AT_BT_sAT_sBT", Au_trans, Bu_trans, Asu_trans, Bsu_trans),
            ("noT_noT_sAT_sBT", Au_notrans, Bu_notrans, Asu_trans, Bsu_trans),
            ("AT_noT_sAT", Au_trans, Bu_notrans, Asu_trans, Bsu_notrans),
            ("noT_BT_sBT", Au_notrans, Bu_trans, Asu_notrans, Bsu_trans),
        ]

        for name, af, bf, asc, bsc in configs:
            try:
                C = _mod.hbl_gemm(af, bf, asc, bsc, M, N, K)
                d = (C - C_ref).abs().max().item()
                nan_count = torch.isnan(C).sum().item()
                tag = f" NaN={nan_count}" if nan_count > 0 else ""
                print(f"[T] {name}: maxdiff={d:.4f}{tag}", flush=True)
                if d < best_d and nan_count == 0:
                    best_d = d; best_name = name
            except Exception as e:
                print(f"[T] {name}: ERR {str(e)[:80]}", flush=True)

        # Also try nibble swap + transpose
        Au_swap = ((Au >> 4) | (Au << 4)).to(torch.uint8)
        Au_swap_trans = Au_swap.view(M, K//2).t().contiguous()
        Bu_swap = ((Bu >> 4) | (Bu << 4)).to(torch.uint8)
        Bu_swap_trans = Bu_swap.view(N, K//2).t().contiguous()

        for name, af, bf, asc, bsc in [
            ("swAT_swBT", Au_swap_trans, Bu_swap_trans, Asu_notrans, Bsu_notrans),
            ("swAT_swBT_sT", Au_swap_trans, Bu_swap_trans, Asu_trans, Bsu_trans),
            ("swAT_BT", Au_swap_trans, Bu_trans, Asu_notrans, Bsu_notrans),
            ("AT_swBT", Au_trans, Bu_swap_trans, Asu_notrans, Bsu_notrans),
        ]:
            try:
                C = _mod.hbl_gemm(af, bf, asc, bsc, M, N, K)
                d = (C - C_ref).abs().max().item()
                nan_count = torch.isnan(C).sum().item()
                tag = f" NaN={nan_count}" if nan_count > 0 else ""
                print(f"[T] {name}: maxdiff={d:.4f}{tag}", flush=True)
                if d < best_d and nan_count == 0:
                    best_d = d; best_name = name
            except Exception as e:
                print(f"[T] {name}: ERR {str(e)[:80]}", flush=True)

        print(f"[T] === BEST: {best_name} maxdiff={best_d:.4f} ===", flush=True)

        if best_d < 1.0:
            _ok = True; _best_config = best_name
            print(f"[T] *** ACCURACY PASS! Config={best_name} ***", flush=True)

            # Benchmark the winning config
            torch.cuda.synchronize()
            t0 = time.time()
            # Get the right tensors for the winning config
            cfg_map = dict(configs)
            af, bf, asc, bsc = cfg_map.get(best_name, (Au_notrans, Bu_notrans, Asu_notrans, Bsu_notrans))
            for _ in range(50):
                _mod.hbl_gemm(af, bf, asc, bsc, M, N, K)
            torch.cuda.synchronize()
            us = (time.time()-t0)/50*1e6
            print(f"[T] Benchmark ({M},{N},{K}): {us:.1f}us", flush=True)

    except Exception as e:
        import traceback
        print(f"[T] FAIL: {str(e)[:300]}", flush=True)
        traceback.print_exc()

# Triton fallback
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
    A,B,B_q,B_shuffle,B_scale_sh=data;m,k=A.shape;n=B.shape[0]
    global _bsr,_bqu,_braw
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

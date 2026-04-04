#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — DEFINITIVE hipBLASLt FP4 test with CORRECT setup.
Research found: opA MUST be T, opB MUST be N for block-scaled FP4.
Previous probes used wrong transpose! This is the missing test.
Also: workspace required, M%16, N%16, K%128 alignment.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch, time, sys
from task import input_t, output_t

_call = 0; _probed = False

CPP_FWD = """
std::string probe_TN_fp4_v1(int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <string>

// Type constants
static const hipDataType FP4   = (hipDataType)33;  // HIP_R_4F_E2M1
static const hipDataType FP8   = (hipDataType)28;  // HIP_R_8F_E4M3
static const hipDataType BF16  = (hipDataType)14;
static const hipDataType F32   = (hipDataType)0;

std::string probe_TN_fp4_v1(int64_t M, int64_t N, int64_t K) {
    std::string r;
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);

    // === THE DEFINITIVE TEST: opA=T, opB=N with FP4 + block scales ===

    // Per research: D[M,N] = alpha * A^T[K,M] * B[K,N] + beta * C[M,N]
    // opA=T means A is stored as (K,M) col-major, transposed to (M,K)
    // opB=N means B is stored as (K,N) col-major, not transposed

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32);

    // SET TRANSPOSE: opA=T, opB=N (MANDATORY for block-scaled!)
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // SET SCALE MODE: VEC32_UE8M0 for both A and B
    int32_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    auto s1 = hipblasLtMatmulDescSetAttribute(desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    auto s2 = hipblasLtMatmulDescSetAttribute(desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    r += "SM_A:" + std::to_string((int)s1) + " SM_B:" + std::to_string((int)s2) + " ";

    // Allocate dummy scale pointers on device
    void* d_scaleA = nullptr;
    void* d_scaleB = nullptr;
    hipMalloc(&d_scaleA, (K/32) * M);  // E8M0 scales: [K/32, M]
    hipMalloc(&d_scaleB, (K/32) * N);  // E8M0 scales: [K/32, N]

    // SET SCALE POINTERS
    hipblasLtMatmulDescSetAttribute(desc,
        HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc,
        HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(void*));

    // CREATE MATRIX LAYOUTS
    // A stored as (K, M) col-major (transposed to get M,K)
    // B stored as (K, N) col-major (not transposed)
    // C/D stored as (M, N) col-major
    hipblasLtMatrixLayout_t lA, lB, lC;
    auto sA = hipblasLtMatrixLayoutCreate(&lA, FP4, K, M, K);   // (K,M), ld=K
    auto sB = hipblasLtMatrixLayoutCreate(&lB, FP4, K, N, K);   // (K,N), ld=K
    auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16, M, N, M);  // (M,N), ld=M
    r += "LA:" + std::to_string((int)sA) + " LB:" + std::to_string((int)sB) + " LC:" + std::to_string((int)sC) + " ";

    // GET HEURISTICS WITH WORKSPACE
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 32*1024*1024; // 32MB workspace
    hipblasLtMatmulPreferenceSetAttribute(pref,
        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC,
        pref, 8, results, &nAlgo);
    r += "HEUR_TN:" + std::to_string((int)sH) + " ALGOS=" + std::to_string(nAlgo) + " ";

    // Also try with ext type (BLK32_UE8M0_32_8_EXT = 6)
    if (nAlgo == 0) {
        int32_t extMode = 6; // HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT
        hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &extMode, sizeof(extMode));
        hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &extMode, sizeof(extMode));
        int nAlgo2 = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC,
            pref, 8, results, &nAlgo2);
        r += "EXT_ALGOS=" + std::to_string(nAlgo2) + " ";
    }

    // Try FP8 layout with TN + scales (as comparison)
    {
        hipblasLtMatmulDesc_t desc2;
        hipblasLtMatmulDescCreate(&desc2, HIPBLAS_COMPUTE_32F, F32);
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(void*));
        hipblasLtMatmulDescSetAttribute(desc2, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(void*));

        hipblasLtMatrixLayout_t lA2, lB2, lC2;
        hipblasLtMatrixLayoutCreate(&lA2, FP8, K, M, K);
        hipblasLtMatrixLayoutCreate(&lB2, FP8, K, N, K);
        hipblasLtMatrixLayoutCreate(&lC2, BF16, M, N, M);

        hipblasLtMatmulHeuristicResult_t res2[8];
        int nAlgo3 = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc2, lA2, lB2, lC2, lC2,
            pref, 8, res2, &nAlgo3);
        r += "FP8_TN_SCALE_ALGOS=" + std::to_string(nAlgo3) + " ";

        hipblasLtMatrixLayoutDestroy(lA2);
        hipblasLtMatrixLayoutDestroy(lB2);
        hipblasLtMatrixLayoutDestroy(lC2);
        hipblasLtMatmulDescDestroy(desc2);
    }

    // Try FP4 with ONLY A scale (B no scale)
    {
        hipblasLtMatmulDesc_t desc3;
        hipblasLtMatmulDescCreate(&desc3, HIPBLAS_COMPUTE_32F, F32);
        hipblasLtMatmulDescSetAttribute(desc3, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc3, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        int32_t sm_a = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        int32_t sm_b = 0; // SCALAR_32F
        hipblasLtMatmulDescSetAttribute(desc3, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_a, sizeof(sm_a));
        hipblasLtMatmulDescSetAttribute(desc3, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_b, sizeof(sm_b));

        hipblasLtMatrixLayout_t lA3, lB3, lC3;
        hipblasLtMatrixLayoutCreate(&lA3, FP4, K, M, K);
        hipblasLtMatrixLayoutCreate(&lB3, FP4, K, N, K);
        hipblasLtMatrixLayoutCreate(&lC3, BF16, M, N, M);

        hipblasLtMatmulHeuristicResult_t res3[8];
        int nAlgo4 = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc3, lA3, lB3, lC3, lC3,
            pref, 8, res3, &nAlgo4);
        r += "FP4_TN_SCALAR_B_ALGOS=" + std::to_string(nAlgo4) + " ";

        hipblasLtMatrixLayoutDestroy(lA3);
        hipblasLtMatrixLayoutDestroy(lB3);
        hipblasLtMatrixLayoutDestroy(lC3);
        hipblasLtMatmulDescDestroy(desc3);
    }

    // Try FP4 TN without any scale mode set
    {
        hipblasLtMatmulDesc_t desc4;
        hipblasLtMatmulDescCreate(&desc4, HIPBLAS_COMPUTE_32F, F32);
        hipblasLtMatmulDescSetAttribute(desc4, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc4, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        hipblasLtMatrixLayout_t lA4, lB4, lC4;
        hipblasLtMatrixLayoutCreate(&lA4, FP4, K, M, K);
        hipblasLtMatrixLayoutCreate(&lB4, FP4, K, N, K);
        hipblasLtMatrixLayoutCreate(&lC4, BF16, M, N, M);

        hipblasLtMatmulHeuristicResult_t res4[8];
        int nAlgo5 = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc4, lA4, lB4, lC4, lC4,
            pref, 8, res4, &nAlgo5);
        r += "FP4_TN_NOSCALE_ALGOS=" + std::to_string(nAlgo5) + " ";

        hipblasLtMatrixLayoutDestroy(lA4);
        hipblasLtMatrixLayoutDestroy(lB4);
        hipblasLtMatrixLayoutDestroy(lC4);
        hipblasLtMatmulDescDestroy(desc4);
    }

    // Try plain FP8 TN (no scales, baseline)
    {
        hipblasLtMatmulDesc_t desc5;
        hipblasLtMatmulDescCreate(&desc5, HIPBLAS_COMPUTE_32F, F32);
        hipblasLtMatmulDescSetAttribute(desc5, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        hipblasLtMatmulDescSetAttribute(desc5, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        hipblasLtMatrixLayout_t lA5, lB5, lC5;
        hipblasLtMatrixLayoutCreate(&lA5, FP8, K, M, K);
        hipblasLtMatrixLayoutCreate(&lB5, FP8, K, N, K);
        hipblasLtMatrixLayoutCreate(&lC5, BF16, M, N, M);

        hipblasLtMatmulHeuristicResult_t res5[8];
        int nAlgo6 = 0;
        hipblasLtMatmulAlgoGetHeuristic(handle, desc5, lA5, lB5, lC5, lC5,
            pref, 8, res5, &nAlgo6);
        r += "FP8_TN_NOSCALE_ALGOS=" + std::to_string(nAlgo6) + " ";

        hipblasLtMatrixLayoutDestroy(lA5);
        hipblasLtMatrixLayoutDestroy(lB5);
        hipblasLtMatrixLayoutDestroy(lC5);
        hipblasLtMatmulDescDestroy(desc5);
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
    hipFree(d_scaleA);
    hipFree(d_scaleB);
    hipblasLtDestroy(handle);
    return r;
}
"""

def _run_probe():
    global _probed
    if _probed: return
    _probed = True
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        mod = load_inline(
            name="hipblaslt_TN_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["probe_TN_fp4_v1"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        t1 = time.time()
        print(f"[PROBE] Compiled in {t1-t0:.1f}s", flush=True)

        # Test aligned shapes (M%16, N%16, K%128)
        for M, N, K in [(16, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
                         (64, 7168, 2048), (256, 3072, 1536), (16, 128, 128)]:
            r = mod.probe_TN_fp4_v1(M, N, K)
            print(f"[PROBE] ({M},{N},{K}): {r}", flush=True)

    except Exception as e:
        import traceback
        print(f"[PROBE] FAIL: {str(e)[:500]}", flush=True)
        traceback.print_exc()

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
    if _call == 1: _run_probe()
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

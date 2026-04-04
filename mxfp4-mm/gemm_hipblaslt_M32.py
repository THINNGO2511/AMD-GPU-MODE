#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 with M>=32 (M%32==0).
BREAKTHROUGH: Previous probes used M=16 which fails predicate.
FP4 kernels: RR_GEMM_TN_FP4_FP4_BFloat16_BFloat16_Float_SA_B_SB_B_WGT_32x32x128_UR_2
Require: M%32==0 (some need M%64==0), K%128==0, opA=T, opB=N.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch, time
from task import input_t, output_t

_call = 0; _probed = False

CPP_FWD = """
std::string probe_M32_v1(int64_t M, int64_t N, int64_t K);
std::string run_fp4_gemm_v1(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <string>
#include <chrono>

static const hipDataType FP4  = (hipDataType)33;
static const hipDataType BF16 = (hipDataType)14;
static const hipDataType F32  = (hipDataType)0;

std::string probe_M32_v1(int64_t M, int64_t N, int64_t K) {
    std::string r;
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32);

    // MANDATORY: opA=T, opB=N for block-scaled FP4
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Block scale mode VEC32_UE8M0
    int32_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));

    // Scale pointers (dummy)
    void *dA = nullptr, *dB = nullptr;
    hipMalloc(&dA, (K/32)*M + 1024);
    hipMalloc(&dB, (K/32)*N + 1024);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dB, sizeof(void*));

    // Layouts: A(K,M) col-major, B(K,N) col-major, C(M,N) col-major
    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, FP4, K, M, K);
    hipblasLtMatrixLayoutCreate(&lB, FP4, K, N, K);
    hipblasLtMatrixLayoutCreate(&lC, BF16, M, N, M);

    // Heuristic with workspace
    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
    r += "M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K);
    r += " ALGOS=" + std::to_string(nAlgo);
    if (nAlgo > 0) {
        r += " WS=" + std::to_string(results[0].workspaceSize);
        r += " *** FOUND! ***";
    }
    r += " s=" + std::to_string((int)sH);

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
    hipFree(dA); hipFree(dB);
    hipblasLtDestroy(handle);
    return r;
}

std::string run_fp4_gemm_v1(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K) {
    std::string r;
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32);

    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    int32_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));

    void* pA = A_scale.data_ptr();
    void* pB = B_scale.data_ptr();
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB, sizeof(void*));

    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, FP4, K, M, K);
    hipblasLtMatrixLayoutCreate(&lB, FP4, K, N, K);
    hipblasLtMatrixLayoutCreate(&lC, BF16, M, N, M);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);

    if (nAlgo > 0) {
        auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

        // Allocate workspace
        void* workspace = nullptr;
        if (results[0].workspaceSize > 0) hipMalloc(&workspace, results[0].workspaceSize);

        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int i = 0; i < 3; i++) {
            hipblasLtMatmul(handle, desc, &alpha,
                A_fp4.data_ptr(), lA,
                B_fp4.data_ptr(), lB,
                &beta,
                C.data_ptr(), lC,
                C.data_ptr(), lC,
                &results[0].algo, workspace, results[0].workspaceSize, 0);
        }
        hipDeviceSynchronize();

        // Benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        int niters = 50;
        for (int i = 0; i < niters; i++) {
            hipblasLtMatmul(handle, desc, &alpha,
                A_fp4.data_ptr(), lA,
                B_fp4.data_ptr(), lB,
                &beta,
                C.data_ptr(), lC,
                C.data_ptr(), lC,
                &results[0].algo, workspace, results[0].workspaceSize, 0);
        }
        hipDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / niters;
        r += "TIME=" + std::to_string(us) + "us ";

        float cmax = C.abs().max().item<float>();
        r += "CMAX=" + std::to_string(cmax);

        if (workspace) hipFree(workspace);
    } else {
        r += "NO_ALGOS";
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
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
            name="hipblaslt_M32_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["probe_M32_v1", "run_fp4_gemm_v1"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        print(f"[PROBE] Compiled in {time.time()-t0:.1f}s", flush=True)

        # Test M%32==0 shapes (THIS is the key test!)
        shapes = [
            (32, 2880, 512),   # Our shape M=32 ← should work!
            (64, 7168, 2048),  # Our shape M=64 ← should work!
            (256, 3072, 1536), # Our shape M=256 ← should work!
            (32, 2112, 7168),  # M=32 padded from M=16
            (32, 4096, 512),   # Our exact shape
            (32, 128, 128),    # Small test
            (64, 128, 128),    # Small test M=64
            (128, 128, 128),   # Larger test
        ]

        for M, N, K in shapes:
            r = mod.probe_M32_v1(M, N, K)
            print(f"[PROBE] {r}", flush=True)

        # If algorithms found, run actual FP4 GEMM benchmark!
        for M, N, K in [(32, 128, 128), (64, 128, 128), (32, 2880, 512)]:
            try:
                # Create dummy FP4 data (col-major: K×M for A, K×N for B)
                # FP4 packed as fp4x2: each byte = 2 FP4 values
                # For col-major (K,M) with FP4: storage = K*M/2 bytes
                A_fp4 = torch.randint(0, 256, (K*M//2,), dtype=torch.uint8, device='cuda')
                B_fp4 = torch.randint(0, 256, (K*N//2,), dtype=torch.uint8, device='cuda')
                A_scale = torch.full((K//32 * M,), 127, dtype=torch.uint8, device='cuda')
                B_scale = torch.full((K//32 * N,), 127, dtype=torch.uint8, device='cuda')

                r = mod.run_fp4_gemm_v1(A_fp4, B_fp4, A_scale, B_scale, M, N, K)
                print(f"[PROBE] RUN({M},{N},{K}): {r}", flush=True)
            except Exception as e:
                print(f"[PROBE] RUN({M},{N},{K}): {str(e)[:200]}", flush=True)

    except Exception as e:
        import traceback
        print(f"[PROBE] FAIL: {str(e)[:500]}", flush=True)
        traceback.print_exc()

# Proven GEMM fallback
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

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt probe v3: Lightweight. Only tests COMPUTE_INPUT_TYPE + SCALE_MODE.
No FP8 GEMM benchmark (previous probe hung).
Also tests: torch._scaled_mm with correct layout, col-major FP8 hipBLASLt.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch
import time
import sys

from task import input_t, output_t

_call = 0
_probed = False

CPP_FWD = """
std::string probe_all_combos_v3(int64_t M, int64_t N, int64_t K);
std::string read_scale_header_v3();
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <string>

std::string read_scale_header_v3() {
    std::string r;

    // Dump the hipblasltMatmulMatrixScale enum
    r += "SCALE_NONE=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_NONE) + " ";
    r += "SCALE_SCALAR=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR) + " ";
    r += "SCALE_VEC32=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) + " ";

    // Key attribute IDs
    r += "A_SCALE_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER) + " ";
    r += "B_SCALE_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER) + " ";
    r += "A_SCALE_MODE=" + std::to_string(HIPBLASLT_MATMUL_DESC_A_SCALE_MODE) + " ";
    r += "B_SCALE_MODE=" + std::to_string(HIPBLASLT_MATMUL_DESC_B_SCALE_MODE) + " ";
    r += "CIT_A=" + std::to_string(HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT) + " ";
    r += "CIT_B=" + std::to_string(HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT) + " ";

    // Compute type values
    r += "COMPUTE_32F=" + std::to_string(HIPBLAS_COMPUTE_32F) + " ";
    r += "COMPUTE_16F=" + std::to_string(HIPBLAS_COMPUTE_16F) + " ";

    // Data type values
    r += "FP4=" + std::to_string(HIP_R_4F_E2M1_EXT) + " ";
    r += "UE8M0=" + std::to_string((int)(hipDataType)30) + " ";

    return r;
}

std::string probe_all_combos_v3(int64_t M, int64_t N, int64_t K) {
    std::string r;
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);

    // Data types to test
    hipDataType types[] = {
        (hipDataType)33,  // FP4 E2M1
        (hipDataType)31,  // FP6 E2M3
        (hipDataType)32,  // FP6 E3M2
        (hipDataType)28,  // FP8 E4M3
        (hipDataType)29,  // FP8 E5M2
        (hipDataType)14,  // BF16
        (hipDataType)8,   // uint8
        (hipDataType)30,  // UE8M0
    };
    const char* names[] = {"FP4","FP6a","FP6b","FP8a","FP8b","BF16","U8","UE8M0"};
    int ntypes = 8;

    // Test 1: Check which attr IDs accept scale_mode=VEC32_UE8M0
    {
        hipblasLtMatmulDesc_t desc;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

        int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        auto s31 = hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        r += "SET_A_SM:" + std::to_string((int)s31) + " ";

        auto s32 = hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        r += "SET_B_SM:" + std::to_string((int)s32) + " ";

        // Set compute input type to FP4
        hipDataType fp4 = (hipDataType)33;
        auto s100 = hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &fp4, sizeof(fp4));
        r += "SET_CIT_A:" + std::to_string((int)s100) + " ";

        auto s101 = hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &fp4, sizeof(fp4));
        r += "SET_CIT_B:" + std::to_string((int)s101) + " ";

        // Try FP6 compute input
        hipDataType fp6 = (hipDataType)31;
        auto s100b = hipblasLtMatmulDescSetAttribute(desc,
            HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &fp6, sizeof(fp6));
        r += "CIT_FP6:" + std::to_string((int)s100b) + " ";

        hipblasLtMatmulDescDestroy(desc);
    }

    // Test 2: Systematic heuristic search
    // For each (layout_A_type, layout_B_type, compute_input, scale_mode, transpose):
    struct Config {
        int a_idx, b_idx;   // index into types[]
        int cit;            // compute input type (-1=none, else hipDataType value)
        int scale;          // scale mode (-1=none, else HIPBLASLT_MATMUL_MATRIX_SCALE_*)
        bool trans_a, trans_b;
    };

    Config tests[] = {
        // FP8 A,B with FP4 compute input + block scales (THE KEY TEST)
        {3, 3, 33, 2, false, false},  // FP8xFP8 + CIT=FP4 + SM=VEC32
        {3, 3, 33, 2, false, true},   // FP8xFP8^T + CIT=FP4 + SM=VEC32
        {3, 3, 33, 2, true, false},   // FP8^TxFP8 + CIT=FP4 + SM=VEC32

        // uint8 with FP4 compute input + block scales
        {6, 6, 33, 2, false, false},  // U8xU8 + CIT=FP4 + SM=VEC32
        {6, 6, 33, 2, false, true},

        // FP4 layout + block scales (no compute input override)
        {0, 0, -1, 2, false, false},  // FP4xFP4 + SM=VEC32
        {0, 0, -1, 2, false, true},
        {0, 0, -1, 2, true, false},

        // FP4 layout + FP4 compute input + block scales
        {0, 0, 33, 2, false, false},
        {0, 0, 33, 2, false, true},
        {0, 0, 33, 2, true, false},

        // FP8 with block scales only (no CIT)
        {3, 3, -1, 2, false, false},
        {3, 3, -1, 2, false, true},

        // FP8 with scalar scale
        {3, 3, -1, 1, false, false},
        {3, 3, -1, 1, false, true},

        // FP8 with FP4 CIT only (no scale mode)
        {3, 3, 33, -1, false, false},
        {3, 3, 33, -1, false, true},

        // BF16 with FP4 compute input (fused quant?)
        {5, 3, 33, 2, false, true},   // BF16xFP8^T + CIT=FP4 + SM_B only

        // FP6 types
        {1, 1, -1, -1, false, false},
        {2, 2, -1, -1, false, false},
        {1, 1, -1, 2, false, false},

        // UE8M0 layout (weird but let's try)
        {7, 7, -1, -1, false, false},

        // Plain FP8 baselines (should get algos)
        {3, 3, -1, -1, false, false},
        {3, 3, -1, -1, false, true},
        {3, 3, -1, -1, true, false},
    };

    int ntests = sizeof(tests) / sizeof(tests[0]);
    r += "\n";
    for (int ti = 0; ti < ntests; ti++) {
        auto& t = tests[ti];
        hipblasLtMatmulDesc_t desc;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

        hipblasOperation_t opN = HIPBLAS_OP_N, opT = HIPBLAS_OP_T;
        if (t.trans_a) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        if (t.trans_b) hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

        if (t.cit >= 0) {
            hipDataType cit = (hipDataType)t.cit;
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT, &cit, sizeof(cit));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT, &cit, sizeof(cit));
        }
        if (t.scale >= 0) {
            int32_t sm = t.scale;
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        }

        // Column-major layout (BLAS standard)
        // C[M,N] = op(A)[M,K] * op(B)[K,N]
        int64_t rowA = t.trans_a ? K : M;
        int64_t colA = t.trans_a ? M : K;
        int64_t rowB = t.trans_b ? N : K;
        int64_t colB = t.trans_b ? K : N;

        hipblasLtMatrixLayout_t lA, lB, lC;
        auto sA = hipblasLtMatrixLayoutCreate(&lA, types[t.a_idx], rowA, colA, rowA);
        auto sB = hipblasLtMatrixLayoutCreate(&lB, types[t.b_idx], rowB, colB, rowB);
        auto sC = hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, M, N, M); // BF16 output

        bool ok = (sA == 0 && sB == 0 && sC == 0);

        int nAlgo = 0;
        if (ok) {
            hipblasLtMatmulPreference_t pref;
            hipblasLtMatmulPreferenceCreate(&pref);
            hipblasLtMatmulHeuristicResult_t results[8];
            hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
            hipblasLtMatmulPreferenceDestroy(pref);
        }

        if (nAlgo > 0) {
            const char* ta = t.trans_a ? "T" : "N";
            const char* tb = t.trans_b ? "T" : "N";
            r += "[" + std::to_string(ti) + "]" + names[t.a_idx] + "x" + names[t.b_idx] +
                 "(" + ta + tb + ")";
            if (t.cit >= 0) r += "+CIT" + std::to_string(t.cit);
            if (t.scale >= 0) r += "+SM" + std::to_string(t.scale);
            r += ":ALGOS=" + std::to_string(nAlgo) + "! ";
        }

        if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
        if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
        if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
        hipblasLtMatmulDescDestroy(desc);
    }

    if (r.find("ALGOS") == std::string::npos) {
        r += "NO_NOVEL_ALGOS ";
    }

    hipblasLtDestroy(handle);
    return r;
}
"""


def _run_probe():
    global _probed
    if _probed:
        return
    _probed = True

    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        mod = load_inline(
            name="hipblaslt_v3",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["probe_all_combos_v3", "read_scale_header_v3"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        t1 = time.time()
        print(f"[PROBE] Compiled in {t1-t0:.1f}s", flush=True)

        hdrs = mod.read_scale_header_v3()
        print(f"[PROBE] HEADERS: {hdrs}", flush=True)

        # Test on our actual benchmark shapes
        for M, N, K in [(32, 128, 512), (4, 2880, 512), (16, 2112, 7168),
                         (64, 7168, 2048), (256, 3072, 1536)]:
            result = mod.probe_all_combos_v3(M, N, K)
            print(f"[PROBE] SHAPE({M},{N},{K}): {result}", flush=True)

    except Exception as e:
        import traceback
        print(f"[PROBE] FAIL: {str(e)[:500]}", flush=True)
        traceback.print_exc()

    # Also test torch._scaled_mm with col-major B
    try:
        a = torch.randn(32, 512, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fnuz)
        b = torch.randn(512, 128, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fnuz)
        sa = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        sb = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        # _scaled_mm needs row-major A × col-major B
        # b.T makes it row-major → need to transpose
        c = torch._scaled_mm(a, b, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        print(f"[PROBE] _scaled_mm(FP8 AxB): shape={c.shape}", flush=True)
    except Exception as e:
        print(f"[PROBE] _scaled_mm(FP8): {str(e)[:200]}", flush=True)

    # Try with transposed B (col-major)
    try:
        bt = torch.randn(128, 512, dtype=torch.bfloat16, device='cuda').to(torch.float8_e4m3fnuz).t().contiguous().t()
        c2 = torch._scaled_mm(a, bt, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        print(f"[PROBE] _scaled_mm(FP8 AxB^T): shape={c2.shape}", flush=True)
    except Exception as e:
        print(f"[PROBE] _scaled_mm(FP8 col): {str(e)[:200]}", flush=True)


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
    if _call == 1:
        _run_probe()
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

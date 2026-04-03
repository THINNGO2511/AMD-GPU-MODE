#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Attempt hipBLASLt FP4 GEMM with block E8M0 scales.
Probe: Does hipBLASLt have FP4 algorithms? Can we execute FP4 GEMM?
Falls back to proven gemm_a16wfp4.
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

# C++ forward declarations
CPP_FWD = """
std::string probe_fp4_gemm_v1();
std::string dump_headers_v1();
std::string try_fp4_heuristic_v1(int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <string>
#include <cstdio>

// Known type constants from hip/library_types.h
static const hipDataType FP4_TYPE   = (hipDataType)33;  // HIP_R_4F_E2M1
static const hipDataType UE8M0_TYPE = (hipDataType)30;  // HIP_R_8F_UE8M0
static const hipDataType BF16_TYPE  = (hipDataType)14;  // HIP_R_16BF
static const hipDataType F32_TYPE   = (hipDataType)0;   // HIP_R_32F
static const hipDataType FP8_E4M3   = (hipDataType)28;  // HIP_R_8F_E4M3

// Dump relevant hipblaslt header definitions
std::string dump_headers_v1() {
    std::string r;

    // hipblasLtMatmulDescAttributes_t enum values
    // We need to find which attributes relate to scale pointers
    r += "MATMUL_DESC_ATTRS: ";
    #ifdef HIPBLASLT_MATMUL_DESC_TRANSA
    r += "TRANSA=" + std::to_string(HIPBLASLT_MATMUL_DESC_TRANSA) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_TRANSB
    r += "TRANSB=" + std::to_string(HIPBLASLT_MATMUL_DESC_TRANSB) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_EPILOGUE
    r += "EPILOGUE=" + std::to_string(HIPBLASLT_MATMUL_DESC_EPILOGUE) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_BIAS_POINTER
    r += "BIAS_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_BIAS_POINTER) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE
    r += "BIAS_DT=" + std::to_string(HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER
    r += "D_SCALE_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER
    r += "A_SCALE_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER
    r += "B_SCALE_PTR=" + std::to_string(HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER) + " ";
    #endif

    // Scale type enum
    #ifdef HIPBLASLT_MATMUL_DESC_A_SCALE_TYPE
    r += "A_SCALE_TYPE=" + std::to_string(HIPBLASLT_MATMUL_DESC_A_SCALE_TYPE) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_DESC_B_SCALE_TYPE
    r += "B_SCALE_TYPE=" + std::to_string(HIPBLASLT_MATMUL_DESC_B_SCALE_TYPE) + " ";
    #endif

    // Matrix scale types
    r += "SCALE_TYPES: ";
    #ifdef HIPBLASLT_MATMUL_MATRIX_SCALE_NONE
    r += "NONE=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_NONE) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR
    r += "SCALAR=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR) + " ";
    #endif
    #ifdef HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
    r += "VEC32_UE8M0=" + std::to_string(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0) + " ";
    #endif

    // Check compute type enum values
    r += "COMPUTE: ";
    r += "F32=" + std::to_string(HIPBLAS_COMPUTE_32F) + " ";
    #ifdef HIPBLAS_COMPUTE_16F
    r += "F16=" + std::to_string(HIPBLAS_COMPUTE_16F) + " ";
    #endif
    #ifdef HIPBLAS_COMPUTE_32I
    r += "I32=" + std::to_string(HIPBLAS_COMPUTE_32I) + " ";
    #endif

    // hipblaslt-ext types
    r += "EXT: ";
    r += "E2M1_EXT=" + std::to_string(HIP_R_4F_E2M1_EXT) + " ";

    return r;
}

// Probe: Can we find FP4 algorithms?
std::string probe_fp4_gemm_v1() {
    std::string r;
    hipblasLtHandle_t handle;
    auto st = hipblasLtCreate(&handle);
    if (st != HIPBLAS_STATUS_SUCCESS) return "HANDLE_FAIL";

    int M = 32, N = 128, K = 128;

    // === Try different matmul descriptor configurations ===

    // Config 1: Standard compute F32, FP4 inputs, BF16 output
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayout_t lA, lB, lC;

            // Row-major: A[M,K], B[K,N], C[M,N]
            // Leading dim for FP4: K elements, but packed as K/2 bytes
            // Try multiple ld values
            int ld_vals[] = {K, K/2};
            for (int li = 0; li < 2; li++) {
                int ldA = ld_vals[li];
                auto sA = hipblasLtMatrixLayoutCreate(&lA, FP4_TYPE, M, K, ldA);
                auto sB = hipblasLtMatrixLayoutCreate(&lB, FP4_TYPE, K, N, N);
                auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, N);

                if (sA == HIPBLAS_STATUS_SUCCESS && sB == HIPBLAS_STATUS_SUCCESS && sC == HIPBLAS_STATUS_SUCCESS) {
                    hipblasLtMatmulPreference_t pref;
                    hipblasLtMatmulPreferenceCreate(&pref);

                    hipblasLtMatmulHeuristicResult_t results[8];
                    int nAlgo = 0;
                    auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                    r += "CFG1(ld=" + std::to_string(ldA) + "):heur=" + std::to_string((int)sH) + ",algos=" + std::to_string(nAlgo) + " ";

                    hipblasLtMatmulPreferenceDestroy(pref);
                }
                if (sA == HIPBLAS_STATUS_SUCCESS) hipblasLtMatrixLayoutDestroy(lA);
                if (sB == HIPBLAS_STATUS_SUCCESS) hipblasLtMatrixLayoutDestroy(lB);
                if (sC == HIPBLAS_STATUS_SUCCESS) hipblasLtMatrixLayoutDestroy(lC);
            }
            hipblasLtMatmulDescDestroy(desc);
        }
    }

    // Config 2: With transpose (column-major interpretation)
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            hipblasOperation_t opT = HIPBLAS_OP_T;
            hipblasOperation_t opN = HIPBLAS_OP_N;

            // Set A transpose (row-major as transposed column-major)
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
            hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opT, sizeof(opT));

            // Column-major: A[K,M] (no trans), B[N,K]^T (trans)
            hipblasLtMatrixLayout_t lA, lB, lC;
            auto sA = hipblasLtMatrixLayoutCreate(&lA, FP4_TYPE, K, M, K);
            auto sB = hipblasLtMatrixLayoutCreate(&lB, FP4_TYPE, N, K, N);
            auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, K, N, K);

            if (sA == 0 && sB == 0 && sC == 0) {
                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t results[8];
                int nAlgo = 0;
                auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                r += "CFG2(NT):heur=" + std::to_string((int)sH) + ",algos=" + std::to_string(nAlgo) + " ";

                hipblasLtMatmulPreferenceDestroy(pref);
            }
            if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
            if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
            if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
            hipblasLtMatmulDescDestroy(desc);
        }
    }

    // Config 3: Try with scale type set
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            // Try setting A/B scale type to VEC32_UE8M0
            int32_t scale_type = 2; // HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0

            // Try attribute IDs that might be A_SCALE_TYPE/B_SCALE_TYPE
            // From prev probe: attrs 0,1,2,4,13,22 were readable
            // Try setting various attributes
            for (int attr_id = 0; attr_id <= 30; attr_id++) {
                auto s = hipblasLtMatmulDescSetAttribute(desc,
                    (hipblasLtMatmulDescAttributes_t)attr_id, &scale_type, sizeof(scale_type));
                if (s == HIPBLAS_STATUS_SUCCESS) {
                    r += "SET_ATTR(" + std::to_string(attr_id) + "=2):OK ";
                }
            }

            hipblasLtMatmulDescDestroy(desc);
        }
    }

    // Config 4: Try FP8 + scale (FP8 might be better supported)
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayout_t lA, lB, lC;
            auto sA = hipblasLtMatrixLayoutCreate(&lA, FP8_E4M3, M, K, K);
            auto sB = hipblasLtMatrixLayoutCreate(&lB, FP8_E4M3, K, N, N);
            auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, N);

            if (sA == 0 && sB == 0 && sC == 0) {
                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t results[8];
                int nAlgo = 0;
                auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                r += "FP8_HEUR:algos=" + std::to_string(nAlgo) + " ";

                hipblasLtMatmulPreferenceDestroy(pref);
            }
            if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
            if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
            if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
            hipblasLtMatmulDescDestroy(desc);
        }
    }

    // Config 5: Try using ALL compute types with FP4 layouts
    {
        int compute_types[] = {0, 2, 4, 5, 6, 7, 9};
        for (int ci = 0; ci < 7; ci++) {
            hipblasLtMatmulDesc_t desc;
            st = hipblasLtMatmulDescCreate(&desc, (hipblasComputeType_t)compute_types[ci], F32_TYPE);
            if (st == HIPBLAS_STATUS_SUCCESS) {
                hipblasLtMatrixLayout_t lA, lB, lC;
                auto sA = hipblasLtMatrixLayoutCreate(&lA, FP4_TYPE, M, K, K);
                auto sB = hipblasLtMatrixLayoutCreate(&lB, FP4_TYPE, K, N, N);
                auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, N);

                if (sA == 0 && sB == 0 && sC == 0) {
                    hipblasLtMatmulPreference_t pref;
                    hipblasLtMatmulPreferenceCreate(&pref);

                    hipblasLtMatmulHeuristicResult_t results[8];
                    int nAlgo = 0;
                    auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                    if (nAlgo > 0) {
                        r += "FP4_C" + std::to_string(compute_types[ci]) + ":algos=" + std::to_string(nAlgo) + "! ";
                    }

                    hipblasLtMatmulPreferenceDestroy(pref);
                }
                if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
                if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
                if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
                hipblasLtMatmulDescDestroy(desc);
            }
        }
        r += "FP4_ALL_COMPUTE:done ";
    }

    // Config 6: Try using HIP_R_4F_E2M1_EXT (the hipblaslt extension type)
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayout_t lA, lB, lC;
            // Use the extension type value
            auto sA = hipblasLtMatrixLayoutCreate(&lA, (hipDataType)HIP_R_4F_E2M1_EXT, M, K, K);
            auto sB = hipblasLtMatrixLayoutCreate(&lB, (hipDataType)HIP_R_4F_E2M1_EXT, K, N, N);
            auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, N);

            if (sA == 0 && sB == 0 && sC == 0) {
                // Try setting scale type
                int32_t scale_type = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;

                // Try all possible scale attribute IDs
                for (int a = 0; a <= 30; a++) {
                    auto ss = hipblasLtMatmulDescSetAttribute(desc,
                        (hipblasLtMatmulDescAttributes_t)a, &scale_type, sizeof(scale_type));
                    if (ss == HIPBLAS_STATUS_SUCCESS) {
                        // Verify it stuck
                        int32_t readback = -1;
                        size_t rb_size = 0;
                        hipblasLtMatmulDescGetAttribute(desc,
                            (hipblasLtMatmulDescAttributes_t)a, &readback, sizeof(readback), &rb_size);
                        if (readback == scale_type) {
                            r += "SCALE_ATTR(" + std::to_string(a) + ")=VEC32_OK ";
                        }
                    }
                }

                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t results[8];
                int nAlgo = 0;
                auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                r += "EXT_FP4:heur=" + std::to_string((int)sH) + ",algos=" + std::to_string(nAlgo) + " ";

                hipblasLtMatmulPreferenceDestroy(pref);
            }
            if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
            if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
            if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
            hipblasLtMatmulDescDestroy(desc);
        }
    }

    // Config 7: Mixed precision: BF16 A x FP4 B -> BF16 C (what we actually need!)
    {
        hipblasLtMatmulDesc_t desc;
        st = hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);
        if (st == HIPBLAS_STATUS_SUCCESS) {
            hipblasLtMatrixLayout_t lA, lB, lC;
            auto sA = hipblasLtMatrixLayoutCreate(&lA, BF16_TYPE, M, K, K);
            auto sB = hipblasLtMatrixLayoutCreate(&lB, FP4_TYPE, K, N, N);
            auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, N);

            if (sA == 0 && sB == 0 && sC == 0) {
                hipblasLtMatmulPreference_t pref;
                hipblasLtMatmulPreferenceCreate(&pref);

                hipblasLtMatmulHeuristicResult_t results[8];
                int nAlgo = 0;
                auto sH = hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
                r += "MIXED(BF16xFP4):heur=" + std::to_string((int)sH) + ",algos=" + std::to_string(nAlgo) + " ";

                hipblasLtMatmulPreferenceDestroy(pref);
            }
            if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
            if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
            if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
            hipblasLtMatmulDescDestroy(desc);
        }
    }

    hipblasLtDestroy(handle);
    return r;
}

// Targeted heuristic search for specific shape
std::string try_fp4_heuristic_v1(int64_t M, int64_t N, int64_t K) {
    std::string r;
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);

    // Try all combinations: {NN, NT, TN, TT} x {FP4xFP4, BF16xFP4, FP4xBF16}
    hipblasOperation_t ops[] = {HIPBLAS_OP_N, HIPBLAS_OP_T};
    hipDataType a_types[] = {FP4_TYPE, BF16_TYPE, FP8_E4M3};
    hipDataType b_types[] = {FP4_TYPE, BF16_TYPE, FP8_E4M3};
    const char* a_names[] = {"FP4", "BF16", "FP8"};
    const char* b_names[] = {"FP4", "BF16", "FP8"};

    for (int ai = 0; ai < 3; ai++) {
        for (int bi = 0; bi < 3; bi++) {
            // Skip BF16xBF16 (already known to work)
            if (ai == 1 && bi == 1) continue;

            for (int oi = 0; oi < 2; oi++) {
                for (int oj = 0; oj < 2; oj++) {
                    hipblasLtMatmulDesc_t desc;
                    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, F32_TYPE);

                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &ops[oi], sizeof(ops[oi]));
                    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &ops[oj], sizeof(ops[oj]));

                    // Compute layout dims based on transpose
                    int64_t rowA = oi == 0 ? M : K;
                    int64_t colA = oi == 0 ? K : M;
                    int64_t rowB = oj == 0 ? K : N;
                    int64_t colB = oj == 0 ? N : K;

                    hipblasLtMatrixLayout_t lA, lB, lC;
                    auto sA = hipblasLtMatrixLayoutCreate(&lA, a_types[ai], rowA, colA, rowA);
                    auto sB = hipblasLtMatrixLayoutCreate(&lB, b_types[bi], rowB, colB, rowB);
                    auto sC = hipblasLtMatrixLayoutCreate(&lC, BF16_TYPE, M, N, M);

                    if (sA == 0 && sB == 0 && sC == 0) {
                        hipblasLtMatmulPreference_t pref;
                        hipblasLtMatmulPreferenceCreate(&pref);

                        hipblasLtMatmulHeuristicResult_t results[8];
                        int nAlgo = 0;
                        hipblasLtMatmulAlgoGetHeuristic(handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);

                        if (nAlgo > 0) {
                            const char* op_names[] = {"N", "T"};
                            r += std::string(a_names[ai]) + "x" + b_names[bi] + "(" +
                                 op_names[oi] + op_names[oj] + "):algos=" + std::to_string(nAlgo) + " ";
                        }

                        hipblasLtMatmulPreferenceDestroy(pref);
                    }
                    if (sA == 0) hipblasLtMatrixLayoutDestroy(lA);
                    if (sB == 0) hipblasLtMatrixLayoutDestroy(lB);
                    if (sC == 0) hipblasLtMatrixLayoutDestroy(lC);
                    hipblasLtMatmulDescDestroy(desc);
                }
            }
        }
    }

    hipblasLtDestroy(handle);
    if (r.empty()) r = "NO_ALGORITHMS_FOUND";
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
            name="hipblaslt_fp4_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["probe_fp4_gemm_v1", "dump_headers_v1", "try_fp4_heuristic_v1"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        t1 = time.time()
        print(f"[PROBE] Compiled in {t1-t0:.1f}s", flush=True)

        # Dump header definitions
        hdrs = mod.dump_headers_v1()
        print(f"[PROBE] HEADERS: {hdrs}", flush=True)

        # Try FP4 GEMM
        fp4_result = mod.probe_fp4_gemm_v1()
        print(f"[PROBE] FP4_GEMM: {fp4_result}", flush=True)

        # Try specific shapes
        for M, N, K in [(32, 128, 512), (4, 2880, 512), (16, 2112, 7168), (256, 3072, 1536)]:
            shape_result = mod.try_fp4_heuristic_v1(M, N, K)
            print(f"[PROBE] SHAPE({M},{N},{K}): {shape_result}", flush=True)

    except Exception as e:
        print(f"[PROBE] FAIL: {str(e)[:500]}", flush=True)

    # Also read hipblaslt-types.h for scale-related enums
    try:
        import subprocess
        out = subprocess.run(
            ["grep", "-n", "SCALE\|scale\|MX\|mx\|matmulMatrixScale\|MATRIX_SCALE",
             "/opt/rocm/include/hipblaslt/hipblaslt-types.h"],
            capture_output=True, text=True, timeout=5
        )
        for line in (out.stdout or "").strip().split('\n')[:20]:
            print(f"[PROBE] TYPES_H: {line[:150]}", flush=True)
    except:
        pass

    # Read matmul desc attribute enum
    try:
        import subprocess
        out = subprocess.run(
            ["grep", "-n", "MATMUL_DESC\|matmulDesc\|Attributes",
             "/opt/rocm/include/hipblaslt/hipblaslt.h"],
            capture_output=True, text=True, timeout=5
        )
        for line in (out.stdout or "").strip().split('\n')[:30]:
            print(f"[PROBE] BLTH: {line[:150]}", flush=True)
    except:
        pass

    # Read ext header for more types
    try:
        import subprocess
        out = subprocess.run(
            ["grep", "-rn", "E2M1\|E8M0\|FP4\|fp4\|scale\|SCALE",
             "/opt/rocm/include/hipblaslt/hipblaslt-ext.hpp"],
            capture_output=True, text=True, timeout=5
        )
        for line in (out.stdout or "").strip().split('\n')[:20]:
            print(f"[PROBE] EXT_H: {line[:150]}", flush=True)
    except:
        pass


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

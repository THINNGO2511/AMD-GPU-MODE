#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 for M>=32, Triton fallback for M<32.
Key insight: row-major FP4[M,K] = col-major FP4(K,M) — no transpose needed!
hipBLASLt: D_cm(N,M) = B_fp4^T(N,K) * A_fp4(K,M) → row-major D[M,N] = C[M,N]
Swap A/B in the call: matA=B_q(opA=T), matB=A_fp4(opB=N)
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time
from task import input_t, output_t

_call = 0
_hbl_mod = None
_hbl_ok = False

CPP_FWD = """
torch::Tensor hipblaslt_fp4_gemm_v1(
    torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K, int64_t algo_idx);
std::string hipblaslt_probe_v1(int64_t M, int64_t N, int64_t K);
"""

# Use 0 for hipblasLtMatmul last arg (= default exec context)
HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>

static hipblasLtHandle_t g_handle = nullptr;

static void ensure_handle() {
    if (!g_handle) hipblasLtCreate(&g_handle);
}

std::string hipblaslt_probe_v1(int64_t M, int64_t N, int64_t K) {
    ensure_handle();
    std::string r;

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));

    void *dA=nullptr, *dB=nullptr;
    hipMalloc(&dA, 1024); hipMalloc(&dB, 1024);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dB, sizeof(void*));

    // SWAPPED: matA=B(K,N) opA=T, matB=A(K,M) opB=N → D(N,M) = B^T*A
    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, (hipDataType)33, K, N, K);  // "A"=B: (K,N)
    hipblasLtMatrixLayoutCreate(&lB, (hipDataType)33, K, M, K);  // "B"=A: (K,M)
    hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, N, M, N);  // D: (N,M)

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);
    r += "ALGOS=" + std::to_string(nAlgo);

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
    hipFree(dA); hipFree(dB);
    return r;
}

// Full FP4 GEMM: C[M,N] = A_fp4[M,K] * B_fp4[N,K]^T with block E8M0 scales
// A_fp4: uint8 [M, K/2] row-major (= col-major FP4(K,M))
// B_fp4: uint8 [N, K/2] row-major (= col-major FP4(K,N))
// A_scale: uint8 [M, K/32] row-major (= col-major E8M0(K/32,M))
// B_scale: uint8 [N, K/32] row-major (= col-major E8M0(K/32,N))
torch::Tensor hipblaslt_fp4_gemm_v1(
    torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K, int64_t algo_idx)
{
    ensure_handle();

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));

    // SWAPPED: hipBLASLt "A" = our B, hipBLASLt "B" = our A
    void* pBscale = B_scale.data_ptr();  // goes to HIPBLASLT A_SCALE (for B_fp4)
    void* pAscale = A_scale.data_ptr();  // goes to HIPBLASLT B_SCALE (for A_fp4)
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pBscale, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pAscale, sizeof(void*));

    // Layouts: matA=B(K,N), matB=A(K,M), matC=D(N,M)
    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, (hipDataType)33, K, N, K);
    hipblasLtMatrixLayoutCreate(&lB, (hipDataType)33, K, M, K);
    hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, N, M, N);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_handle, desc, lA, lB, lC, lC, pref, 8, results, &nAlgo);

    // Output: [M, N] row-major bf16 = col-major (N, M) bf16
    auto C = torch::empty({M, N}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

    if (nAlgo > 0) {
        int ai = (algo_idx < nAlgo) ? algo_idx : 0;
        void* workspace = nullptr;
        if (results[ai].workspaceSize > 0) hipMalloc(&workspace, results[ai].workspaceSize);

        float alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmul(g_handle, desc, &alpha,
            B_fp4.data_ptr(), lA,   // "A" in BLAS = our B
            A_fp4.data_ptr(), lB,   // "B" in BLAS = our A
            &beta,
            C.data_ptr(), lC,
            C.data_ptr(), lC,
            &results[ai].algo, workspace, results[ai].workspaceSize, 0);

        if (workspace) hipFree(workspace);
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
    return C;
}
"""

def _compile():
    global _hbl_mod, _hbl_ok
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _hbl_mod = load_inline(
            name="hbl_fp4_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["hipblaslt_fp4_gemm_v1", "hipblaslt_probe_v1"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        print(f"[HBL] Compiled in {time.time()-t0:.1f}s", flush=True)

        # Verify algorithms exist for our shapes
        for M, N, K in [(32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536)]:
            r = _hbl_mod.hipblaslt_probe_v1(M, N, K)
            print(f"[HBL] probe({M},{N},{K}): {r}", flush=True)

        # Quick accuracy test: quant A, call hipBLASLt, compare to Triton
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        M, N, K = 32, 128, 512
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B_fp4_test = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
        B_scale_test = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')

        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        A_fp4_u8 = A_fp4.view(torch.uint8)
        A_scale_u8 = A_scale.view(torch.uint8)

        # Test 1: Normal order
        C_hbl = _hbl_mod.hipblaslt_fp4_gemm_v1(
            A_fp4_u8, B_fp4_test, A_scale_u8, B_scale_test, M, N, K, 0)
        C_ref = gemm_afp4wfp4(A_fp4_u8, B_fp4_test, A_scale_u8, B_scale_test, dtype=torch.bfloat16)
        d1 = (C_hbl - C_ref).abs().max().item()
        print(f"[HBL] Normal: maxdiff={d1:.2f} hbl=[{C_hbl.min().item():.1f},{C_hbl.max().item():.1f}] ref=[{C_ref.min().item():.1f},{C_ref.max().item():.1f}]", flush=True)

        # Test 2: Nibble-swapped A
        A_swapped = ((A_fp4_u8 >> 4) | (A_fp4_u8 << 4)).to(torch.uint8)
        C_hbl2 = _hbl_mod.hipblaslt_fp4_gemm_v1(
            A_swapped, B_fp4_test, A_scale_u8, B_scale_test, M, N, K, 0)
        d2 = (C_hbl2 - C_ref).abs().max().item()
        print(f"[HBL] SwapA: maxdiff={d2:.2f}", flush=True)

        # Test 3: Nibble-swapped B
        B_swapped = ((B_fp4_test >> 4) | (B_fp4_test << 4)).to(torch.uint8)
        C_hbl3 = _hbl_mod.hipblaslt_fp4_gemm_v1(
            A_fp4_u8, B_swapped, A_scale_u8, B_scale_test, M, N, K, 0)
        d3 = (C_hbl3 - C_ref).abs().max().item()
        print(f"[HBL] SwapB: maxdiff={d3:.2f}", flush=True)

        # Test 4: Both swapped
        C_hbl4 = _hbl_mod.hipblaslt_fp4_gemm_v1(
            A_swapped, B_swapped, A_scale_u8, B_scale_test, M, N, K, 0)
        d4 = (C_hbl4 - C_ref).abs().max().item()
        print(f"[HBL] SwapBoth: maxdiff={d4:.2f}", flush=True)

        # Test 5: Transposed output (maybe col/row major confusion)
        C_hbl5_T = C_hbl[:N, :M] if C_hbl.shape[0] >= N else C_hbl
        # Actually check if the transpose matches
        if C_hbl.shape == C_ref.shape:
            d5 = (C_hbl.T[:N,:M] - C_ref[:N,:M]).abs().max().item() if min(C_hbl.shape) >= min(M,N) else 999
        else:
            d5 = 999
        print(f"[HBL] TransCheck: {d5:.2f}", flush=True)

        best = min(d1, d2, d3, d4)
        best_name = ["Normal", "SwapA", "SwapB", "SwapBoth"][[d1,d2,d3,d4].index(best)]
        print(f"[HBL] BEST: {best_name} maxdiff={best:.2f}", flush=True)

        if best < 1.0:
            _hbl_ok = True
            print(f"[HBL] *** ACCURACY OK ({best_name}) — USING hipBLASLt ***", flush=True)
        else:
            print("[HBL] Accuracy too low, debugging data format", flush=True)

    except Exception as e:
        import traceback
        print(f"[HBL] FAIL: {str(e)[:300]}", flush=True)
        traceback.print_exc()

# ===== Proven Triton GEMM =====
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
        _compile()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    global _bsr, _bqu, _braw

    # Cache B processing
    if _bsr is not B_scale_sh:
        _bsr = B_scale_sh
        su = B_scale_sh.view(torch.uint8); sm, sn = su.shape
        if (sm, sn) not in _gc:
            _gc[(sm, sn)] = _bgc(sm, sn, su.device)
        _braw = _fu(su.reshape(-1), sm, sn)
        _bqu = B_q.view(torch.uint8)

    # Try hipBLASLt for M>=32
    if _hbl_ok and m >= 32:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_fp4_u8 = A_fp4.view(torch.uint8)
            A_scale_u8 = A_scale.view(torch.uint8)

            # B_scale needs to be UNSHUFFLED for hipBLASLt
            C = _hbl_mod.hipblaslt_fp4_gemm_v1(
                A_fp4_u8, _bqu, A_scale_u8, _braw,
                m, n, k, 0)
            return C
        except Exception as e:
            pass  # Fall through to Triton

    # Triton fallback
    _pw(A.device)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _yc:
        _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _yc[key]
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bqu, asc, _braw, dtype=torch.bfloat16)
    cfg = _K7168 if k == 7168 else (_K2048 if k == 2048 else _K512)
    gemm_a16wfp4(A, _bqu, _braw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

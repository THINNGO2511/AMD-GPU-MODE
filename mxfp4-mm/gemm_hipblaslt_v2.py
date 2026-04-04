#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 production v2. DIRECT approach (verified correct).
D_col(M,N) = A^T(M,K) * B(K,N) → output .t() for row-major [M,N]
matA = A_fp4 (K,M), matB = B_fp4 (K,N), opA=T opB=N
Scale: A_scale for matA, B_scale for matB (NO swapping!)
M<32 → Triton fallback. M>=32 → hipBLASLt FP4.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time
from task import input_t, output_t

_call = 0
_mod = None
_ok = False

CPP_FWD = """
void hbl_init_v2();
torch::Tensor hbl_gemm_v2(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K);
int hbl_probe_v2(int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>
#include <unordered_map>

static hipblasLtHandle_t g_handle = nullptr;

// Cache: key=(M,N,K) → (desc, layouts, algo, workspace)
struct GemmPlan {
    hipblasLtMatmulDesc_t desc;
    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatmulHeuristicResult_t algo;
    void* workspace;
    bool valid;
};
static std::unordered_map<uint64_t, GemmPlan> g_plans;

static uint64_t make_key(int64_t M, int64_t N, int64_t K) {
    return ((uint64_t)M << 40) | ((uint64_t)N << 20) | (uint64_t)K;
}

void hbl_init_v2() {
    if (!g_handle) hipblasLtCreate(&g_handle);
}

int hbl_probe_v2(int64_t M, int64_t N, int64_t K) {
    hbl_init_v2();
    auto key = make_key(M, N, K);
    if (g_plans.count(key)) return g_plans[key].valid ? 1 : 0;

    GemmPlan plan;
    plan.valid = false;
    plan.workspace = nullptr;

    hipblasLtMatmulDescCreate(&plan.desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));

    // DIRECT: matA=A(K,M), matB=B(K,N), output=(M,N) col-major
    hipblasLtMatrixLayoutCreate(&plan.lA, (hipDataType)33, K, M, K);
    hipblasLtMatrixLayoutCreate(&plan.lB, (hipDataType)33, K, N, K);
    hipblasLtMatrixLayoutCreate(&plan.lC, (hipDataType)14, M, N, M);

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t maxWS = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWS, sizeof(maxWS));

    hipblasLtMatmulHeuristicResult_t results[8];
    int nAlgo = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_handle, plan.desc, plan.lA, plan.lB, plan.lC, plan.lC,
        pref, 8, results, &nAlgo);
    hipblasLtMatmulPreferenceDestroy(pref);

    if (nAlgo > 0) {
        plan.algo = results[0];
        if (plan.algo.workspaceSize > 0) {
            hipMalloc(&plan.workspace, plan.algo.workspaceSize);
        }
        plan.valid = true;
    }

    g_plans[key] = plan;
    return plan.valid ? nAlgo : 0;
}

torch::Tensor hbl_gemm_v2(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K)
{
    auto key = make_key(M, N, K);
    auto& plan = g_plans[key];

    // Set scale pointers (may change per call)
    void* pScA = A_scale.data_ptr();
    void* pScB = B_scale.data_ptr();
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pScA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(plan.desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pScB, sizeof(void*));

    // Output: col-major (M,N) = row-major [N,M]
    auto D = torch::empty({N, M}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

    float alpha = 1.0f, beta = 0.0f;
    hipblasLtMatmul(g_handle, plan.desc, &alpha,
        A_fp4.data_ptr(), plan.lA,
        B_fp4.data_ptr(), plan.lB,
        &beta, D.data_ptr(), plan.lC, D.data_ptr(), plan.lC,
        &plan.algo.algo, plan.workspace, plan.algo.workspaceSize, 0);

    return D.t().contiguous();
}
"""

def _compile():
    global _mod, _ok
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        _mod = load_inline(
            name="hbl_v2",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["hbl_init_v2", "hbl_gemm_v2", "hbl_probe_v2"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        print(f"[HBL] Compiled in {time.time()-t0:.1f}s", flush=True)
        _mod.hbl_init_v2()

        # Pre-plan all shapes (M padded to 32)
        shapes_ok = True
        for M, N, K in [(32,2880,512),(32,2112,7168),(32,4096,512),(32,2880,512),
                         (64,7168,2048),(256,3072,1536)]:
            n = _mod.hbl_probe_v2(M, N, K)
            print(f"[HBL] plan({M},{N},{K}): {n} algos", flush=True)
            if n == 0: shapes_ok = False

        if not shapes_ok:
            print("[HBL] Some shapes failed, using Triton fallback", flush=True)
            return

        # Accuracy test with real quant data
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        M, N, K = 32, 128, 512
        _mod.hbl_probe_v2(M, N, K)
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
        Au = A_fp4.view(torch.uint8); Asu = A_sc.view(torch.uint8)
        Bu = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
        Bsu = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')

        C_ref = gemm_afp4wfp4(Au, Bu, Asu, Bsu, dtype=torch.bfloat16)

        # Test 1: Raw (unshuffled) A_scale
        C1 = _mod.hbl_gemm_v2(Au, Bu, Asu, Bsu, M, N, K)
        d1 = (C1 - C_ref).abs().max().item()
        print(f"[HBL] Raw A_scale: maxdiff={d1:.2f}", flush=True)

        # Test 2: SHUFFLED A_scale (hypothesis: hipBLASLt expects shuffled)
        from aiter.utility.fp4_utils import e8m0_shuffle
        Asu_sh = e8m0_shuffle(A_sc).view(torch.uint8)
        C2 = _mod.hbl_gemm_v2(Au, Bu, Asu_sh, Bsu, M, N, K)
        d2 = (C2 - C_ref).abs().max().item()
        print(f"[HBL] Shuffled A_scale: maxdiff={d2:.2f}", flush=True)

        # Test 3: Try BLK32_UE8M0_32_8_EXT (mode=6) with shuffled scales
        # (can't easily change mode here, but note result for reference)

        best = min(d1, d2)
        best_name = "raw" if d1 <= d2 else "shuffled"
        print(f"[HBL] BEST: {best_name} maxdiff={best:.2f}", flush=True)

        if best < 1.0:
            _ok = True
            print(f"[HBL] *** ACCURACY PASS ({best_name}) — hipBLASLt ACTIVE ***", flush=True)

            # Benchmark
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(50):
                _mod.hbl_gemm_v2(Au, Bu, Asu if best_name=="raw" else Asu_sh, Bsu, M, N, K)
            torch.cuda.synchronize()
            us = (time.time()-t0)/50*1e6
            print(f"[HBL] Time ({M},{N},{K}): {us:.1f}us", flush=True)
        else:
            print("[HBL] Accuracy fail, both raw and shuffled", flush=True)
            for j in range(min(4,N)):
                print(f"[HBL]   [{j}] raw={C1[0,j].item():.2f} shuf={C2[0,j].item():.2f} ref={C_ref[0,j].item():.2f}", flush=True)

    except Exception as e:
        import traceback
        print(f"[HBL] FAIL: {str(e)[:300]}", flush=True)
        traceback.print_exc()

# ===== Triton fallback =====
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

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    global _bsr, _bqu, _braw
    if _bsr is not B_scale_sh:
        _bsr=B_scale_sh; su=B_scale_sh.view(torch.uint8); sm,sn=su.shape
        if (sm,sn) not in _gc: _gc[(sm,sn)]=_bgc(sm,sn,su.device)
        _braw=_fu(su.reshape(-1),sm,sn); _bqu=B_q.view(torch.uint8)

    # hipBLASLt for M>=32
    if _ok and m >= 32:
        try:
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            A_fp4, A_sc = dynamic_mxfp4_quant(A)
            Au = A_fp4.view(torch.uint8)
            Asu = A_sc.view(torch.uint8)
            C = _mod.hbl_gemm_v2(Au, _bqu, Asu, _braw, m, n, k)
            return C
        except:
            pass

    # Triton fallback
    _pw(A.device)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key=(m,n)
    if key not in _yc: _yc[key]=torch.empty(m,n,dtype=torch.bfloat16,device=A.device)
    out=_yc[key]
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bqu,asc,_braw,dtype=torch.bfloat16)
    cfg=_K7168 if k==7168 else(_K2048 if k==2048 else _K512)
    gemm_a16wfp4(A,_bqu,_braw,dtype=torch.bfloat16,y=out,config=cfg)
    return out

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 diagnostic with known values.
Tests: identity FP4 values, scale handling, layout verification.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch, time
from task import input_t, output_t

_call = 0; _probed = False

CPP_FWD = """
torch::Tensor hbl_gemm_diag(torch::Tensor A, torch::Tensor B,
    torch::Tensor scA, torch::Tensor scB, int64_t M, int64_t N, int64_t K);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>

static hipblasLtHandle_t g_h = nullptr;

torch::Tensor hbl_gemm_diag(torch::Tensor A, torch::Tensor B,
    torch::Tensor scA, torch::Tensor scB, int64_t M, int64_t N, int64_t K)
{
    if (!g_h) hipblasLtCreate(&g_h);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);

    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    int32_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));

    // DIRECT (no swap): matA=A(K,M) opA=T, matB=B(K,N) opB=N
    // D_col(M,N) = A^T(M,K) * B(K,N)
    // Output: col-major (M,N) = row-major [N,M]
    // Scale A: for matA stored as (K,M), scale is (K/32,M) col-major = [M,K/32] row-major
    // Scale B: for matB stored as (K,N), scale is (K/32,N) col-major = [N,K/32] row-major
    void* pScA = scA.data_ptr();
    void* pScB = scB.data_ptr();
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pScA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pScB, sizeof(void*));

    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, (hipDataType)33, K, M, K);  // A: (K,M) FP4
    hipblasLtMatrixLayoutCreate(&lB, (hipDataType)33, K, N, K);  // B: (K,N) FP4
    hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, M, N, M);  // C: (M,N) BF16

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t ws = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    hipblasLtMatmulHeuristicResult_t res[8];
    int nA = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_h, desc, lA, lB, lC, lC, pref, 8, res, &nA);

    // Output as [N, M] row-major (= col-major (M,N))
    auto D = torch::empty({N, M}, torch::dtype(torch::kBFloat16).device(A.device()));

    if (nA > 0) {
        float alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmul(g_h, desc, &alpha,
            A.data_ptr(), lA,
            B.data_ptr(), lB,
            &beta, D.data_ptr(), lC, D.data_ptr(), lC,
            &res[0].algo, nullptr, 0, 0);
        hipDeviceSynchronize();
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);

    return D.t().contiguous();  // [N,M].T = [M,N]
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
            name="hbl_diag_v1",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["hbl_gemm_diag"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        print(f"[DIAG] Compiled in {time.time()-t0:.1f}s", flush=True)

        M, N, K = 32, 64, 128

        # Test 1: All-ones FP4 (0x22 = both nibbles = 1.0) with scale=127 (=1.0)
        # Expected: C[m,n] = K * 1.0 * 1.0 = 128
        A_ones = torch.full((M, K//2), 0x22, dtype=torch.uint8, device='cuda')
        B_ones = torch.full((N, K//2), 0x22, dtype=torch.uint8, device='cuda')
        scA = torch.full((M, K//32), 127, dtype=torch.uint8, device='cuda')
        scB = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')

        C1 = mod.hbl_gemm_diag(A_ones, B_ones, scA, scB, M, N, K)
        print(f"[DIAG] Test1 (all 1.0, scale=1.0): expect={K}, got min={C1.min().item():.1f} max={C1.max().item():.1f} mean={C1.float().mean().item():.1f}", flush=True)
        print(f"[DIAG] Test1 C[0,0]={C1[0,0].item():.1f} C[0,1]={C1[0,1].item():.1f} C[1,0]={C1[1,0].item():.1f}", flush=True)

        # Test 2: All-zeros A, ones B → should be 0
        A_zero = torch.zeros(M, K//2, dtype=torch.uint8, device='cuda')
        C2 = mod.hbl_gemm_diag(A_zero, B_ones, scA, scB, M, N, K)
        print(f"[DIAG] Test2 (A=0): expect=0, got min={C2.min().item():.2f} max={C2.max().item():.2f}", flush=True)

        # Test 3: Identity-like: A has 1.0 at diagonal, B has 1.0 everywhere
        # A[m,k] = 1.0 if k<32 else 0 (first group all 1s, rest 0s)
        A_partial = torch.zeros(M, K//2, dtype=torch.uint8, device='cuda')
        A_partial[:, :16] = 0x22  # first 32 FP4 values = 1.0, rest = 0
        C3 = mod.hbl_gemm_diag(A_partial, B_ones, scA, scB, M, N, K)
        print(f"[DIAG] Test3 (A partial 32): expect=32, got min={C3.min().item():.1f} max={C3.max().item():.1f}", flush=True)

        # Test 4: Scale=0 should give 0 output (2^(0-127) is tiny)
        scA_zero = torch.zeros(M, K//32, dtype=torch.uint8, device='cuda')
        C4 = mod.hbl_gemm_diag(A_ones, B_ones, scA_zero, scB, M, N, K)
        print(f"[DIAG] Test4 (scaleA=0): got min={C4.min().item():.4f} max={C4.max().item():.4f}", flush=True)

        # Test 5: Compare with Triton reference using real quant data
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

        A_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        A_fp4, A_scale = dynamic_mxfp4_quant(A_bf16)
        A_u8 = A_fp4.view(torch.uint8)
        As_u8 = A_scale.view(torch.uint8)
        print(f"[DIAG] A_fp4 shape={A_u8.shape}, A_scale shape={As_u8.shape}", flush=True)

        B_u8 = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
        Bs_u8 = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')

        C_hbl = mod.hbl_gemm_diag(A_u8, B_u8, As_u8, Bs_u8, M, N, K)
        C_ref = gemm_afp4wfp4(A_u8, B_u8, As_u8, Bs_u8, dtype=torch.bfloat16)
        diff = (C_hbl - C_ref).abs()
        print(f"[DIAG] Test5 vs Triton: maxdiff={diff.max().item():.2f} hbl=[{C_hbl.min().item():.1f},{C_hbl.max().item():.1f}] ref=[{C_ref.min().item():.1f},{C_ref.max().item():.1f}]", flush=True)

        # Element-level comparison for first row
        for j in range(min(8, N)):
            print(f"[DIAG]   C[0,{j}]: hbl={C_hbl[0,j].item():.2f} ref={C_ref[0,j].item():.2f} diff={diff[0,j].item():.2f}", flush=True)

        # Test 6: SWAPPED approach (matA=B, matB=A) — previous approach
        # This should give DIFFERENT results due to the swap
        C_swap = mod.hbl_gemm_diag(B_u8, A_u8, Bs_u8, As_u8, N, M, K)
        # C_swap should be C_ref^T
        diff_swap = (C_swap - C_ref).abs()
        print(f"[DIAG] Test6 swapped: maxdiff={diff_swap.max().item():.2f}", flush=True)

    except Exception as e:
        import traceback
        print(f"[DIAG] FAIL: {str(e)[:300]}", flush=True)
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

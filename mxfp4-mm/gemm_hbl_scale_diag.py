#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — hipBLASLt FP4 scale diagnostic. Tests ALL scale formats + layout orders.
The FP4 data is CORRECT (all-ones test = perfect). Only scales are wrong.
"""
import os
os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
os.environ["ROCM_HOME"] = "/opt/rocm"
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch, time
from task import input_t, output_t

_call = 0; _probed = False

CPP_FWD = """
torch::Tensor hbl_gemm_scaled(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K, int64_t scale_mode, int64_t order_a, int64_t order_b);
int hbl_check_order(int64_t M, int64_t N, int64_t K, int64_t scale_mode, int64_t order_a, int64_t order_b);
"""

HIP_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-types.h>

static hipblasLtHandle_t g_h = nullptr;

int hbl_check_order(int64_t M, int64_t N, int64_t K, int64_t scale_mode, int64_t order_a, int64_t order_b) {
    if (!g_h) hipblasLtCreate(&g_h);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    int32_t sm = (int32_t)scale_mode;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    void *dA = nullptr, *dB = nullptr;
    hipMalloc(&dA, 4096); hipMalloc(&dB, 4096);
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dB, sizeof(void*));

    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, (hipDataType)33, K, M, K);
    hipblasLtMatrixLayoutCreate(&lB, (hipDataType)33, K, N, K);
    hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, M, N, M);

    if (order_a >= 0) {
        int32_t o = (int32_t)order_a;
        hipblasLtMatrixLayoutSetAttribute(lA, (hipblasLtMatrixLayoutAttribute_t)1, &o, sizeof(o));
    }
    if (order_b >= 0) {
        int32_t o = (int32_t)order_b;
        hipblasLtMatrixLayoutSetAttribute(lB, (hipblasLtMatrixLayoutAttribute_t)1, &o, sizeof(o));
    }

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t ws = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    hipblasLtMatmulHeuristicResult_t res[8];
    int nA = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_h, desc, lA, lB, lC, lC, pref, 8, res, &nA);

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);
    hipFree(dA); hipFree(dB);
    return nA;
}

torch::Tensor hbl_gemm_scaled(torch::Tensor A_fp4, torch::Tensor B_fp4,
    torch::Tensor A_scale, torch::Tensor B_scale,
    int64_t M, int64_t N, int64_t K, int64_t scale_mode, int64_t order_a, int64_t order_b)
{
    if (!g_h) hipblasLtCreate(&g_h);

    hipblasLtMatmulDesc_t desc;
    hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, (hipDataType)0);
    hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    int32_t sm = (int32_t)scale_mode;
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    void* pA = A_scale.data_ptr();
    void* pB = B_scale.data_ptr();
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &pA, sizeof(void*));
    hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &pB, sizeof(void*));

    hipblasLtMatrixLayout_t lA, lB, lC;
    hipblasLtMatrixLayoutCreate(&lA, (hipDataType)33, K, M, K);
    hipblasLtMatrixLayoutCreate(&lB, (hipDataType)33, K, N, K);
    hipblasLtMatrixLayoutCreate(&lC, (hipDataType)14, M, N, M);

    if (order_a >= 0) {
        int32_t o = (int32_t)order_a;
        hipblasLtMatrixLayoutSetAttribute(lA, (hipblasLtMatrixLayoutAttribute_t)1, &o, sizeof(o));
    }
    if (order_b >= 0) {
        int32_t o = (int32_t)order_b;
        hipblasLtMatrixLayoutSetAttribute(lB, (hipblasLtMatrixLayoutAttribute_t)1, &o, sizeof(o));
    }

    hipblasLtMatmulPreference_t pref;
    hipblasLtMatmulPreferenceCreate(&pref);
    int64_t ws = 64*1024*1024;
    hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    hipblasLtMatmulHeuristicResult_t res[8];
    int nA = 0;
    hipblasLtMatmulAlgoGetHeuristic(g_h, desc, lA, lB, lC, lC, pref, 8, res, &nA);

    auto D = torch::empty({N, M}, torch::dtype(torch::kBFloat16).device(A_fp4.device()));

    if (nA > 0) {
        void* wk = nullptr;
        if (res[0].workspaceSize > 0) hipMalloc(&wk, res[0].workspaceSize);
        float alpha = 1.0f, beta = 0.0f;
        hipblasLtMatmul(g_h, desc, &alpha,
            A_fp4.data_ptr(), lA, B_fp4.data_ptr(), lB,
            &beta, D.data_ptr(), lC, D.data_ptr(), lC,
            &res[0].algo, wk, res[0].workspaceSize, 0);
        hipDeviceSynchronize();
        if (wk) hipFree(wk);
    }

    hipblasLtMatmulPreferenceDestroy(pref);
    hipblasLtMatrixLayoutDestroy(lA);
    hipblasLtMatrixLayoutDestroy(lB);
    hipblasLtMatrixLayoutDestroy(lC);
    hipblasLtMatmulDescDestroy(desc);

    return D.t().contiguous();
}
"""

def _run_diag():
    global _probed
    if _probed: return
    _probed = True
    try:
        from torch.utils.cpp_extension import load_inline
        t0 = time.time()
        mod = load_inline(
            name="hbl_diag_v2",
            cpp_sources=CPP_FWD,
            cuda_sources=HIP_SRC,
            functions=["hbl_gemm_scaled", "hbl_check_order"],
            extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-I/opt/rocm/include"],
            extra_ldflags=["-L/opt/rocm/lib", "-lhipblaslt"],
            verbose=False,
        )
        print(f"[D] Compiled {time.time()-t0:.1f}s", flush=True)

        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.utility.fp4_utils import e8m0_shuffle

        M, N, K = 32, 64, 512
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
        Au = A_fp4.view(torch.uint8)
        Asu_raw = A_sc.view(torch.uint8)
        Asu_sh = e8m0_shuffle(A_sc).view(torch.uint8)
        Bu = torch.randint(0, 256, (N, K//2), dtype=torch.uint8, device='cuda')
        Bsu = torch.full((N, K//32), 127, dtype=torch.uint8, device='cuda')
        C_ref = gemm_afp4wfp4(Au, Bu, Asu_raw, Bsu, dtype=torch.bfloat16)

        print(f"[D] A_scale raw shape={Asu_raw.shape} sh shape={Asu_sh.shape}", flush=True)
        print(f"[D] C_ref range=[{C_ref.min().item():.1f},{C_ref.max().item():.1f}]", flush=True)

        # Test 1: Check which layout orders give algorithms
        print("[D] === ORDER CHECK ===", flush=True)
        for oa in [-1, 0, 102, 103]:
            for ob in [-1, 0, 102, 103]:
                n = mod.hbl_check_order(M, N, K, 2, oa, ob)
                if n > 0:
                    print(f"[D] order A={oa} B={ob} sm=2: {n} algos", flush=True)

        # Also check scale_mode=6 (BLK32_UE8M0_32_8_EXT)
        for oa in [-1, 0, 102, 103]:
            for ob in [-1, 0, 102, 103]:
                n = mod.hbl_check_order(M, N, K, 6, oa, ob)
                if n > 0:
                    print(f"[D] order A={oa} B={ob} sm=6: {n} algos", flush=True)

        # CRITICAL: e8m0_shuffle returns [256,16] for input [32,16] — 8x larger!
        # hipBLASLt might expect this expanded format, not the compact [32,16]
        print(f"[D] Asu_sh total bytes: {Asu_sh.numel()}", flush=True)

        # Test 2: All scale variants
        print("[D] === SCALE VARIANTS ===", flush=True)
        best_d = 999; best_name = ""

        # Variant A: raw [32,16]
        try:
            C = mod.hbl_gemm_scaled(Au, Bu, Asu_raw, Bsu, M, N, K, 2, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] raw[{Asu_raw.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "raw"
        except Exception as e:
            print(f"[D] raw: {str(e)[:100]}", flush=True)

        # Variant B: FULL shuffled [256,16] — pass as-is (larger buffer!)
        try:
            C = mod.hbl_gemm_scaled(Au, Bu, Asu_sh.contiguous(), Bsu, M, N, K, 2, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] shuffled_full[{Asu_sh.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "shuffled_full"
        except Exception as e:
            print(f"[D] shuffled_full: {str(e)[:100]}", flush=True)

        # Variant C: raw transposed [16,32]
        try:
            raw_T = Asu_raw.t().contiguous()
            C = mod.hbl_gemm_scaled(Au, Bu, raw_T, Bsu, M, N, K, 2, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] raw_T[{raw_T.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "raw_T"
        except Exception as e:
            print(f"[D] raw_T: {str(e)[:100]}", flush=True)

        # Variant D: shuffled then truncated to [32,16]
        try:
            sh_trunc = Asu_sh.contiguous().view(-1)[:M*(K//32)].view(M, K//32)
            C = mod.hbl_gemm_scaled(Au, Bu, sh_trunc, Bsu, M, N, K, 2, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] sh_trunc[{sh_trunc.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "sh_trunc"
        except Exception as e:
            print(f"[D] sh_trunc: {str(e)[:100]}", flush=True)

        # Variant E: shuffled with scale_mode=6 (pre-swizzled)
        try:
            C = mod.hbl_gemm_scaled(Au, Bu, Asu_sh.contiguous(), Bsu, M, N, K, 6, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] sh_sm6[{Asu_sh.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "sh_sm6"
        except Exception as e:
            print(f"[D] sh_sm6: {str(e)[:100]}", flush=True)

        # Variant F: raw with scale_mode=6
        try:
            C = mod.hbl_gemm_scaled(Au, Bu, Asu_raw, Bsu, M, N, K, 6, -1, -1)
            d = (C - C_ref).abs().max().item()
            print(f"[D] raw_sm6[{Asu_raw.shape}]: maxdiff={d:.2f}", flush=True)
            if d < best_d: best_d = d; best_name = "raw_sm6"
        except Exception as e:
            print(f"[D] raw_sm6: {str(e)[:100]}", flush=True)

        # Variant G: shuffled with layout orders
        for oa, ob in [(102,102), (103,103), (102,103)]:
            try:
                C = mod.hbl_gemm_scaled(Au, Bu, Asu_sh.contiguous(), Bsu, M, N, K, 2, oa, ob)
                d = (C - C_ref).abs().max().item()
                print(f"[D] sh_o{oa}_{ob}: maxdiff={d:.2f}", flush=True)
                if d < best_d: best_d = d; best_name = f"sh_o{oa}_{ob}"
            except Exception as e:
                print(f"[D] sh_o{oa}_{ob}: {str(e)[:80]}", flush=True)

        # Variant H: raw with layout orders
        for oa, ob in [(102,102), (103,103)]:
            try:
                C = mod.hbl_gemm_scaled(Au, Bu, Asu_raw, Bsu, M, N, K, 2, oa, ob)
                d = (C - C_ref).abs().max().item()
                print(f"[D] raw_o{oa}_{ob}: maxdiff={d:.2f}", flush=True)
                if d < best_d: best_d = d; best_name = f"raw_o{oa}_{ob}"
            except Exception as e:
                print(f"[D] raw_o{oa}_{ob}: {str(e)[:80]}", flush=True)

        # Test 5: 2-block diagnostic (K=64, 2 scale groups)
        print("[D] === 2-BLOCK DIAGNOSTIC ===", flush=True)
        M2, N2, K2 = 32, 32, 128  # K/32=4 scale groups
        A2 = torch.full((M2, K2//2), 0x22, dtype=torch.uint8, device='cuda')  # all 1.0
        B2 = torch.full((N2, K2//2), 0x22, dtype=torch.uint8, device='cuda')  # all 1.0
        # Scale: group 0=127(1.0), group 1=128(2.0), group 2=129(4.0), group 3=127(1.0)
        sA2 = torch.full((M2, K2//32), 127, dtype=torch.uint8, device='cuda')
        sA2[:, 1] = 128  # group 1 scale = 2.0
        sA2[:, 2] = 129  # group 2 scale = 4.0
        sB2 = torch.full((N2, K2//32), 127, dtype=torch.uint8, device='cuda')
        # Expected: each output = 32*1.0*1.0 + 32*1.0*2.0 + 32*1.0*4.0 + 32*1.0*1.0
        #         = 32 + 64 + 128 + 32 = 256
        C5 = mod.hbl_gemm_scaled(A2, B2, sA2, sB2, M2, N2, K2, 2, -1, -1)
        expected = 32*1.0 + 32*2.0 + 32*4.0 + 32*1.0  # = 256
        print(f"[D] 2block: expect={expected} got C[0,0]={C5[0,0].item():.1f} C[0,1]={C5[0,1].item():.1f} C[1,0]={C5[1,0].item():.1f}", flush=True)

        # Also check uniform scale=127 → expect 128
        sA2u = torch.full((M2, K2//32), 127, dtype=torch.uint8, device='cuda')
        C5u = mod.hbl_gemm_scaled(A2, B2, sA2u, sB2, M2, N2, K2, 2, -1, -1)
        print(f"[D] uniform: expect=128 got={C5u[0,0].item():.1f}", flush=True)

        print(f"[D] === BEST: {best_name} maxdiff={best_d:.2f} ===", flush=True)

    except Exception as e:
        import traceback
        print(f"[D] FAIL: {str(e)[:300]}", flush=True)
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
    if _call == 1: _run_diag()
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

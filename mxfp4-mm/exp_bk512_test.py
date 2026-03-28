#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Test BK=512 for K=7168 fused kernel (more work per tile, fewer iterations).
Also test fused non-splitK for K=7168 M=16 (maybe enough tiles?).
Print timing to STDOUT.
"""
from task import input_t, output_t
import torch, sys, time, triton, triton.language as tl

try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
    _HAS = True
except:
    _HAS = False

_ref=None; _raw=None; _sh=None; _bq=None; _tested=set()

def _unshuffle(s):
    s=s.view(torch.uint8); sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
def _shuffle(s):
    s=s.clone().view(torch.uint8); sm,sn=s.shape
    return s.view(sm//32,2,16,sn//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(sm//32,sn*32)

@triton.jit
def _fk(a,b,c,bs, M,N,K, sa0,sa1,sb0,sb1,sc0,sc1,sbs0,sbs1,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,mf:tl.constexpr):
    SG:tl.constexpr=32
    pid=tl.program_id(0); npn=tl.cdiv(N,BN)
    pm=pid//npn; pn=pid%npn
    om=(pm*BM+tl.arange(0,BM))%M; on=(pn*BN+tl.arange(0,BN))%N
    ap=a+om[:,None]*sa0+tl.arange(0,BK)[None,:]*sa1
    bp=b+tl.arange(0,BK//2)[:,None]*sb0+on[None,:]*sb1
    obn=(pn*(BN//32)+tl.arange(0,BN//32))%(N//32)
    oks=tl.arange(0,BK//SG*32)
    bsp=bs+obn[:,None]*sbs0+oks[None,:]*sbs1
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    nki=tl.cdiv(K,BK)
    for ki in range(0,nki):
        ab=tl.load(ap).to(tl.float32)
        af,asc=_mxfp4_quant_op(ab,BK,BM,SG)
        bv=tl.load(bp)
        bsc=tl.load(bsp).reshape(BN//32,BK//SG//8,4,16,2,2,1).permute(0,5,3,1,4,2,6).reshape(BN,BK//SG)
        acc+=tl.dot_scaled(af,asc,"e2m1",bv,bsc,"e2m1")
        ap+=BK*sa1; bp+=(BK//2)*sb0; bsp+=BK*sbs1
    cv=acc.to(tl.bfloat16)
    ocm=pm*BM+tl.arange(0,BM).to(tl.int64); ocn=pn*BN+tl.arange(0,BN).to(tl.int64)
    tl.store(c+ocm[:,None]*sc0+ocn[None,:]*sc1,cv,mask=(ocm[:,None]<M)&(ocn[None,:]<N))

def _run(A,bq,bsh,m,n,k,BM,BN,BK):
    pad=(BM-m%BM)%BM
    Ap=torch.nn.functional.pad(A,(0,0,0,pad),value=0.0) if pad>0 else A
    mp=Ap.shape[0]
    C=torch.empty((mp,n),dtype=torch.bfloat16,device='cuda')
    g=(triton.cdiv(mp,BM)*triton.cdiv(n,BN),)
    _fk[g](Ap,bq,C,bsh,mp,n,k,Ap.stride(0),Ap.stride(1),bq.stride(1),bq.stride(0),
           C.stride(0),C.stride(1),bsh.stride(0),bsh.stride(1),
           BM=BM,BN=BN,BK=BK,mf=16,num_warps=4,num_stages=2,matrix_instr_nonkdim=16)
    return C[:m,:n] if pad>0 else C

def _test(A,bq,braw,bsh,m,n,k):
    sk=(m,n,k)
    if sk in _tested: return
    _tested.add(sk)
    if not _HAS or k not in (7168,2048): return
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    K7={"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None,"NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
    NI=10
    for _ in range(5): gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(NI): gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
    torch.cuda.synchronize()
    da=(time.time()-t0)/NI*1e6
    # Test fused non-splitK with BK=256 (single pass through all K)
    try:
        BM=32; BN=64; BK=256
        for _ in range(3): _run(A,bq,bsh,m,n,k,BM,BN,BK)
        torch.cuda.synchronize()
        Cc=_run(A,bq,bsh,m,n,k,BM,BN,BK); torch.cuda.synchronize()
        Cr=gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
        err=(Cc.float()-Cr.float()).abs().max().item()
        t0=time.time()
        for _ in range(NI): _run(A,bq,bsh,m,n,k,BM,BN,BK)
        torch.cuda.synchronize()
        dc=(time.time()-t0)/NI*1e6
        print(f"FUSED_NOSPLIT M{m}N{n}K{k} BK256: {dc:.1f}us aiter={da:.1f}us x{da/dc:.2f} err={err:.6f}",flush=True)
    except Exception as e:
        print(f"FUSED_NOSPLIT M{m}N{n}K{k}: FAIL {type(e).__name__}: {str(e)[:100]}",flush=True)

_K7={"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None,"NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}

def custom_kernel(data: input_t) -> output_t:
    global _ref,_raw,_sh,_bq
    A,B,Bq,Bs,Bss=data; m,k=A.shape; n=B.shape[0]
    if _ref is not Bss:
        _ref=Bss; _raw=_unshuffle(Bss); _bq=Bq.view(torch.uint8); _sh=_shuffle(_raw)
    _test(A,_bq,_raw,_sh,m,n,k)
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bq,asc,_raw,dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A,_bq,_raw,dtype=torch.bfloat16,config=_K7 if k==7168 else None)

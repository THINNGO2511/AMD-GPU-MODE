#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Timing diagnostic for fused split-K shapes only (K=7168, K=2048).
Prints to STDOUT: custom vs aiter timing for each.
"""
from task import input_t, output_t
import torch, sys, time, triton, triton.language as tl

try:
    from aiter.ops.triton.quant import _mxfp4_quant_op
    _HAS_QOP = True
except ImportError:
    _HAS_QOP = False

_ref=None; _raw=None; _sh=None; _bq=None; _tested=set(); _c={}; _pp={}

def _unshuffle(s):
    s=s.view(torch.uint8); sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
def _shuffle(s):
    s=s.clone().view(torch.uint8); sm,sn=s.shape
    return s.view(sm//32,2,16,sn//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(sm//32,sn*32)

@triton.jit
def _sk(a,b,pp,bs, M,N,K, sa0,sa1,sb0,sb1,sp0,sp1,sp2,sbs0,sbs1,
        BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
        KS:tl.constexpr,KPS:tl.constexpr,mf:tl.constexpr):
    SG:tl.constexpr=32
    pid=tl.program_id(0); tiles=tl.cdiv(M,BM)*tl.cdiv(N,BN)
    si=pid//tiles; ti=pid%tiles
    npn=tl.cdiv(N,BN); pm=ti//npn; pn=ti%npn
    om=(pm*BM+tl.arange(0,BM))%M; on=(pn*BN+tl.arange(0,BN))%N
    ks=si*KPS
    ap=a+om[:,None]*sa0+(ks+tl.arange(0,BK))[None,:]*sa1
    bp=b+(ks//2+tl.arange(0,BK//2))[:,None]*sb0+on[None,:]*sb1
    obn=(pn*(BN//32)+tl.arange(0,BN//32))%(N//32)
    oks=tl.arange(0,BK//SG*32)
    bsp=bs+obn[:,None]*sbs0+(ks+oks)[None,:]*sbs1
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    iters=tl.cdiv(KPS,BK)
    for ki in range(0,iters):
        ko=ks+ki*BK
        if ko<K:
            ab=tl.load(ap).to(tl.float32)
            af,asc=_mxfp4_quant_op(ab,BK,BM,SG)
            bv=tl.load(bp)
            bsc=tl.load(bsp).reshape(BN//32,BK//SG//8,4,16,2,2,1).permute(0,5,3,1,4,2,6).reshape(BN,BK//SG)
            acc+=tl.dot_scaled(af,asc,"e2m1",bv,bsc,"e2m1")
            ap+=BK*sa1; bp+=(BK//2)*sb0; bsp+=BK*sbs1
    ocm=pm*BM+tl.arange(0,BM).to(tl.int64); ocn=pn*BN+tl.arange(0,BN).to(tl.int64)
    mk=(ocm[:,None]<M)&(ocn[None,:]<N)
    tl.store(pp+si*sp0+ocm[:,None]*sp1+ocn[None,:]*sp2,acc,mask=mk)

@triton.jit
def _red(pp,c,M,N,sp0,sp1,sp2,sc0,sc1,KS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    pid=tl.program_id(0); npn=tl.cdiv(N,BN); pm=pid//npn; pn=pid%npn
    om=pm*BM+tl.arange(0,BM); on=pn*BN+tl.arange(0,BN)
    mk=(om[:,None]<M)&(on[None,:]<N)
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for s in range(KS):
        acc+=tl.load(pp+s*sp0+om[:,None]*sp1+on[None,:]*sp2,mask=mk,other=0.0)
    tl.store(c+om[:,None].to(tl.int64)*sc0+on[None,:].to(tl.int64)*sc1,acc.to(tl.bfloat16),mask=mk)

def _run_sk(A,bq,bsh,m,n,k,BM,BN,KS):
    BK=256; KPS=k//KS
    pad=(BM-m%BM)%BM
    Ap=torch.nn.functional.pad(A,(0,0,0,pad),value=0.0) if pad>0 else A
    mp=Ap.shape[0]
    pk=(KS,mp,n)
    if pk not in _pp: _pp[pk]=torch.empty((KS,mp,n),dtype=torch.float32,device='cuda')
    pp=_pp[pk]
    tiles=triton.cdiv(mp,BM)*triton.cdiv(n,BN)
    _sk[(KS*tiles,)](Ap,bq,pp,bsh,mp,n,k,Ap.stride(0),Ap.stride(1),bq.stride(1),bq.stride(0),
                      pp.stride(0),pp.stride(1),pp.stride(2),bsh.stride(0),bsh.stride(1),
                      BM=BM,BN=BN,BK=BK,KS=KS,KPS=KPS,mf=16,num_warps=4,num_stages=2,matrix_instr_nonkdim=16)
    ck=(mp,n)
    if ck not in _c: _c[ck]=torch.empty((mp,n),dtype=torch.bfloat16,device='cuda')
    C=_c[ck]
    RBM,RBN=min(BM,32),min(BN,128)
    _red[(triton.cdiv(mp,RBM)*triton.cdiv(n,RBN),)](pp,C,mp,n,pp.stride(0),pp.stride(1),pp.stride(2),
                                                      C.stride(0),C.stride(1),KS=KS,BM=RBM,BN=RBN,
                                                      num_warps=4,num_stages=1)
    return C[:m,:n] if pad>0 else C

def _test(A,bq,braw,bsh,m,n,k):
    sk=(m,n,k)
    if sk in _tested or k not in (7168,2048): return
    _tested.add(sk)
    if not _HAS_QOP: print(f"NO_QOP",flush=True); return
    cfgs=[]
    if k==7168: cfgs=[(32,64,7,"BM32_BN64_KS7"),(32,128,7,"BM32_BN128_KS7")]
    elif k==2048: cfgs=[(64,128,2,"BM64_BN128_KS2"),(32,64,2,"BM32_BN64_KS2")]
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    K7={"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None,"NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
    NI=10
    for _ in range(5): gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
    torch.cuda.synchronize()
    t0=time.time()
    for _ in range(NI): gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
    torch.cuda.synchronize()
    da=(time.time()-t0)/NI*1e6
    for BM,BN,KS,name in cfgs:
        try:
            for _ in range(3): _run_sk(A,bq,bsh,m,n,k,BM,BN,KS)
            torch.cuda.synchronize()
            Cc=_run_sk(A,bq,bsh,m,n,k,BM,BN,KS); torch.cuda.synchronize()
            Cr=gemm_a16wfp4(A,bq,braw,dtype=torch.bfloat16,config=K7 if k==7168 else None)
            err=(Cc.float()-Cr.float()).abs().max().item()
            t0=time.time()
            for _ in range(NI): _run_sk(A,bq,bsh,m,n,k,BM,BN,KS)
            torch.cuda.synchronize()
            dc=(time.time()-t0)/NI*1e6
            print(f"SPLITK M{m}N{n}K{k} {name}: {dc:.1f}us aiter={da:.1f}us x{da/dc:.2f} err={err:.6f}",flush=True)
        except Exception as e:
            print(f"SPLITK M{m}N{n}K{k} {name}: FAIL {type(e).__name__}: {str(e)[:100]}",flush=True)

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

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
from task import input_t, output_t
import torch, triton, triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
_br=None;_bs=None;_bq=None;_cc={}
def _ush(s):
    s=s.view(torch.uint8);sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
@triton.autotune(configs=[triton.Config({'BM':32,'BN':256,'BK':128,'GS':4,'waves_per_eu':4,'matrix_instr_nonkdim':16},num_warps=4,num_stages=2),triton.Config({'BM':32,'BN':256,'BK':128,'GS':4},num_warps=2,num_stages=2)],key=['M','N','K'])
@triton.jit
def _fqg(a_ptr,b_ptr,c_ptr,bs_ptr,M,N,K,sa,sk,sbk,sbn,scm,scn,ssn,ssk,
    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GS:tl.constexpr):
    SG:tl.constexpr=32;pid=tl.program_id(0)
    npm=tl.cdiv(M,BM);npn=tl.cdiv(N,BN);nig=GS*npn;gid=pid//nig
    fm=gid*GS;gsm=min(npm-fm,GS);pm=fm+((pid%nig)%gsm);pn=(pid%nig)//gsm
    om=(pm*BM+tl.arange(0,BM))%M;on=(pn*BN+tl.arange(0,BN))%N
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for ki in range(tl.cdiv(K,BK)):
        ks=ki*BK
        a=tl.load(a_ptr+om[:,None]*sa+(ks+tl.arange(0,BK))[None,:]*sk).to(tl.float32)
        af,asc=_mxfp4_quant_op(a,BK,BM,SG)
        bf=tl.load(b_ptr+(ks//2+tl.arange(0,BK//2))[:,None]*sbk+on[None,:]*sbn)
        bsc=tl.load(bs_ptr+on[:,None]*ssn+(ks//SG+tl.arange(0,BK//SG))[None,:]*ssk)
        acc=tl.dot_scaled(af,asc,"e2m1",bf,bsc,"e2m1",acc)
    c=acc.to(tl.bfloat16)
    ocm=pm*BM+tl.arange(0,BM).to(tl.int64);ocn=pn*BN+tl.arange(0,BN).to(tl.int64)
    tl.store(c_ptr+ocm[:,None]*scm+ocn[None,:]*scn,c,mask=(ocm[:,None]<M)&(ocn[None,:]<N))
def custom_kernel(data:input_t)->output_t:
    global _br,_bs,_bq
    A,B,Bq,Bsh,Bss=data;m,k=A.shape;n=B.shape[0]
    if _br is not Bss:_br=Bss;_bs=_ush(Bss);_bq=Bq.view(torch.uint8)
    if k<=1024:
        ck=(m,n)
        if ck not in _cc:_cc[ck]=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
        C=_cc[ck]
        grid=lambda META:(triton.cdiv(m,META['BM'])*triton.cdiv(n,META['BN']),)
        _fqg[grid](A,_bq,C,_bs,m,n,k,A.stride(0),A.stride(1),_bq.stride(1),_bq.stride(0),
            C.stride(0),C.stride(1),_bs.stride(0),_bs.stride(1))
        return C
    else:
        Af,As=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(Af,_bq,As,_bs,dtype=torch.bfloat16)

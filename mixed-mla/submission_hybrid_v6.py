#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA Hybrid v6: Use the PROVEN nonpersist_v2 code for kv<=1024 bs<=64,
and PROVEN hybrid_v5 persistent code for everything else.
Completely separate code paths, no shared state.
"""
import torch, triton, triton.language as tl, aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

FP8 = aiter_dtypes.fp8
BF16 = torch.bfloat16
PS = 2
FM = float(torch.finfo(FP8).max)
_c = {}

@triton.jit
def _am(q, a, N, B: tl.constexpr):
    p = tl.program_id(0); o = p*B+tl.arange(0,B); m = o<N
    tl.atomic_max(a, tl.max(tl.abs(tl.load(q+o,mask=m,other=0.0).to(tl.float32))))

@triton.jit
def _cf(q, out, s, a, FM: tl.constexpr, N, B: tl.constexpr):
    am = tl.load(a); am = tl.where(am<1e-12,1e-12,am); sc = am/FM
    if tl.program_id(0)==0: tl.store(s, sc)
    p = tl.program_id(0); o = p*B+tl.arange(0,B); m = o<N
    x = tl.load(q+o,mask=m,other=0.0).to(tl.float32)/sc
    tl.store(out+o, tl.clamp(x,-FM,FM).to(out.dtype.element_ty), mask=m)

def _qfp8(q, ab, sb, qf):
    N=q.numel(); B=4096; g=((N+B-1)//B,)
    ab.zero_(); _am[g](q,ab,N,B=B); _cf[g](q,qf,sb,ab,FM=FM,N=N,B=B)

def _splits(bs, tkv):
    cu=304; ak=tkv/bs; oh=84.1
    t=[(bs*i/((bs*i+cu-1)//cu*cu)*ak/(ak+oh*i),i) for i in range(1,17)]
    ns=sorted(t,key=lambda x:x[0],reverse=True)[0][1]
    mb=32; ns=min(ns,int(tkv/bs+mb-1)//mb)
    if ns>1: ns=min(ns,int(abs(tkv/bs-1)//mb+1))
    return max(1,ns)

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs=config["batch_size"]; nq=config["num_heads"]; nkv=config["num_kv_heads"]
    dq=config["qk_head_dim"]; dv=config["v_head_dim"]; sm=config["sm_scale"]
    kvl=config["kv_seq_len"]; tkv=bs*kvl
    kv_fp8, kv_sc = kv_data["fp8"]
    
    use_np = (kvl <= 1024 and bs <= 64)
    ck = (bs, kvl, use_np)
    
    if ck not in _c:
        sl=kv_indptr[1:]-kv_indptr[:-1]; np_=(sl+PS-1)//PS
        ki=torch.zeros(bs+1,dtype=torch.int32,device=q.device)
        ki[1:]=torch.cumsum(np_,0); tp=ki[-1].item()
        kl=sl%PS; kl[kl==0]=PS
        kx=torch.arange(tp,dtype=torch.int32,device=q.device)
        o=torch.empty(q.shape[0],nq,dv,dtype=BF16,device="cuda")
        ab=torch.zeros(1,dtype=torch.float32,device="cuda")
        sb=torch.empty(1,dtype=torch.float32,device="cuda")
        qf=torch.empty(q.shape[0]*nq*dq,dtype=FP8,device="cuda")
        
        if use_np:
            ns=_splits(bs,tkv)
            ind=torch.arange(0,(bs+1)*ns,ns,dtype=torch.int32,device=q.device)
            if ns==1:
                lg=o.view(q.shape[0],1,nq,dv)
            else:
                lg=torch.empty(q.shape[0],ns,nq,dv,dtype=torch.float32,device="cuda")
            ls=torch.empty(q.shape[0],ns,nq,1,dtype=torch.float32,device="cuda")
            _c[ck]=("np",ki,kl,kx,o,ab,sb,qf,ns,ind,lg,ls)
        else:
            nks=16 if tkv>8192 else 8
            ua=kvl<=1024
            dt=BF16 if ua else FP8
            info=get_mla_metadata_info_v1(bs,1,nq,dt,FP8,is_sparse=False,fast_mode=False,
                num_kv_splits=nks,intra_batch_mode=True)
            wk=[torch.empty(s,dtype=t,device="cuda") for s,t in info]
            wm,wi,wis,ri,rfm,rpm=wk
            get_mla_metadata_v1(qo_indptr,ki,kl,nq//nkv,nkv,True,wm,wis,wi,ri,rfm,rpm,
                page_size=PS,kv_granularity=max(PS,16),max_seqlen_qo=1,uni_seqlen_qo=1,
                fast_mode=False,max_split_per_batch=nks,intra_batch_mode=True,dtype_q=dt,dtype_kv=FP8)
            np_=rpm.size(0)
            lg=torch.empty(np_,1,nq,dv,dtype=torch.float32,device="cuda")
            ls=torch.empty(np_,1,nq,1,dtype=torch.float32,device="cuda")
            _c[ck]=("ps",ki,kl,kx,o,ab,sb,qf,wm,wi,wis,ri,rfm,rpm,lg,ls,ua)
    
    e=_c[ck]
    kv4=kv_fp8.view(-1,PS,nkv,kv_fp8.shape[-1])
    
    if e[0]=="np":
        _,ki,kl,kx,o,ab,sb,qf,ns,ind,lg,ls=e
        _qfp8(q,ab,sb,qf); qf8=qf.view(q.shape[0],nq,dq)
        aiter.mla_decode_stage1_asm_fwd(qf8,kv4,qo_indptr,ki,kx,kl,
            ind,None,None,None,1,PS,nkv,sm,lg,ls,o,sb,kv_sc)
    else:
        _,ki,kl,kx,o,ab,sb,qf,wm,wi,wis,ri,rfm,rpm,lg,ls,ua=e
        if ua:
            aiter.mla_decode_stage1_asm_fwd(q,kv4,qo_indptr,ki,kx,kl,
                None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,None,kv_sc)
        else:
            _qfp8(q,ab,sb,qf); qf8=qf.view(q.shape[0],nq,dq)
            aiter.mla_decode_stage1_asm_fwd(qf8,kv4,qo_indptr,ki,kx,kl,
                None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,sb,kv_sc)
        aiter.mla_reduce_v1(lg,ls,ri,rfm,rpm,1,o,None)
    return o

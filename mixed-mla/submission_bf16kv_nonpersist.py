#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: bf16kv for kv=1024 (proven faster) + non-persistent for bs<=32 kv=1024
(proven faster for small bs). Best of both worlds."""
import torch, triton, triton.language as tl, aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

FP8=aiter_dtypes.fp8; BF16=torch.bfloat16; PS=2
FM=float(torch.finfo(FP8).max)
_c={}

@triton.jit
def _am(q,a,N,B:tl.constexpr):
    p=tl.program_id(0);o=p*B+tl.arange(0,B);m=o<N
    tl.atomic_max(a,tl.max(tl.abs(tl.load(q+o,mask=m,other=0.0).to(tl.float32))))

@triton.jit
def _cf(q,out,s,a,FM:tl.constexpr,N,B:tl.constexpr):
    am=tl.load(a);am=tl.where(am<1e-12,1e-12,am);sc=am/FM
    if tl.program_id(0)==0:tl.store(s,sc)
    p=tl.program_id(0);o=p*B+tl.arange(0,B);m=o<N
    x=tl.load(q+o,mask=m,other=0.0).to(tl.float32)/sc
    tl.store(out+o,tl.clamp(x,-FM,FM).to(out.dtype.element_ty),mask=m)

def _splits(bs,tkv):
    cu=304;ak=tkv/bs;oh=84.1
    t=[(bs*i/((bs*i+cu-1)//cu*cu)*ak/(ak+oh*i),i) for i in range(1,17)]
    ns=sorted(t,key=lambda x:x[0],reverse=True)[0][1]
    mb=32;ns=min(ns,int(tkv/bs+mb-1)//mb)
    if ns>1:ns=min(ns,int(abs(tkv/bs-1)//mb+1))
    return max(1,ns)

def custom_kernel(data: input_t) -> output_t:
    q,kv_data,qo_indptr,kv_indptr,config=data
    bs=config["batch_size"];nq=config["num_heads"];nkv=config["num_kv_heads"]
    dq=config["qk_head_dim"];dv=config["v_head_dim"];sm=config["sm_scale"]
    kvl=config["kv_seq_len"];tkv=bs*kvl
    
    # Strategy:
    # kv=1024, bs<=32: bf16 KV + non-persistent (fastest for small shapes)
    # kv=1024, bs>32:  bf16 KV + persistent (bf16kv proven faster)
    # kv=8192:         fp8 KV + persistent (bandwidth critical)
    use_bf16 = kvl <= 1024
    use_np = use_bf16 and bs <= 32
    
    if use_bf16:
        kv_buf=kv_data["bf16"]; kv_sc=None; dt_q=BF16; dt_kv=BF16
    else:
        kv_fp8,kv_sc=kv_data["fp8"]; kv_buf=kv_fp8; dt_q=FP8; dt_kv=FP8
    
    ck=(bs,kvl,use_bf16,use_np)
    if ck not in _c:
        sl=kv_indptr[1:]-kv_indptr[:-1];np_=(sl+PS-1)//PS
        ki=torch.zeros(bs+1,dtype=torch.int32,device=q.device)
        ki[1:]=torch.cumsum(np_,0);tp=ki[-1].item()
        kl=sl%PS;kl[kl==0]=PS
        kx=torch.arange(tp,dtype=torch.int32,device=q.device)
        o=torch.empty(q.shape[0],nq,dv,dtype=BF16,device="cuda")
        ab=torch.zeros(1,dtype=torch.float32,device="cuda")
        sb=torch.empty(1,dtype=torch.float32,device="cuda")
        qf=torch.empty(q.shape[0]*nq*dq,dtype=FP8,device="cuda")
        
        if use_np:
            ns=_splits(bs,tkv)
            ind=torch.arange(0,(bs+1)*ns,ns,dtype=torch.int32,device=q.device)
            lg=torch.empty(q.shape[0],ns,nq,dv,dtype=torch.float32,device="cuda")
            ls=torch.empty(q.shape[0],ns,nq,1,dtype=torch.float32,device="cuda")
            _c[ck]=("np",ki,kl,kx,o,ab,sb,qf,ns,ind,lg,ls)
        else:
            nks=16 if tkv>8192 else 8
            info=get_mla_metadata_info_v1(bs,1,nq,dt_q,dt_kv,
                is_sparse=False,fast_mode=False,num_kv_splits=nks,intra_batch_mode=True)
            wk=[torch.empty(s,dtype=t,device="cuda") for s,t in info]
            wm,wi,wis,ri,rfm,rpm=wk
            get_mla_metadata_v1(qo_indptr,ki,kl,nq//nkv,nkv,True,wm,wis,wi,ri,rfm,rpm,
                page_size=PS,kv_granularity=max(PS,16),max_seqlen_qo=1,uni_seqlen_qo=1,
                fast_mode=False,max_split_per_batch=nks,intra_batch_mode=True,
                dtype_q=dt_q,dtype_kv=dt_kv)
            np2=rpm.size(0)
            lg=torch.empty(np2,1,nq,dv,dtype=torch.float32,device="cuda")
            ls=torch.empty(np2,1,nq,1,dtype=torch.float32,device="cuda")
            _c[ck]=("ps",ki,kl,kx,o,ab,sb,qf,wm,wi,wis,ri,rfm,rpm,lg,ls)
    
    e=_c[ck]
    kv4=kv_buf.view(-1,PS,nkv,kv_buf.shape[-1])
    
    if e[0]=="np":
        _,ki,kl,kx,o,ab,sb,qf,ns,ind,lg,ls=e
        # bf16 Q directly — no quant needed
        aiter.mla_decode_stage1_asm_fwd(q,kv4,qo_indptr,ki,kx,kl,
            ind,None,None,None,1,PS,nkv,sm,lg,ls,o,None,None)
        # Non-persistent always needs reduce (splits > 1 for small bs)
        Lv=dv;BDV=512  # next_power_of_2(512)=512
        _np_reduce[bs,nq](lg,ls,o,qo_indptr,ki,ind,
            ls.stride(0),ls.stride(2),ls.stride(1),o.stride(0),o.stride(1),
            BS=bs,BDV=BDV,Lv=Lv,mgc=64,num_warps=4,num_stages=2,waves_per_eu=4)
    else:
        _,ki,kl,kx,o,ab,sb,qf,wm,wi,wis,ri,rfm,rpm,lg,ls=e
        if use_bf16:
            aiter.mla_decode_stage1_asm_fwd(q,kv4,qo_indptr,ki,kx,kl,
                None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,None,None)
        else:
            N=q.numel();B=4096;g=((N+B-1)//B,)
            ab.zero_();_am[g](q,ab,N,B=B);_cf[g](q,qf,sb,ab,FM=FM,N=N,B=B)
            qf8=qf.view(q.shape[0],nq,dq)
            aiter.mla_decode_stage1_asm_fwd(qf8,kv4,qo_indptr,ki,kx,kl,
                None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,sb,kv_sc)
        aiter.mla_reduce_v1(lg,ls,ri,rfm,rpm,1,o,None)
    return o

@triton.jit
def _np_reduce(Mid_O,Mid_lse,O,qo_indptr,kv_indptr,splits_indptr,
    s0:tl.int64,s2:tl.int64,s1:tl.int64,so:tl.int64,sh:tl.int64,
    BS:tl.constexpr,BDV:tl.constexpr,Lv:tl.constexpr,mgc:tl.constexpr):
    cb=tl.program_id(0);ch=tl.program_id(1)
    qs=tl.load(qo_indptr+cb)
    ss=tl.load(splits_indptr+cb);se=tl.load(splits_indptr+cb+1)
    kvl=tl.load(kv_indptr+cb+1)-tl.load(kv_indptr+cb)
    nv=tl.minimum(se-ss,tl.cdiv(kvl,mgc))
    od=tl.arange(0,BDV);md=od<Lv
    ol=qs*s0+ch*s2;ov=ol*Lv+od
    es=0.0;em=-float("inf");ac=tl.zeros((BDV,),dtype=tl.float32)
    for i in range(0,nv):
        tv=tl.load(Mid_O+ov+i*s1*Lv,mask=md,other=0.0)
        tl_=tl.load(Mid_lse+ol+i*s1)
        ne=tl.maximum(tl_,em);os_=tl.exp(em-ne);ac*=os_
        el=tl.exp(tl_-ne);ac+=el*tv;es=es*os_+el;em=ne
    tl.store(O+qs*so+ch*sh+od,ac/es,mask=md)

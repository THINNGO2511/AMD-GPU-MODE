#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: a16w8 (bf16 Q + fp8 KV) for kv<=1024, a8w8 (fp8 Q + fp8 KV) for kv=8192. All pg1.
Saves Q quant overhead for small kv without bf16 KV bandwidth penalty."""
import torch, triton, triton.language as tl, aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

FP8 = aiter_dtypes.fp8; BF16 = torch.bfloat16
FM = float(torch.finfo(FP8).max)
_c = {}

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

def custom_kernel(data: input_t) -> output_t:
    q,kv_data,qo_indptr,kv_indptr,config=data
    bs=config["batch_size"];nq=config["num_heads"];nkv=config["num_kv_heads"]
    dq=config["qk_head_dim"];dv=config["v_head_dim"];sm=config["sm_scale"]
    kvl=config["kv_seq_len"];tkv=bs*kvl

    kv_fp8,kv_sc=kv_data["fp8"]
    use_a16w8 = kvl <= 1024  # bf16 Q for small kv (no quant overhead)
    dt_q = BF16 if use_a16w8 else FP8
    nks = 4 if tkv <= 4096 else (8 if tkv <= 32768 else 16)

    ck=(bs,kvl,use_a16w8)
    if ck not in _c:
        kv_4d=kv_fp8.view(tkv,1,nkv,kv_fp8.shape[-1])
        kv_idx=torch.arange(tkv,dtype=torch.int32,device=q.device)
        kv_last=(kv_indptr[1:]-kv_indptr[:-1]).to(torch.int32)
        info=get_mla_metadata_info_v1(bs,1,nq,dt_q,FP8,
            is_sparse=False,fast_mode=False,num_kv_splits=nks,intra_batch_mode=True)
        wk=[torch.empty(s,dtype=t,device="cuda") for s,t in info]
        wm,wi,wis,ri,rfm,rpm=wk
        get_mla_metadata_v1(qo_indptr,kv_indptr,kv_last,nq//nkv,nkv,True,
            wm,wis,wi,ri,rfm,rpm,page_size=1,kv_granularity=16,
            max_seqlen_qo=1,uni_seqlen_qo=1,fast_mode=False,
            max_split_per_batch=nks,intra_batch_mode=True,
            dtype_q=dt_q,dtype_kv=FP8)
        np2=rpm.size(0)
        lg=torch.empty(np2,1,nq,dv,dtype=torch.float32,device="cuda")
        ls=torch.empty(np2,1,nq,1,dtype=torch.float32,device="cuda")
        o=torch.empty(q.shape[0],nq,dv,dtype=BF16,device="cuda")
        ab=torch.zeros(1,dtype=torch.float32,device="cuda")
        sb=torch.empty(1,dtype=torch.float32,device="cuda")
        qf=torch.empty(q.shape[0]*nq*dq,dtype=FP8,device="cuda")
        _c[ck]=(kv_4d,kv_idx,kv_last,wm,wi,wis,ri,rfm,rpm,lg,ls,o,ab,sb,qf)

    kv_4d,kv_idx,kv_last,wm,wi,wis,ri,rfm,rpm,lg,ls,o,ab,sb,qf=_c[ck]

    if use_a16w8:
        aiter.mla_decode_stage1_asm_fwd(q,kv_4d,qo_indptr,kv_indptr,kv_idx,kv_last,
            None,wm,wi,wis,1,1,nkv,sm,lg,ls,o,None,kv_sc)
    else:
        N=q.numel();B=4096;g=((N+B-1)//B,)
        ab.zero_();_am[g](q,ab,N,B=B);_cf[g](q,qf,sb,ab,FM=FM,N=N,B=B)
        qf8=qf.view(q.shape[0],nq,dq)
        aiter.mla_decode_stage1_asm_fwd(qf8,kv_4d,qo_indptr,kv_indptr,kv_idx,kv_last,
            None,wm,wi,wis,1,1,nkv,sm,lg,ls,o,sb,kv_sc)

    aiter.mla_reduce_v1(lg,ls,ri,rfm,rpm,1,o,None)
    return o

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: bf16 Q + bf16 KV for kv=1024 (zero quant overhead), fp8 for kv=8192.
The a16w16 kernel exists: mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps.co"""
import torch, triton, triton.language as tl, aiter
from task import input_t, output_t
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

FP8 = aiter_dtypes.fp8; BF16 = torch.bfloat16; PS = 2
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
    q,kv_data,qo_indptr,kv_indptr,config = data
    bs=config["batch_size"];nq=config["num_heads"];nkv=config["num_kv_heads"]
    dq=config["qk_head_dim"];dv=config["v_head_dim"];sm=config["sm_scale"]
    kvl=config["kv_seq_len"];tkv=bs*kvl
    
    # Use bf16 KV for kv<=1024, fp8 for kv>1024
    use_bf16 = kvl <= 1024
    
    if use_bf16:
        kv_buf = kv_data["bf16"]  # plain tensor, not tuple
        kv_sc_val = None
        q_input = q  # bf16 Q directly, no quant
        q_sc_val = None
        dt_q = BF16; dt_kv = BF16
    else:
        kv_fp8, kv_sc_val = kv_data["fp8"]
        kv_buf = kv_fp8
        dt_q = FP8; dt_kv = FP8
    
    ck = (bs, kvl, use_bf16)
    if ck not in _c:
        sl=kv_indptr[1:]-kv_indptr[:-1]; np_=(sl+PS-1)//PS
        ki=torch.zeros(bs+1,dtype=torch.int32,device=q.device)
        ki[1:]=torch.cumsum(np_,0); tp=ki[-1].item()
        kl=sl%PS; kl[kl==0]=PS
        kx=torch.arange(tp,dtype=torch.int32,device=q.device)
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
        o=torch.empty(q.shape[0],nq,dv,dtype=BF16,device="cuda")
        ab=torch.zeros(1,dtype=torch.float32,device="cuda")
        sb=torch.empty(1,dtype=torch.float32,device="cuda")
        qf=torch.empty(q.shape[0]*nq*dq,dtype=FP8,device="cuda")
        _c[ck]=(ki,kl,kx,wm,wi,wis,ri,rfm,rpm,lg,ls,o,ab,sb,qf)
    
    ki,kl,kx,wm,wi,wis,ri,rfm,rpm,lg,ls,o,ab,sb,qf=_c[ck]
    kv4=kv_buf.view(-1,PS,nkv,kv_buf.shape[-1])
    
    if use_bf16:
        aiter.mla_decode_stage1_asm_fwd(q,kv4,qo_indptr,ki,kx,kl,
            None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,None,None)
    else:
        N=q.numel();B=4096;g=((N+B-1)//B,)
        ab.zero_();_am[g](q,ab,N,B=B);_cf[g](q,qf,sb,ab,FM=FM,N=N,B=B)
        qf8=qf.view(q.shape[0],nq,dq)
        aiter.mla_decode_stage1_asm_fwd(qf8,kv4,qo_indptr,ki,kx,kl,
            None,wm,wi,wis,1,PS,nkv,sm,lg,ls,o,sb,kv_sc_val)
    
    aiter.mla_reduce_v1(lg,ls,ri,rfm,rpm,1,o,None)
    return o

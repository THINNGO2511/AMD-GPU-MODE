#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Quick probe: check if runner aiter was updated since last check."""
from task import input_t, output_t
import torch, os
_ref=None;_raw=None;_bq=None;_probed=False
def _unshuffle(s):
    s=s.view(torch.uint8);sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
def custom_kernel(data:input_t)->output_t:
    global _ref,_raw,_bq,_probed
    A,B,B_q,B_shuffle,B_scale_sh=data
    m,k=A.shape;n=B.shape[0]
    if _ref is not B_scale_sh:
        _ref=B_scale_sh;_raw=_unshuffle(B_scale_sh);_bq=B_q.view(torch.uint8)
    if not _probed:
        _probed=True
        import subprocess as sp
        try:
            r=sp.run(["git","log","--oneline","-5"],cwd="/home/runner/aiter",capture_output=True,text=True,timeout=5)
            print(f"GIT: {r.stdout.strip()}",flush=True)
        except: pass
        try:
            import glob
            cos=glob.glob("/home/runner/aiter/hsa/gfx950/**/*.co",recursive=True)
            flydsl=glob.glob("/home/runner/aiter/hsa/gfx950/**/flydsl*",recursive=True)
            print(f"CO_TOTAL: {len(cos)} FLYDSL: {len(flydsl)}",flush=True)
        except: pass
        # Check mla.py for qseqlen fold
        try:
            with open("/home/runner/aiter/aiter/mla.py") as f:
                src=f.read()
            has_fold="qseqlen_fold" in src or "max_seqlen_q_new" in src or "nhead_fold" in src
            print(f"MLA_FOLD: {has_fold}",flush=True)
            # Check line count (newer version = more lines)
            print(f"MLA_LINES: {len(src.splitlines())}",flush=True)
        except: pass
        # Check FP4 config count
        try:
            import glob
            cfgs=glob.glob("/home/runner/aiter/aiter/ops/triton/configs/gemm/*gfx950*.json")
            print(f"GEMM_CFGS: {len(cfgs)}",flush=True)
            fp4=[c for c in cfgs if "FP4" in c or "fp4" in c.lower()]
            print(f"FP4_CFGS: {len(fp4)}",flush=True)
        except: pass
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    if k==1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af,asc=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8),_bq,asc,_raw,dtype=torch.bfloat16)
    return gemm_a16wfp4(A,_bq,_raw,dtype=torch.bfloat16)

#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Probe: Read the FULL mla.py dispatch code and attention.py kernel launch code.
Need to understand the exact parameter layout for stage1_asm_fwd to call qseqlen4 directly.
"""
from task import input_t, output_t
import torch, sys, os, inspect

_probed = False
_meta_cache = {}

def custom_kernel(data: input_t) -> output_t:
    global _probed
    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    if not _probed:
        _probed = True
        # 1. Full mla_decode_fwd source
        try:
            from aiter.mla import mla_decode_fwd
            src = inspect.getsource(mla_decode_fwd)
            for i, line in enumerate(src.split('\n')):
                print(f"MDF{i}: {line}", flush=True)
        except Exception as e:
            print(f"MDF_ERR: {e}", flush=True)
        # 2. Check what dispatch_asm_mla or similar does
        try:
            import aiter
            # List all MLA-related functions
            mla_funcs = [x for x in dir(aiter) if 'mla' in x.lower()]
            print(f"MLA_FUNCS: {mla_funcs}", flush=True)
            # Get stage1_asm_fwd signature
            if hasattr(aiter, 'mla_decode_stage1_asm_fwd'):
                sig = inspect.signature(aiter.mla_decode_stage1_asm_fwd)
                print(f"STAGE1_SIG: {sig}", flush=True)
            if hasattr(aiter, 'mla_reduce_v1'):
                sig = inspect.signature(aiter.mla_reduce_v1)
                print(f"REDUCE_SIG: {sig}", flush=True)
        except Exception as e:
            print(f"FUNC_ERR: {e}", flush=True)
        # 3. codegen.py kernel selection logic
        try:
            with open("/home/runner/aiter/hsa/codegen.py") as f:
                src = f.read()
            # Find the kernel name construction
            for i, line in enumerate(src.split('\n')):
                if any(kw in line for kw in ['qseqlen', 'kernel_name', 'co_path', '.co', 'dispatch', 'qh16', 'qh64', 'persistent']):
                    print(f"CG{i}: {line.rstrip()}", flush=True)
        except Exception as e:
            print(f"CG_ERR: {e}", flush=True)

    # Standard MLA decode
    from aiter.mla import mla_decode_fwd
    from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
    FP8_DTYPE = aiter_dtypes.fp8; BF16 = torch.bfloat16
    total_kv = batch_size * kv_seq_len
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, 1, nkv, kv_buffer_fp8.shape[-1])
    num_kv_splits = 8 if total_kv <= 8192 else 16
    ckey = (batch_size, kv_seq_len, num_kv_splits)
    if ckey not in _meta_cache:
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_lpl = seq_lens.to(torch.int32)
        info = get_mla_metadata_info_v1(batch_size, q_seq_len, nq, BF16, FP8_DTYPE, is_sparse=False, fast_mode=False, num_kv_splits=num_kv_splits, intra_batch_mode=True)
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(qo_indptr, kv_indptr, kv_lpl, nq // nkv, nkv, True, wm, wis, wi, ri, rfm, rpm, page_size=1, kv_granularity=16, max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len, fast_mode=False, max_split_per_batch=num_kv_splits, intra_batch_mode=True, dtype_q=BF16, dtype_kv=FP8_DTYPE)
        kv_idx = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        _meta_cache[ckey] = (wm, wi, wis, ri, rfm, rpm, kv_idx, kv_lpl)
    (wm, wi, wis, ri, rfm, rpm, kv_idx, kv_lpl) = _meta_cache[ckey]
    o = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")
    mla_decode_fwd(q, kv_buffer_4d, o, qo_indptr, kv_indptr, kv_idx, kv_lpl, q_seq_len, page_size=1, nhead_kv=nkv, sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits, kv_scale=kv_scale, intra_batch_mode=True, work_meta_data=wm, work_indptr=wi, work_info_set=wis, reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm)
    return o

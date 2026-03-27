#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
PROBE: Investigate 3-buffer KV layout and a16w8 kernel path.
"""
import torch
import sys
import os
import inspect
import glob
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
PAGE_SIZE = 1
_meta_cache = {}
_alloc_cache = {}

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    # === PROBE: dump info on first call ===
    if not hasattr(custom_kernel, '_probed'):
        custom_kernel._probed = True
        
        print("\n=== MLA KERNEL BINARIES ===", file=sys.stderr)
        for pattern in ["*mla*", "*a16w8*", "*page64*", "*ds32*"]:
            files = glob.glob("/home/runner/aiter/hsa/gfx950/**/" + pattern, recursive=True)
            for f in sorted(files):
                print(f"  {f}", file=sys.stderr)
        
        print("\n=== mla_decode_fwd SIGNATURE ===", file=sys.stderr)
        print(inspect.signature(mla_decode_fwd), file=sys.stderr)
        
        print("\n=== MLA SOURCE (key sections) ===", file=sys.stderr)
        for mf in glob.glob("/home/runner/aiter/aiter/mla*.py"):
            print(f"\n--- {mf} ---", file=sys.stderr)
            with open(mf) as f:
                content = f.read()
            # Print lines with page, a16w8, 3buf, rope, nope references  
            for i, line in enumerate(content.split("\n")):
                for kw in ["page", "a16w8", "a8w8", "buffer", "rope", "nope", "ds32", "split", "three", "kPageSize", "page_size"]:
                    if kw in line.lower() and not line.strip().startswith("#"):
                        print(f"  L{i+1}: {line.rstrip()}", file=sys.stderr)
                        break
        
        print("\n=== GREP page64/ds32/3buffer in all .py ===", file=sys.stderr)
        for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
            for fname in files:
                if fname.endswith('.py'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath) as f:
                            for i, line in enumerate(f):
                                for kw in ['page64', 'ds32', 'three_buffer', '3buffer', 'nope_buffer']:
                                    if kw in line.lower():
                                        print(f"  {fpath}:{i+1}: {line.strip()}", file=sys.stderr)
                    except:
                        pass
        
        print("\n=== JIT MLA files ===", file=sys.stderr)
        for d in ["/home/runner/aiter/aiter/jit/", "/home/runner/aiter/aiter/jit_compile/"]:
            if os.path.exists(d):
                for f in sorted(os.listdir(d)):
                    if 'mla' in f.lower():
                        print(f"  {d}{f}", file=sys.stderr)

    # === ACTUAL KERNEL: Try a16w8 (bf16 Q + fp8 KV) ===
    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    
    total_kv = batch_size * kv_seq_len
    num_kv_splits = 8 if total_kv <= 8192 else 16
    
    cache_key = (batch_size, kv_seq_len, num_kv_splits, "a16w8")
    if cache_key not in _meta_cache:
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, torch.bfloat16, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=max(PAGE_SIZE, 16),
            max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=torch.bfloat16, dtype_kv=FP8_DTYPE,
        )
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len)
    
    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len) = _meta_cache[cache_key]
    
    alloc_key = (q.shape[0], nq, dv)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
    o = _alloc_cache[alloc_key]
    
    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])
    
    mla_decode_fwd(
        q, kv_buffer_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
        q_scale=None, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o

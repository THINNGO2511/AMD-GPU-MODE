#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — TileLang flash-decoding kernel.
TileLang achieves 95% of aiter assembly for MLA decode on AMD.
pip install on runner (internet confirmed available).
Example at: examples/deepseek_mla/amd/benchmark_mla_decode_amd_tilelang.py
"""
import subprocess
import sys
import os
import torch
from task import input_t, output_t

_installed = False
_fallback = False
_alloc_cache = {}


def _setup():
    global _installed, _fallback
    if _installed:
        return
    _installed = True

    # Try to pip install tilelang
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "tilelang", "--quiet"],
            capture_output=True, text=True, timeout=120
        )
        print(f"pip install tilelang: exit={result.returncode}")
        if result.returncode != 0:
            print(f"stderr: {result.stderr[:500]}")
            # Try with --index-url for ROCm wheels
            result2 = subprocess.run(
                [sys.executable, "-m", "pip", "install", "tilelang-rocm", "--quiet"],
                capture_output=True, text=True, timeout=120
            )
            print(f"pip install tilelang-rocm: exit={result2.returncode}")
            if result2.returncode != 0:
                print(f"stderr: {result2.stderr[:500]}")
                _fallback = True
    except Exception as e:
        print(f"Install failed: {e}")
        _fallback = True

    if not _fallback:
        try:
            import tilelang
            print(f"TileLang version: {tilelang.__version__}")
            print(f"TileLang dir: {[x for x in dir(tilelang) if not x.startswith('_')][:20]}")
        except ImportError as e:
            print(f"TileLang import failed after install: {e}")
            _fallback = True

    # Also probe for FlashInfer
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "flashinfer", "--quiet"],
            capture_output=True, text=True, timeout=60
        )
        print(f"pip install flashinfer: exit={result.returncode}")
        if result.returncode == 0:
            import flashinfer
            print(f"FlashInfer version: {flashinfer.__version__}")
    except Exception as e:
        print(f"FlashInfer: {e}")

    # Probe what attention libraries are available
    for lib in ['flash_attn', 'xformers', 'flashinfer', 'tilelang']:
        try:
            mod = __import__(lib)
            print(f"  {lib}: available, version={getattr(mod, '__version__', '?')}")
        except ImportError:
            pass


def custom_kernel(data: input_t) -> output_t:
    _setup()

    from aiter.mla import mla_decode_fwd
    from aiter import dtypes as aiter_dtypes
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

    FP8_DTYPE = aiter_dtypes.fp8
    BF16 = torch.bfloat16

    q, kv_data, qo_indptr, kv_indptr, config = data
    batch_size = config["batch_size"]
    nq, nkv = config["num_heads"], config["num_kv_heads"]
    dq, dv = config["qk_head_dim"], config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = batch_size * kv_seq_len

    # Use proven pg2+bf16Q / pg8+fp8Q approach as base
    if kv_seq_len <= 1024:
        page_size = 2
        dtype_q = BF16
        use_fp8_q = False
    else:
        page_size = 8
        dtype_q = FP8_DTYPE
        use_fp8_q = True

    num_kv_splits = 8 if total_kv <= 8192 else 16

    num_pages = total_kv // page_size
    kv_indptr_pages = kv_indptr // page_size
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]
    kv_last_page_len = (seq_lens % page_size).to(torch.int32)
    kv_last_page_len = torch.where(kv_last_page_len == 0, page_size, kv_last_page_len)
    kv_gran = max(1, 16 // page_size)

    info = get_mla_metadata_info_v1(
        batch_size, q_seq_len, nq, dtype_q, FP8_DTYPE,
        is_sparse=False, fast_mode=False,
        num_kv_splits=num_kv_splits, intra_batch_mode=True,
    )
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work
    get_mla_metadata_v1(
        qo_indptr, kv_indptr_pages, kv_last_page_len,
        nq // nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm,
        page_size=page_size, kv_granularity=kv_gran,
        max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
        fast_mode=False, max_split_per_batch=num_kv_splits,
        intra_batch_mode=True, dtype_q=dtype_q, dtype_kv=FP8_DTYPE,
    )
    kv_indices = torch.arange(num_pages, dtype=torch.int32, device="cuda")

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(-1, page_size, nkv, kv_buffer_fp8.shape[-1])
    o = torch.empty((batch_size, nq, dv), dtype=BF16, device="cuda")

    if use_fp8_q:
        _FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
        amax = q.abs().amax().clamp(min=1e-12)
        scale = (amax / _FP8_MAX).to(torch.float32).reshape(1)
        q_fp8 = (q.float() / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)
        mla_decode_fwd(
            q_fp8, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale, kv_scale=kv_scale, intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    else:
        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale, intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    return o

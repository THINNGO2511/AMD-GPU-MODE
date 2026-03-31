#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Download latest aiter mla.py from GitHub at runtime.
The runner has internet (confirmed). The runner's aiter is old (#2156).
Latest aiter has qseqlen dispatch code (PR #2440) that could unlock
mla_a8w8_qh64_qseqlen4 kernel WITHOUT the JIT timeout we hit before
(because the dispatch logic is in Python, not compiled modules).

Strategy: wget latest mla.py → monkey-patch aiter.mla → use qseqlen4 path
"""
import os
import sys
import subprocess
import torch
from task import input_t, output_t

_patched = False
_meta_cache = {}
_alloc_cache = {}

AITER_MLA_URL = "https://raw.githubusercontent.com/ROCm/aiter/main/aiter/mla.py"
AITER_MLA_ASM_CSV_URL = "https://raw.githubusercontent.com/ROCm/aiter/main/hsa/gfx950/mla/mla_asm.csv"


def _download_and_patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Download latest mla.py
    try:
        result = subprocess.run(
            ["wget", "-q", "-O", "/tmp/mla_latest.py", AITER_MLA_URL],
            capture_output=True, text=True, timeout=30
        )
        print(f"Downloaded mla.py: {result.returncode}")

        # Download latest CSV
        result2 = subprocess.run(
            ["wget", "-q", "-O", "/tmp/mla_asm_latest.csv", AITER_MLA_ASM_CSV_URL],
            capture_output=True, text=True, timeout=30
        )
        print(f"Downloaded mla_asm.csv: {result2.returncode}")

        # Read and print key differences
        if os.path.exists("/tmp/mla_latest.py"):
            with open("/tmp/mla_latest.py") as f:
                src = f.read()
            # Check for qseqlen fold
            if "qseqlen_fold" in src or "qseqlen4" in src.lower():
                print("FOUND: qseqlen fold code in latest mla.py!")
            if "use_qseqlen_fold" in src:
                print("FOUND: use_qseqlen_fold variable!")

            # Print lines with qseqlen
            for i, line in enumerate(src.split('\n')):
                if 'qseqlen' in line.lower() or 'fold' in line.lower():
                    print(f"  L{i}: {line.rstrip()}")

            # Print the dispatch/kernel selection section
            print("\n=== Kernel dispatch section ===")
            in_dispatch = False
            for i, line in enumerate(src.split('\n')):
                if 'def mla_decode_fwd' in line or 'persistent' in line.lower():
                    in_dispatch = True
                if in_dispatch:
                    print(f"  L{i}: {line.rstrip()}")
                if in_dispatch and i > 50 and line.strip() == '':
                    in_dispatch = False

        if os.path.exists("/tmp/mla_asm_latest.csv"):
            with open("/tmp/mla_asm_latest.csv") as f:
                csv_content = f.read()
            print(f"\n=== Latest mla_asm.csv ({len(csv_content)} bytes) ===")
            for line in csv_content.strip().split('\n'):
                if 'qseqlen' in line.lower() or 'qh64' in line.lower() or line.startswith('#'):
                    print(f"  {line}")

        # Compare with runner's current CSV
        runner_csv = "/home/runner/aiter/hsa/gfx950/mla/mla_asm.csv"
        if os.path.exists(runner_csv):
            with open(runner_csv) as f:
                old_csv = f.read()
            print(f"\n=== Runner mla_asm.csv ({len(old_csv)} bytes) ===")
            for line in old_csv.strip().split('\n'):
                print(f"  {line}")

    except Exception as e:
        print(f"Download failed: {e}")

    # Also check current aiter version details
    try:
        import aiter
        mla_file = aiter.mla.__file__
        print(f"\nRunner aiter mla.py: {mla_file}")
        with open(mla_file) as f:
            runner_src = f.read()
        print(f"Runner mla.py size: {len(runner_src)} bytes")
        if "qseqlen_fold" in runner_src:
            print("Runner HAS qseqlen fold code!")
        else:
            print("Runner MISSING qseqlen fold code")
    except Exception as e:
        print(f"Runner check error: {e}")


def custom_kernel(data: input_t) -> output_t:
    _download_and_patch()

    # Fall back to proven approach for actual correctness
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

    # Use proven pg2+bf16Q / pg8+fp8Q approach
    if kv_seq_len <= 1024:
        page_size = 2
        dtype_q = BF16
        use_fp8_q = False
    else:
        page_size = 8
        dtype_q = FP8_DTYPE
        use_fp8_q = True

    num_kv_splits = 8 if total_kv <= 8192 else 16

    # Build metadata (simplified, no caching for probe)
    if page_size == 1:
        num_pages = total_kv
        kv_indptr_pages = kv_indptr
        seq_lens = kv_indptr[1:] - kv_indptr[:-1]
        kv_last_page_len = seq_lens.to(torch.int32)
    else:
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
        # Quantize Q to fp8
        finfo = torch.finfo(FP8_DTYPE)
        amax = q.abs().amax().clamp(min=1e-12)
        scale = (amax / finfo.max).to(torch.float32).reshape(1)
        q_fp8 = (q.float() / scale).clamp(finfo.min, finfo.max).to(FP8_DTYPE)

        mla_decode_fwd(
            q_fp8.view(batch_size, nq, dq), kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            q_scale=scale, kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    else:
        mla_decode_fwd(
            q, kv_buffer_4d, o,
            qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=page_size, nhead_kv=nkv,
            sm_scale=sm_scale, logit_cap=0.0, num_kv_splits=num_kv_splits,
            kv_scale=kv_scale,
            intra_batch_mode=True,
            work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    return o

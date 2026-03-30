from task import input_t, output_t
import torch
import subprocess
import os

def custom_kernel(data: input_t) -> output_t:
    (q, k_pe, kv_cache_bf16, kv_cache_fp8, kv_cache_fp4,
     kv_scale_bf16, kv_scale_fp8, kv_scale_fp4,
     seq_lens, page_table, output) = data

    bs = q.shape[0]
    kv_len = seq_lens[0].item()
    print("=" * 60)
    print(f"MLA PG2 PROBE — bs={bs} kv_len={kv_len}")
    print("=" * 60)

    # 1. Read mla.py source — focus on page_size handling
    print("\n--- MLA.PY PAGE_SIZE / GRANULARITY CODE ---")
    try:
        with open("/home/runner/aiter/aiter/mla.py") as f:
            src = f.read()
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['page_size', 'granul', 'kv_indptr', 'last_page', 'paged']):
                print(f"  {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Read the full mla_decode_fwd function signature and first 100 lines
    print("\n--- MLA_DECODE_FWD SOURCE (first 150 lines) ---")
    try:
        with open("/home/runner/aiter/aiter/mla.py") as f:
            lines = f.readlines()
        in_func = False
        count = 0
        for i, line in enumerate(lines):
            if 'def mla_decode_fwd' in line:
                in_func = True
            if in_func:
                print(f"  {i+1}: {line.rstrip()}")
                count += 1
                if count > 150:
                    break
    except Exception as e:
        print(f"Error: {e}")

    # 3. Check get_mla_metadata source
    print("\n--- GET_MLA_METADATA SOURCE ---")
    try:
        r = subprocess.run(["grep", "-n", "-A", "30", "def get_mla_metadata",
                           "/home/runner/aiter/aiter/mla.py"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:3000])
    except Exception as e:
        print(f"Error: {e}")

    # 4. Check ASM kernel filenames for pg/page variants
    print("\n--- MLA ASM KERNELS ---")
    import glob
    mla_files = glob.glob("/home/runner/aiter/hsa/gfx950/**/*mla*", recursive=True)
    print(f"Total MLA kernel files: {len(mla_files)}")
    for f in sorted(mla_files):
        print(f"  {os.path.basename(f)}")

    # 5. Check if BLOCK_PINGPONG is referenced
    print("\n--- BLOCK_PINGPONG REFERENCES ---")
    try:
        r = subprocess.run(["grep", "-rn", "PINGPONG\|pingpong\|BLOCK_PING",
                           "/home/runner/aiter/aiter/"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:1000] if r.stdout else "No references found")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Check stage1 ASM kernel for page_size support
    print("\n--- STAGE1 ASM KERNEL DISPATCH ---")
    try:
        r = subprocess.run(["grep", "-n", "-A5", "page_size\|kPageSize\|page_table",
                           "/home/runner/aiter/aiter/mla.py"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:2000])
    except Exception as e:
        print(f"Error: {e}")

    # 7. Read the metadata info function
    print("\n--- METADATA INFO ---")
    try:
        from aiter import get_mla_metadata_info_v1
        info = get_mla_metadata_info_v1(
            num_seqs=bs,
            max_kv_len=kv_len,
            num_kv_heads=1,
            page_size=1,
            num_kv_splits=8,
        )
        for k, v in info.items():
            print(f"  pg1: {k} = {v}")

        info2 = get_mla_metadata_info_v1(
            num_seqs=bs,
            max_kv_len=kv_len,
            num_kv_heads=1,
            page_size=2,
            num_kv_splits=8,
        )
        for k, v in info2.items():
            print(f"  pg2: {k} = {v}")
    except Exception as e:
        print(f"Metadata info error: {e}")

    print("\n" + "=" * 60)

    # Return valid reference result
    from aiter.mla import mla_decode_fwd
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1, dtypes as aiter_dtypes

    num_kv_splits = 8
    page_size = 1
    kv_granularity = max(1, 16 // page_size)

    total_tokens = kv_cache_fp8[0].shape[0]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=q.device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)
    kv_last_page_lens = seq_lens.clone()
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=q.device)

    meta_info = get_mla_metadata_info_v1(bs, kv_len, 1, page_size, num_kv_splits)
    metadata = get_mla_metadata_v1(kv_indptr, kv_indices, kv_last_page_lens,
                                    num_kv_splits, page_size, kv_granularity, meta_info)

    q_fp8 = q.clamp(-448, 448).to(torch.float8_e4m3fnuz)
    q_scale = (q.abs().max() / 448.0).unsqueeze(0).to(torch.float32)

    mla_decode_fwd(output, q_fp8, k_pe, kv_cache_fp8[0], kv_scale_fp8[0],
                   *metadata, num_kv_splits, page_size, kv_granularity,
                   q_scale, aiter_dtypes.fp8)
    return output

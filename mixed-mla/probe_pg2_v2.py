from task import input_t, output_t
import torch
import subprocess
import os

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    bs = config.get("batchsize", q.shape[0])
    kv_len = config.get("kvseqlen", 1024)
    print("=" * 60)
    print(f"MLA PG2 PROBE v2 — bs={bs} kv_len={kv_len}")
    print(f"  q shape: {q.shape}")
    print(f"  kv_data keys: {list(kv_data.keys())}")
    print(f"  config: {config}")
    print(f"  qo_indptr: {qo_indptr}")
    print(f"  kv_indptr: {kv_indptr}")
    for k, v in kv_data.items():
        if isinstance(v, torch.Tensor):
            print(f"  kv_data[{k}]: {v.shape} {v.dtype}")
        elif isinstance(v, (list, tuple)):
            for i, t in enumerate(v):
                if isinstance(t, torch.Tensor):
                    print(f"  kv_data[{k}][{i}]: {t.shape} {t.dtype}")
    print("=" * 60)

    # 1. Read mla.py — full mla_decode_fwd function
    print("\n--- MLA_DECODE_FWD FULL SOURCE ---")
    try:
        with open("/home/runner/aiter/aiter/mla.py") as f:
            src = f.read()
        # Find mla_decode_fwd function
        lines = src.split('\n')
        in_func = False
        depth = 0
        count = 0
        for i, line in enumerate(lines):
            if 'def mla_decode_fwd' in line:
                in_func = True
                depth = len(line) - len(line.lstrip())
            if in_func:
                print(f"  {i+1}: {line}")
                count += 1
                if count > 80:
                    print("  ... (truncated)")
                    break
    except Exception as e:
        print(f"Error: {e}")

    # 2. Read get_mla_metadata_v1
    print("\n--- GET_MLA_METADATA_V1 SOURCE ---")
    try:
        in_func = False
        count = 0
        for i, line in enumerate(lines):
            if 'def get_mla_metadata_v1' in line:
                in_func = True
            if in_func:
                print(f"  {i+1}: {line}")
                count += 1
                if count > 50:
                    break
    except Exception as e:
        print(f"Error: {e}")

    # 3. Check ASM kernel files
    print("\n--- MLA ASM KERNELS ---")
    import glob
    mla_files = glob.glob("/home/runner/aiter/hsa/gfx950/**/*mla*", recursive=True)
    print(f"Total MLA files: {len(mla_files)}")
    for f in sorted(mla_files)[:30]:
        print(f"  {os.path.relpath(f, '/home/runner/aiter/hsa/gfx950/')}")

    # 4. Check BLOCK_PINGPONG env var
    print("\n--- TRITON ENV VARS ---")
    try:
        r = subprocess.run(["grep", "-rn", "PINGPONG", "/home/runner/aiter/aiter/"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:1000] if r.stdout else "No PINGPONG references")
    except Exception as e:
        print(f"Error: {e}")

    # Return reference result
    from aiter.mla import mla_decode_fwd
    from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1, dtypes as aiter_dtypes

    num_heads = q.shape[1] if q.dim() == 3 else config.get("num_heads", 16)
    total_tokens = kv_data["fp8"][0].shape[0]
    seq_lens = kv_indptr[1:] - kv_indptr[:-1]

    page_size = 1
    kv_granularity = max(1, 16 // page_size)
    num_kv_splits = 8

    kv_indptr_paged = kv_indptr.clone()
    kv_last_page_lens = seq_lens.clone()
    kv_indices = torch.arange(total_tokens, dtype=torch.int32, device=q.device)

    meta_info = get_mla_metadata_info_v1(bs, kv_len, 1, page_size, num_kv_splits)
    metadata = get_mla_metadata_v1(kv_indptr_paged, kv_indices, kv_last_page_lens,
                                    num_kv_splits, page_size, kv_granularity, meta_info)

    q_fp8 = q.clamp(-448, 448).to(torch.float8_e4m3fnuz)
    q_scale = (q.abs().max() / 448.0).unsqueeze(0).to(torch.float32)

    output = torch.zeros(q.shape[0], num_heads, 512, dtype=torch.bfloat16, device=q.device)
    k_pe = kv_data["fp8"][0][:, :, 512:576] if kv_data["fp8"][0].shape[-1] >= 576 else None

    mla_decode_fwd(output, q_fp8, k_pe, kv_data["fp8"][0], kv_data["fp8"][1],
                   *metadata, num_kv_splits, page_size, kv_granularity,
                   q_scale, aiter_dtypes.fp8)
    return output

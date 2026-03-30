#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Probe: Check if mxfp4 KV kernel exists and test it.
mxfp4 = 2x less bandwidth than fp8. Could be huge for decode.
Also list all .co files and check for page64 variants.
"""
import torch
import os
import inspect
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    print("=" * 80)
    print("PROBE: mxfp4 KV + all kernel binaries")
    print("=" * 80)

    # 1. List ALL .co kernel files in mla directory
    try:
        mla_dir = "/home/runner/aiter/hsa/gfx950/mla/"
        print(f"\n--- All .co files in {mla_dir} ---")
        if os.path.exists(mla_dir):
            for f in sorted(os.listdir(mla_dir)):
                if f.endswith('.co'):
                    fsize = os.path.getsize(os.path.join(mla_dir, f))
                    print(f"  {f} ({fsize} bytes)")
        else:
            print(f"  Directory not found!")
            # Search broader
            for d in ["/home/runner/aiter/hsa/gfx950/", "/home/runner/aiter/hsa/"]:
                if os.path.exists(d):
                    print(f"\n  Listing {d}:")
                    for root, dirs, files in os.walk(d):
                        for f in sorted(files):
                            if f.endswith('.co') and 'mla' in f.lower():
                                print(f"    {os.path.join(root, f)}")
    except Exception as e:
        print(f"Error: {e}")

    # 2. Check mxfp4 KV data format
    try:
        print(f"\n--- mxfp4 KV data ---")
        mxfp4_data = kv_data["mxfp4"]
        mxfp4_buf, mxfp4_scale = mxfp4_data
        print(f"  Buffer: dtype={mxfp4_buf.dtype}, shape={mxfp4_buf.shape}, stride={mxfp4_buf.stride()}")
        print(f"  Scale:  dtype={mxfp4_scale.dtype}, shape={mxfp4_scale.shape}, stride={mxfp4_scale.stride()}")
        print(f"  Buffer is contiguous: {mxfp4_buf.is_contiguous()}")
        print(f"  Buffer device: {mxfp4_buf.device}")
        # Check if it's uint8 (fp4x2 packed)
        print(f"  Buffer[0,:3,:5]: {mxfp4_buf[0,:3,:5] if mxfp4_buf.numel() > 0 else 'empty'}")
        print(f"  Scale[0:5]: {mxfp4_scale.flatten()[:5] if mxfp4_scale.numel() > 0 else 'empty'}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Check fp8 KV data for comparison
    try:
        print(f"\n--- fp8 KV data ---")
        fp8_buf, fp8_scale = kv_data["fp8"]
        print(f"  Buffer: dtype={fp8_buf.dtype}, shape={fp8_buf.shape}")
        print(f"  Scale:  dtype={fp8_scale.dtype}, shape={fp8_scale.shape}")
    except Exception as e:
        print(f"Error: {e}")

    # 4. Check codegen.py for kernel dispatch logic (what page sizes are supported)
    try:
        codegen_path = "/home/runner/aiter/hsa/codegen.py"
        print(f"\n--- codegen.py: page_size and kernel selection ---")
        if os.path.exists(codegen_path):
            with open(codegen_path) as f:
                content = f.read()
            for kw in ['page_size', 'kPageSize', 'mxfp4', 'uint8', 'fp4', 'w4', 'a8w4', 'page64']:
                matches = [(i+1, l.strip()) for i, l in enumerate(content.split('\n')) if kw.lower() in l.lower()]
                if matches:
                    print(f"\n  '{kw}' ({len(matches)} matches):")
                    for ln, lt in matches[:8]:
                        print(f"    L{ln}: {lt}")
    except Exception as e:
        print(f"Error: {e}")

    # 5. Check if there are any page_size related configs in the hsa directory
    try:
        print(f"\n--- CSV configs in hsa/gfx950/mla/ ---")
        config_dirs = [
            "/home/runner/aiter/hsa/gfx950/mla/",
            "/home/runner/aiter/hsa/gfx950/",
            "/home/runner/aiter/hsa/",
        ]
        for cdir in config_dirs:
            if os.path.exists(cdir):
                for f in sorted(os.listdir(cdir)):
                    if f.endswith(('.csv', '.json', '.yaml', '.toml', '.txt', '.cfg')):
                        fpath = os.path.join(cdir, f)
                        print(f"\n  {fpath}:")
                        with open(fpath) as fh:
                            content = fh.read()[:2000]
                            print(f"    {content}")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Try calling mla_decode_stage1_asm_fwd with mxfp4 kv to see error message
    try:
        import aiter
        from aiter import dtypes as aiter_dtypes
        from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

        FP8_DTYPE = aiter_dtypes.fp8
        batch_size = config["batch_size"]
        nq = config["num_heads"]
        nkv = config["num_kv_heads"]
        dv = config["v_head_dim"]
        dq = config["qk_head_dim"]
        q_seq_len = config["q_seq_len"]
        sm_scale = config["sm_scale"]
        kv_seq_len = config["kv_seq_len"]

        print(f"\n--- Attempting mxfp4 KV decode ---")
        mxfp4_buf, mxfp4_scale = kv_data["mxfp4"]

        # Try reshaping mxfp4 for 4D
        print(f"  mxfp4_buf shape before reshape: {mxfp4_buf.shape}")
        # mxfp4 is (total_kv, 1, 288) uint8 → try (total_kv, 1, 1, 288)
        mxfp4_4d = mxfp4_buf.view(mxfp4_buf.shape[0], 1, nkv, mxfp4_buf.shape[-1])
        print(f"  mxfp4_4d shape: {mxfp4_4d.shape}")

        # Try with fp8 Q + mxfp4 KV
        amax = q.abs().amax().clamp(min=1e-12)
        q_scale = (amax / float(torch.finfo(FP8_DTYPE).max)).to(torch.float32).reshape(1)
        q_fp8 = (q / q_scale).clamp(-float(torch.finfo(FP8_DTYPE).max), float(torch.finfo(FP8_DTYPE).max)).to(FP8_DTYPE)

        total_kv = batch_size * kv_seq_len
        o = torch.zeros((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")
        logits = torch.empty((q.shape[0], 1, nq, dv), dtype=torch.float32, device="cuda")
        attn_lse = torch.empty((q.shape[0], 1, nq, 1), dtype=torch.float32, device="cuda")

        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        num_kv_splits_indptr = torch.arange(batch_size + 1, dtype=torch.int32, device="cuda")

        try:
            aiter.mla_decode_stage1_asm_fwd(
                q_fp8,
                mxfp4_4d,
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_kv_splits_indptr,
                None, None, None,
                q_seq_len, 1, nkv, sm_scale,
                logits, attn_lse, o,
                q_scale, mxfp4_scale,
            )
            print(f"  SUCCESS! mxfp4 KV decode worked!")
            print(f"  Output sample: {o[0, 0, :5]}")
        except Exception as e:
            print(f"  FAILED: {e}")

    except Exception as e:
        print(f"Error in mxfp4 test: {e}")

    # 7. Check get_meta_param source
    try:
        from aiter.mla import get_meta_param
        print(f"\n--- get_meta_param source ---")
        src = inspect.getsource(get_meta_param)
        print(src[:1500])
    except Exception as e:
        print(f"Error: {e}")

    # Produce valid output
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    return torch.zeros((q.shape[0], nq, dv), dtype=torch.bfloat16, device=q.device)

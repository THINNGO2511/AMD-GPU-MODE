#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA MXFP4 KV probe — explore the mxfp4 KV data format and test bandwidth savings.
Goal: understand tensor shapes, scale layout, and whether Triton tl.dot_scaled can work.
"""
import torch
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
BF16 = torch.bfloat16
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
        # Probe all three KV formats
        kv_bf16 = kv_data["bf16"]
        kv_fp8, kv_fp8_scale = kv_data["fp8"]
        kv_mxfp4, kv_mxfp4_scale = kv_data["mxfp4"]

        print(f"[MXFP4] bf16 KV: {kv_bf16.shape} {kv_bf16.dtype}")
        print(f"[MXFP4] fp8 KV: {kv_fp8.shape} {kv_fp8.dtype}, scale: {kv_fp8_scale.shape} {kv_fp8_scale.dtype}")
        print(f"[MXFP4] mxfp4 KV: {kv_mxfp4.shape} {kv_mxfp4.dtype}")
        print(f"[MXFP4] mxfp4 scale: {kv_mxfp4_scale.shape} {kv_mxfp4_scale.dtype}")

        # Bandwidth comparison per token
        bf16_bytes = kv_bf16.shape[-1] * kv_bf16.element_size()
        fp8_bytes = kv_fp8.shape[-1] * kv_fp8.element_size()
        mxfp4_bytes = kv_mxfp4.shape[-1] * kv_mxfp4.element_size()
        mxfp4_scale_bytes = kv_mxfp4_scale.element_size() * (576 // 32)  # 18 scales per token
        print(f"[MXFP4] Bytes per token: bf16={bf16_bytes}, fp8={fp8_bytes}, mxfp4={mxfp4_bytes}+{mxfp4_scale_bytes}={mxfp4_bytes+mxfp4_scale_bytes}")
        print(f"[MXFP4] BW savings: fp8/mxfp4 = {fp8_bytes/(mxfp4_bytes+mxfp4_scale_bytes):.1f}x")

        # Check mxfp4 scale layout
        print(f"[MXFP4] scale strides: {kv_mxfp4_scale.stride()}")
        print(f"[MXFP4] scale numel: {kv_mxfp4_scale.numel()}")
        print(f"[MXFP4] total_kv={batch_size * kv_seq_len}")

        # Test: dequant mxfp4 to bf16 and compare with original
        try:
            from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
            total_kv = batch_size * kv_seq_len
            # Reshape for dequant: (total_kv * nkv, dim/2)
            fp4_2d = kv_mxfp4.view(-1, kv_mxfp4.shape[-1])  # [total_kv, 288]
            f32_vals = mxfp4_to_f32(fp4_2d)  # [total_kv, 576]

            # Apply scales
            scale_f32 = e8m0_to_f32(kv_mxfp4_scale)  # [padded_rows, scale_cols]
            print(f"[MXFP4] dequant f32: {f32_vals.shape}, scale_f32: {scale_f32.shape}")

            num_blocks = 576 // 32  # 18
            scale_trimmed = scale_f32[:total_kv, :num_blocks]
            f32_blocked = f32_vals.view(total_kv, num_blocks, 32)
            f32_scaled = (f32_blocked * scale_trimmed.unsqueeze(-1)).view(total_kv, 576)

            # Compare with bf16 original
            bf16_flat = kv_bf16.view(-1, 576).to(torch.float32)
            diff = (f32_scaled - bf16_flat).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            rel_err = (diff / (bf16_flat.abs() + 1e-8)).mean().item()
            print(f"[MXFP4] Dequant accuracy: max_err={max_err:.4f}, mean_err={mean_err:.6f}, rel_err={rel_err:.6f}")
        except Exception as e:
            print(f"[MXFP4] Dequant test failed: {e}")
            import traceback
            traceback.print_exc()

        # Check if mla_decode_fwd can accept mxfp4 KV directly
        try:
            import inspect
            sig = inspect.signature(mla_decode_fwd)
            print(f"\n[MLA] mla_decode_fwd params: {list(sig.parameters.keys())}")
        except Exception as e:
            print(f"[MLA] sig failed: {e}")

        # Check available MLA .co files for mxfp4
        import os
        mla_dir = "/home/runner/aiter/hsa/gfx950/mla/"
        if os.path.isdir(mla_dir):
            cos = sorted(os.listdir(mla_dir))
            print(f"\n[MLA] .co files ({len(cos)}):")
            for f in cos:
                print(f"  {f}")
                # Check for mxfp4/fp4 related
            fp4_cos = [f for f in cos if 'fp4' in f.lower() or 'mxfp4' in f.lower() or 'a4' in f.lower()]
            if fp4_cos:
                print(f"\n[MLA] FP4/MXFP4 kernels: {fp4_cos}")
            else:
                print(f"\n[MLA] No FP4/MXFP4 ASM kernels found")

    # Use the working fp8 path for actual computation
    total_kv = batch_size * kv_seq_len
    num_kv_splits = 8 if total_kv <= 8192 else 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        kv_buffer_fp8, kv_scale = kv_data["fp8"]
        kv_buffer_4d = kv_buffer_fp8.view(total_kv, 1, nkv, kv_buffer_fp8.shape[-1])
        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, FP8_DTYPE, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work

        kv_indptr_pages = kv_indptr
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)

        get_mla_metadata_v1(
            qo_indptr, kv_indptr_pages, kv_last_page_len,
            nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm,
            page_size=1,
            kv_granularity=16,
            max_seqlen_qo=q_seq_len,
            uni_seqlen_qo=q_seq_len,
            fast_mode=False,
            max_split_per_batch=num_kv_splits,
            intra_batch_mode=True,
            dtype_q=FP8_DTYPE,
            dtype_kv=FP8_DTYPE,
        )
        kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len, kv_indptr_pages) = _meta_cache[cache_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(total_kv, 1, nkv, kv_buffer_fp8.shape[-1])

    # Use fp8 Q path for kv=8192, bf16 Q for kv=1024
    o = torch.empty((q.shape[0], nq, dv), dtype=BF16, device="cuda")
    if kv_seq_len >= 8192:
        finfo = torch.finfo(FP8_DTYPE)
        amax = q.abs().amax().clamp(min=1e-12)
        scale = amax / finfo.max
        q_fp8 = (q / scale).clamp(min=finfo.min, max=finfo.max).to(FP8_DTYPE)
        q_scale = scale.to(torch.float32).reshape(1)
        mla_decode_fwd(
            q_fp8, kv_buffer_4d, o, qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=1, nhead_kv=nkv, sm_scale=sm_scale, logit_cap=0.0,
            num_kv_splits=num_kv_splits, q_scale=q_scale, kv_scale=kv_scale,
            intra_batch_mode=True, work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    else:
        mla_decode_fwd(
            q, kv_buffer_4d, o, qo_indptr, kv_indptr_pages, kv_indices, kv_last_page_len,
            q_seq_len, page_size=1, nhead_kv=nkv, sm_scale=sm_scale, logit_cap=0.0,
            num_kv_splits=num_kv_splits, kv_scale=kv_scale,
            intra_batch_mode=True, work_meta_data=wm, work_indptr=wi, work_info_set=wis,
            reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
        )
    return o

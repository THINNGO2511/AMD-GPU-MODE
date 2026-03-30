#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
MLA — Try passing mxfp4 KV cache to mla_decode_fwd.
The KV data includes mxfp4 format (half the bandwidth of fp8).
If mla_decode_fwd supports mxfp4 KV, this could be 2x faster for large kv_len.
"""
import torch
import triton
import triton.language as tl
from task import input_t, output_t
from aiter.mla import mla_decode_fwd
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

FP8_DTYPE = aiter_dtypes.fp8
FP4_DTYPE = aiter_dtypes.fp4x2
PAGE_SIZE = 1
_FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
_meta_cache = {}
_alloc_cache = {}
_probed = False


@triton.jit
def _q_amax_kernel(q_ptr, amax_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    local_max = tl.max(tl.abs(x))
    tl.atomic_max(amax_ptr, local_max)


@triton.jit
def _q_to_fp8_kernel(q_ptr, out_ptr, scale_ptr, amax_ptr,
                     FP8_MAX: tl.constexpr, N, BLOCK: tl.constexpr):
    amax = tl.load(amax_ptr)
    amax = tl.where(amax < 1e-12, 1e-12, amax)
    scale = amax / FP8_MAX
    if tl.program_id(0) == 0:
        tl.store(scale_ptr, scale)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(q_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = x / scale
    x = tl.clamp(x, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + offs, x.to(out_ptr.dtype.element_ty), mask=mask)


def custom_kernel(data: input_t) -> output_t:
    global _probed
    q, kv_data, qo_indptr, kv_indptr, config = data

    batch_size = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    q_seq_len = config["q_seq_len"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]

    if not _probed:
        _probed = True
        # Probe mxfp4 KV format
        kv_mxfp4, kv_scale_mxfp4 = kv_data["mxfp4"]
        print(f"[MXFP4] kv_buffer: {kv_mxfp4.shape} {kv_mxfp4.dtype}")
        print(f"[MXFP4] kv_scale: {kv_scale_mxfp4.shape} {kv_scale_mxfp4.dtype}")

        kv_fp8, kv_scale_fp8 = kv_data["fp8"]
        print(f"[FP8] kv_buffer: {kv_fp8.shape} {kv_fp8.dtype}")
        print(f"[FP8] kv_scale: {kv_scale_fp8.shape} {kv_scale_fp8.dtype}")

        kv_bf16 = kv_data["bf16"]
        print(f"[BF16] kv_buffer: {kv_bf16.shape} {kv_bf16.dtype}")

        # Probe mla_decode_fwd signature
        import inspect
        try:
            sig = inspect.signature(mla_decode_fwd)
            print(f"\n[SIG] mla_decode_fwd:")
            for name, p in sig.parameters.items():
                defval = p.default if p.default != inspect.Parameter.empty else 'REQUIRED'
                print(f"  {name}: {defval}")
        except:
            pass

        # Try calling with mxfp4 KV
        try:
            total_kv = batch_size * kv_seq_len
            num_kv_splits = 16

            kv_4d_mxfp4 = kv_mxfp4.view(kv_mxfp4.shape[0], PAGE_SIZE, nkv, kv_mxfp4.shape[-1])

            info = get_mla_metadata_info_v1(
                batch_size, q_seq_len, nq, FP8_DTYPE, FP4_DTYPE,
                is_sparse=False, fast_mode=False,
                num_kv_splits=num_kv_splits, intra_batch_mode=True,
            )
            work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
            (wm, wi, wis, ri, rfm, rpm) = work

            kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
            get_mla_metadata_v1(
                qo_indptr, kv_indptr, kv_last_page_len,
                nq // nkv, nkv, True,
                wm, wis, wi, ri, rfm, rpm,
                page_size=PAGE_SIZE, kv_granularity=16,
                max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
                fast_mode=False, max_split_per_batch=num_kv_splits,
                intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=FP4_DTYPE,
            )

            kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
            o_test = torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda")

            # Quantize Q
            amax = q.abs().amax().clamp(min=1e-12)
            scale = (amax / _FP8_MAX).to(torch.float32).reshape(1)
            q_fp8 = (q / scale).clamp(-_FP8_MAX, _FP8_MAX).to(FP8_DTYPE)

            mla_decode_fwd(
                q_fp8, kv_4d_mxfp4, o_test,
                qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
                q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
                sm_scale=sm_scale, logit_cap=0.0,
                num_kv_splits=num_kv_splits,
                q_scale=scale, kv_scale=kv_scale_mxfp4,
                intra_batch_mode=True,
                work_meta_data=wm, work_indptr=wi, work_info_set=wis,
                reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
            )
            print(f"\n[MXFP4_TEST] SUCCESS! output range: [{o_test.min():.4f}, {o_test.max():.4f}]")
        except Exception as e:
            import traceback
            print(f"\n[MXFP4_TEST] FAILED: {e}")
            traceback.print_exc()

    # Standard fp8 path
    total_kv = batch_size * kv_seq_len
    num_kv_splits = 8 if total_kv <= 8192 else 16

    cache_key = (batch_size, kv_seq_len, num_kv_splits)
    if cache_key not in _meta_cache:
        total_kv_len = batch_size * kv_seq_len
        kv_last_page_len = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        info = get_mla_metadata_info_v1(
            batch_size, q_seq_len, nq, FP8_DTYPE, FP8_DTYPE,
            is_sparse=False, fast_mode=False,
            num_kv_splits=num_kv_splits, intra_batch_mode=True,
        )
        work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        (wm, wi, wis, ri, rfm, rpm) = work
        get_mla_metadata_v1(
            qo_indptr, kv_indptr, kv_last_page_len,
            nq // nkv, nkv, True, wm, wis, wi, ri, rfm, rpm,
            page_size=PAGE_SIZE, kv_granularity=16,
            max_seqlen_qo=q_seq_len, uni_seqlen_qo=q_seq_len,
            fast_mode=False, max_split_per_batch=num_kv_splits,
            intra_batch_mode=True, dtype_q=FP8_DTYPE, dtype_kv=FP8_DTYPE,
        )
        kv_indices = torch.arange(total_kv_len, dtype=torch.int32, device="cuda")
        _meta_cache[cache_key] = (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len)

    (wm, wi, wis, ri, rfm, rpm, kv_indices, kv_last_page_len) = _meta_cache[cache_key]

    alloc_key = (q.shape[0], nq, dv, dq)
    if alloc_key not in _alloc_cache:
        _alloc_cache[alloc_key] = (
            torch.empty((q.shape[0], nq, dv), dtype=torch.bfloat16, device="cuda"),
            torch.zeros(1, dtype=torch.float32, device="cuda"),
            torch.empty(1, dtype=torch.float32, device="cuda"),
            torch.empty(q.shape[0] * nq * dq, dtype=FP8_DTYPE, device="cuda"),
        )
    o, amax_buf, scale_buf, q_fp8_flat = _alloc_cache[alloc_key]

    kv_buffer_fp8, kv_scale = kv_data["fp8"]
    kv_buffer_4d = kv_buffer_fp8.view(kv_buffer_fp8.shape[0], PAGE_SIZE, nkv, kv_buffer_fp8.shape[-1])

    N = q.numel()
    BLOCK = 4096
    grid = ((N + BLOCK - 1) // BLOCK,)
    amax_buf.zero_()
    _q_amax_kernel[grid](q, amax_buf, N, BLOCK=BLOCK)
    _q_to_fp8_kernel[grid](q, q_fp8_flat, scale_buf, amax_buf, FP8_MAX=_FP8_MAX, N=N, BLOCK=BLOCK)
    q_fp8 = q_fp8_flat.view(q.shape[0], nq, dq)

    mla_decode_fwd(
        q_fp8, kv_buffer_4d, o,
        qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
        q_seq_len, page_size=PAGE_SIZE, nhead_kv=nkv,
        sm_scale=sm_scale, logit_cap=0.0,
        num_kv_splits=num_kv_splits,
        q_scale=scale_buf, kv_scale=kv_scale,
        intra_batch_mode=True,
        work_meta_data=wm, work_indptr=wi, work_info_set=wis,
        reduce_indptr=ri, reduce_final_map=rfm, reduce_partial_map=rpm,
    )
    return o

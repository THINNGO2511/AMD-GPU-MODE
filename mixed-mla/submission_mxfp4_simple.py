#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""MLA: Simple MXFP4 KV approach — dequant MXFP4 to bf16, then use a16w16 kernel.
Not a custom Triton kernel yet — just tests if MXFP4 dequant + a16w16 is viable."""
import torch
import aiter
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.utility.fp4_utils import mxfp4_to_f32, e8m0_to_f32
from task import input_t, output_t

BF16 = torch.bfloat16
_c = {}

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]  # 576
    dv = config["v_head_dim"]   # 512
    sm = config["sm_scale"]
    kvl = config["kv_seq_len"]
    tkv = bs * kvl

    # Get MXFP4 KV cache
    kv_fp4, kv_scales = kv_data["mxfp4"]
    # kv_fp4: (total_kv, 1, 288) fp4x2
    # kv_scales: (total_kv, 24) fp8_e8m0

    # Dequant MXFP4 to bf16
    # Reshape for dequant: (total_kv * 1, 288) → mxfp4_to_f32 → (total_kv, 576)
    kv_2d = kv_fp4.view(tkv, -1)  # (total_kv, 288)
    kv_f32 = mxfp4_to_f32(kv_2d)  # (total_kv, 576)

    # Apply scales
    num_blocks = dq // 32  # 576/32 = 18
    scales_f32 = e8m0_to_f32(kv_scales)  # (total_kv, 24) → trim to 18
    scales_f32 = scales_f32[:, :num_blocks]  # (total_kv, 18)
    kv_blocked = kv_f32.view(tkv, num_blocks, 32)
    kv_scaled = (kv_blocked * scales_f32.unsqueeze(-1)).view(tkv, dq)
    kv_bf16 = kv_scaled.to(BF16).unsqueeze(1)  # (total_kv, 1, 576)

    # Now use a16w16 kernel (bf16 Q + bf16 KV)
    nks = 8 if tkv <= 8192 else 16
    ck = (bs, kvl)
    if ck not in _c:
        kv_idx = torch.arange(tkv, dtype=torch.int32, device=q.device)
        kv_last = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
        info = get_mla_metadata_info_v1(bs, 1, nq, BF16, BF16,
            is_sparse=False, fast_mode=False, num_kv_splits=nks, intra_batch_mode=True)
        wk = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
        wm, wi, wis, ri, rfm, rpm = wk
        get_mla_metadata_v1(qo_indptr, kv_indptr, kv_last, nq // nkv, nkv, True,
            wm, wis, wi, ri, rfm, rpm, page_size=1, kv_granularity=16,
            max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
            max_split_per_batch=nks, intra_batch_mode=True,
            dtype_q=BF16, dtype_kv=BF16)
        np2 = rpm.size(0)
        lg = torch.empty(np2, 1, nq, dv, dtype=torch.float32, device="cuda")
        ls = torch.empty(np2, 1, nq, 1, dtype=torch.float32, device="cuda")
        o = torch.empty(q.shape[0], nq, dv, dtype=BF16, device="cuda")
        _c[ck] = (kv_idx, kv_last, wm, wi, wis, ri, rfm, rpm, lg, ls, o)

    kv_idx, kv_last, wm, wi, wis, ri, rfm, rpm, lg, ls, o = _c[ck]

    kv_4d = kv_bf16.view(tkv, 1, nkv, dq)

    aiter.mla_decode_stage1_asm_fwd(q, kv_4d, qo_indptr, kv_indptr, kv_idx, kv_last,
        None, wm, wi, wis, 1, 1, nkv, sm, lg, ls, o, None, None)
    aiter.mla_reduce_v1(lg, ls, ri, rfm, rpm, 1, o, None)
    return o

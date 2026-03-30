#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Probe: Print mla_decode_fwd signature and source."""
import torch, inspect, sys
from task import input_t, output_t

# Print the actual API
from aiter.mla import mla_decode_fwd
print("\n=== mla_decode_fwd signature ===")
try:
    sig = inspect.signature(mla_decode_fwd)
    print(sig)
    for name, param in sig.parameters.items():
        print(f"  {name}: {param.kind.name} default={param.default}")
except Exception as e:
    print(f"sig error: {e}")

print("\n=== mla_decode_fwd source (first 100 lines) ===")
try:
    src = inspect.getsource(mla_decode_fwd)
    for i, line in enumerate(src.split('\n')[:100]):
        print(f"  {i+1}: {line}")
except Exception as e:
    print(f"source error: {e}")

# Minimal working kernel using direct ASM calls
import aiter
from aiter import dtypes as aiter_dtypes, get_mla_metadata_info_v1, get_mla_metadata_v1

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dv = config["v_head_dim"]
    dq = config["qk_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    total_kv = bs * kv_seq_len
    
    kv_fp8, kv_scale = kv_data["fp8"]
    kv_4d = kv_fp8.view(total_kv, 1, nkv, kv_fp8.shape[-1])
    
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device=q.device)
    kv_last = (kv_indptr[1:] - kv_indptr[:-1]).to(torch.int32)
    
    num_splits = 16
    info = get_mla_metadata_info_v1(bs, 1, nq, aiter_dtypes.fp8, aiter_dtypes.fp8,
        is_sparse=False, fast_mode=False, num_kv_splits=num_splits, intra_batch_mode=True)
    work = [torch.empty(s, dtype=t, device="cuda") for s, t in info]
    (wm, wi, wis, ri, rfm, rpm) = work
    get_mla_metadata_v1(qo_indptr, kv_indptr, kv_last, nq//nkv, nkv, True,
        wm, wis, wi, ri, rfm, rpm, page_size=1, kv_granularity=16,
        max_seqlen_qo=1, uni_seqlen_qo=1, fast_mode=False,
        max_split_per_batch=num_splits, intra_batch_mode=True,
        dtype_q=aiter_dtypes.fp8, dtype_kv=aiter_dtypes.fp8)
    
    # Simple fp8 quant
    amax = q.abs().amax().clamp(min=1e-12)
    scale = (amax / 448.0).to(torch.float32).reshape(1)
    q_fp8 = (q / scale).clamp(-448, 448).to(aiter_dtypes.fp8)
    
    n_part = rpm.size(0)
    logits = torch.empty(n_part, 1, nq, dv, dtype=torch.float32, device="cuda")
    lse = torch.empty(n_part, 1, nq, 1, dtype=torch.float32, device="cuda")
    o = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device="cuda")
    
    aiter.mla_decode_stage1_asm_fwd(q_fp8, kv_4d, qo_indptr, kv_indptr, kv_indices, kv_last,
        None, wm, wi, wis, 1, 1, nkv, sm_scale, logits, lse, o, scale, kv_scale)
    aiter.mla_reduce_v1(logits, lse, ri, rfm, rpm, 1, o, None)
    return o

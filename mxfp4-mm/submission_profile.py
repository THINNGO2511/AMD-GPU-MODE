#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Profile where time is spent: quantization vs GEMM vs overhead."""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n = B.shape[0]

    # Warmup
    A_fp4_w, A_scale_w = dynamic_mxfp4_quant(A)
    A_q_w = A_fp4_w.view(dtypes.fp4x2)
    A_s_w = e8m0_shuffle(A_scale_w).view(dtypes.fp8_e8m0)
    _ = aiter.gemm_a4w4(A_q_w, B_shuffle, A_s_w, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    torch.cuda.synchronize()

    N_ITER = 50

    # Time total
    s_total = torch.cuda.Event(enable_timing=True)
    e_total = torch.cuda.Event(enable_timing=True)
    s_total.record()
    for _ in range(N_ITER):
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh_t = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
        out = aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh_t, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    e_total.record()
    torch.cuda.synchronize()
    total_us = s_total.elapsed_time(e_total) * 1000 / N_ITER

    # Time quant only
    s_q = torch.cuda.Event(enable_timing=True)
    e_q = torch.cuda.Event(enable_timing=True)
    s_q.record()
    for _ in range(N_ITER):
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh_t = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
    e_q.record()
    torch.cuda.synchronize()
    quant_us = s_q.elapsed_time(e_q) * 1000 / N_ITER

    # Time GEMM only (with pre-computed A_q)
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh_t = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
    torch.cuda.synchronize()

    s_g = torch.cuda.Event(enable_timing=True)
    e_g = torch.cuda.Event(enable_timing=True)
    s_g.record()
    for _ in range(N_ITER):
        out = aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh_t, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
    e_g.record()
    torch.cuda.synchronize()
    gemm_us = s_g.elapsed_time(e_g) * 1000 / N_ITER

    print(f"M={m}, N={n}, K={k}: total={total_us:.1f}us (quant+shuffle={quant_us:.1f}us, gemm={gemm_us:.1f}us)")

    return out

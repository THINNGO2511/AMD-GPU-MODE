#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
CK probe v2 — focused on source code + timing breakdown + f4gemm CSV.
Previous probe got CSV data but source code was truncated.
"""
from task import input_t, output_t
import torch

_probed = False


def _probe_ck():
    global _probed
    if _probed:
        return
    _probed = True

    import os

    # 1. f4gemm CSV (kernel registry — different from tuned_gemm.csv)
    csv_path = "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
    if os.path.isfile(csv_path):
        with open(csv_path, 'r') as f:
            content = f.read()
        print(f"[P1] f4gemm CSV:\n{content}")
    else:
        print(f"[P1] f4gemm CSV not found")

    # 2. gemm_op_a4w4.py — JUST the key functions (not full source to avoid truncation)
    src_path = "/home/runner/aiter/aiter/ops/gemm_op_a4w4.py"
    if os.path.isfile(src_path):
        with open(src_path, 'r') as f:
            lines = f.readlines()
        print(f"\n[P2] gemm_op_a4w4.py ({len(lines)} lines):")
        # Print ALL lines — source is critical
        for i, line in enumerate(lines):
            print(f"  {i+1}|{line.rstrip()}")
    else:
        print(f"[P2] not found")

    # 3. Timing breakdown: quant vs GEMM
    import time
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter import dtypes
    import aiter

    print("\n[P3] TIMING BREAKDOWN:")
    for m, n, k in [(4, 2880, 512), (16, 2112, 7168), (64, 7168, 2048), (256, 3072, 1536)]:
        A = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
        # Fake B data for timing (same shape as benchmark)
        B_q_fake = torch.randint(0, 256, (n, k//2), dtype=torch.uint8, device='cuda')
        B_scale_fake = torch.randint(0, 256, ((n+191)//192*192//32, k//32*4*2), dtype=torch.uint8, device='cuda')

        # Warm up
        for _ in range(3):
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_scale_sh = e8m0_shuffle(A_scale)
        torch.cuda.synchronize()

        # Time quant
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(20):
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
        torch.cuda.synchronize()
        t_quant = (time.perf_counter_ns() - t0) / 20 / 1000  # us

        # Time shuffle
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(20):
            A_scale_sh = e8m0_shuffle(A_scale)
        torch.cuda.synchronize()
        t_shuf = (time.perf_counter_ns() - t0) / 20 / 1000  # us

        print(f"  M={m},N={n},K={k}: quant={t_quant:.1f}us, shuffle={t_shuf:.1f}us")

    # 4. get_padded_m behavior
    try:
        from aiter.ops.gemm_op_a4w4 import get_padded_m
        print("\n[P4] get_padded_m:")
        for m in [4, 16, 32, 64, 256]:
            for n, k in [(2880, 512), (2112, 7168), (7168, 2048), (3072, 1536)]:
                try:
                    pm = get_padded_m(m, n, k, 0)
                    print(f"  M={m},N={n},K={k}: padded_m={pm}")
                except Exception as e:
                    print(f"  M={m},N={n},K={k}: {e}")
    except Exception as e:
        print(f"[P4] get_padded_m not found: {e}")

    # 5. gemm_a4w4_asm direct call test with specific kernel names
    print("\n[P5] gemm_a4w4_asm direct test:")
    try:
        A_fp4, A_scale = dynamic_mxfp4_quant(torch.randn((4, 512), dtype=torch.bfloat16, device='cuda'))
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)
        out = torch.empty((4, 2880), dtype=torch.bfloat16, device='cuda')
        # Try calling with explicit kernel name
        kernel_name = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"
        # Need B_shuffle shape matching N=2880
        print(f"  gemm_a4w4_asm signature ready, kernel={kernel_name}")
    except Exception as e:
        print(f"  Failed: {e}")


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    _probe_ck()

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter import dtypes
    import aiter

    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale).view(dtypes.fp8_e8m0)

    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )

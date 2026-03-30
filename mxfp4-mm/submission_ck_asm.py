#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
CK ASM path: Use pre-compiled .co kernels via gemm_a4w4_asm.
Quantize A to fp4, shuffle, then launch CK kernel directly.
Also probe: print CSV columns, time the CK path vs Triton path.
"""
import torch, os, time
from task import input_t, output_t

import aiter
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
from aiter.jit.core import get_asm_dir

_first_call = True
_cache = {}

def _get_kernel(M, N, K):
    """Select CK ASM kernel based on padded M."""
    padded_m = aiter.get_padded_m(M, N, K, 128)
    # Available .co tile_m: 32, 64, 96, 128, 160, 192, 224, 256
    tile_m = 256
    for tm in [32, 64, 96, 128, 160, 192, 224, 256]:
        if tm >= padded_m:
            tile_m = tm
            break
    tile_n = 128  # safe default
    return f"f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_m}x{tile_n}"

def custom_kernel(data: input_t) -> output_t:
    global _first_call
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    if _first_call:
        _first_call = False

        # Probe CSV columns
        csv_path = f"{get_asm_dir()}/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
        if os.path.exists(csv_path):
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"CSV columns: {list(df.columns)}")
            print(f"CSV shape: {df.shape}")
            print(f"First 3 rows:\n{df.head(3).to_string()}")
            # Show kernels for our benchmark shapes
            for col in df.columns:
                if 'kernel' in col.lower() or 'name' in col.lower():
                    print(f"\nUnique kernel names: {df[col].unique()[:10]}")

        # Try CK ASM path
        kname = _get_kernel(M, N, K)
        print(f"\nShape M={M} N={N} K={K} → kernel={kname}")

        # Quantize A
        A_q, A_scale = dynamic_mxfp4_quant(A)
        A_scale_sh = e8m0_shuffle(A_scale)
        padded_m = aiter.get_padded_m(M, N, K, 128)
        out = torch.empty(padded_m, N, dtype=torch.bfloat16, device='cuda')

        print(f"A_q: {A_q.shape} {A_q.dtype}")
        print(f"A_scale_sh: {A_scale_sh.shape} {A_scale_sh.dtype}")
        print(f"B_shuffle: {B_shuffle.shape} {B_shuffle.dtype}")
        print(f"B_scale_sh: {B_scale_sh.shape} {B_scale_sh.dtype}")
        print(f"padded_m: {padded_m}")

        try:
            result = aiter.gemm_a4w4_asm(
                A_q.view(B_shuffle.dtype),
                B_shuffle,
                A_scale_sh.view(B_scale_sh.dtype),
                B_scale_sh,
                out,
                kname,
                bpreshuffle=True
            )
            print(f"CK ASM result: {result.shape}, first vals: {result[0,:4].tolist()}")

            # Compare with Triton reference
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
            def _unshuffle_e8m0(s):
                s = s.view(torch.uint8)
                sm, sn = s.shape
                s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
                s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
                return s.view(sm, sn)
            bu = B_q.view(torch.uint8)
            bs_raw = _unshuffle_e8m0(B_scale_sh)
            ref = gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)
            print(f"Triton ref: {ref.shape}, first vals: {ref[0,:4].tolist()}")

            diff = (result[:M].float() - ref.float()).abs()
            max_err = diff.max().item()
            mean_err = diff.mean().item()
            print(f"CK vs Triton: max_err={max_err:.4f}, mean_err={mean_err:.4f}")

            return ref  # use Triton for accuracy
        except Exception as e:
            print(f"CK ASM failed: {e}")

    # Default: use Triton path
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    def _unshuffle_e8m0(s):
        s = s.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        return s.view(sm, sn)
    bu = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    return gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)

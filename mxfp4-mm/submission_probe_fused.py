#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Probe fused_gemm_afp4wfp4_a16w16 and gemm_a16wfp4_preshuffle APIs.
Also read source of gemm_a16wfp4 to understand config/reduce mechanism.
"""
from task import input_t, output_t
import torch
import inspect
import os

_probed = False
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _probed, _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if not _probed:
        _probed = True

        # 1. Probe fused_gemm_afp4wfp4_a16w16
        print("\n=== 1. fused_gemm_afp4wfp4_a16w16 ===")
        try:
            from aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_a16w16 import fused_gemm_afp4wfp4_a16w16
            print(f"  Signature: {inspect.signature(fused_gemm_afp4wfp4_a16w16)}")
            src = inspect.getsource(fused_gemm_afp4wfp4_a16w16)
            # Print first 3000 chars of source
            print(f"  Source ({len(src)} chars):")
            for line in src.split('\n')[:80]:
                print(f"  | {line}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 2. Probe gemm_a16wfp4_preshuffle
        print("\n=== 2. gemm_a16wfp4_preshuffle ===")
        try:
            mod_path = "/home/runner/aiter/aiter/ops/triton/gemm/basic/"
            files = [f for f in os.listdir(mod_path) if 'preshuffle' in f or 'a16w' in f]
            print(f"  Files matching: {files}")
            for f in files:
                fpath = os.path.join(mod_path, f)
                with open(fpath) as fh:
                    content = fh.read()
                print(f"\n  --- {f} ({len(content)} chars) ---")
                # Print function signatures and key parts
                for i, line in enumerate(content.split('\n')):
                    if any(kw in line for kw in ['def ', 'class ', 'import ', 'config', 'prequant', 'shuffle']):
                        print(f"  L{i}: {line}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 3. Read gemm_a16wfp4 source - focus on split-K reduce
        print("\n=== 3. gemm_a16wfp4 split-K mechanism ===")
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
            src = inspect.getsource(gemm_a16wfp4)
            lines = src.split('\n')
            # Find reduce-related code
            for i, line in enumerate(lines):
                if any(kw in line.lower() for kw in ['reduce', 'split', 'ksplit', 'atomic']):
                    s = max(0, i-1)
                    e = min(len(lines), i+2)
                    for j in range(s, e):
                        print(f"  L{j}: {lines[j]}")
                    print("  ---")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 4. Check existing tuned configs for our exact sizes
        print("\n=== 4. Existing tuned configs ===")
        cfg_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
        if os.path.exists(cfg_dir):
            all_files = os.listdir(cfg_dir)
            fp4_files = [f for f in all_files if 'FP4' in f and 'gfx950' in f]
            print(f"  Total FP4 gfx950 configs: {len(fp4_files)}")
            # Check for a16wfp4 specific
            a16w = [f for f in fp4_files if 'A16W' in f or 'a16w' in f.lower()]
            print(f"  A16W configs: {a16w}")
            # Check for configs matching our N/K combos
            for target_nk in ['2112_7168', '7168_2048', '3072_1536', '2880_512', '4096_512']:
                matches = [f for f in fp4_files if target_nk in f]
                if matches:
                    print(f"  Configs for N_K={target_nk}: {matches}")
                    for mf in matches[:2]:
                        with open(os.path.join(cfg_dir, mf)) as fh:
                            print(f"    Content: {fh.read()[:300]}")

        # 5. Check if there's a hipBLASLt fp4 GEMM
        print("\n=== 5. hipBLASLt/rocBLAS fp4 ===")
        try:
            import aiter
            # Check for hipblaslt operations
            hipb = [a for a in dir(aiter) if 'hip' in a.lower() or 'blas' in a.lower() or 'gemm' in a.lower()]
            print(f"  aiter attrs: {hipb}")
        except Exception as e:
            print(f"  {e}")

        try:
            from aiter.jit.core import renew_ck_gemm
            print(f"  renew_ck_gemm found: {inspect.signature(renew_ck_gemm)}")
        except Exception as e:
            print(f"  renew_ck_gemm: {e}")

        # 6. Check all basic gemm files
        print("\n=== 6. All basic GEMM files ===")
        basic_dir = "/home/runner/aiter/aiter/ops/triton/gemm/basic/"
        for f in sorted(os.listdir(basic_dir)):
            if f.endswith('.py') and not f.startswith('__'):
                print(f"  {f}")

    # Return valid result
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16)

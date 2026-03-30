from task import input_t, output_t
import torch
import subprocess
import os
import glob

def custom_kernel(data: input_t) -> output_t:
    """Probe runner state — check if aiter was updated."""
    A, B, B_q, B_shuffle, B_scale_sh = data

    print("=" * 60)
    print("RUNNER PROBE v3 — 2026-03-29")
    print("=" * 60)

    # 1. Git log — check for commits past f3be04a12 (#2156)
    print("\n--- GIT LOG (last 10 commits) ---")
    try:
        r = subprocess.run(["git", "-C", "/home/runner/aiter", "log", "--oneline", "-10"],
                          capture_output=True, text=True, timeout=10)
        print(r.stdout)
        r2 = subprocess.run(["git", "-C", "/home/runner/aiter", "rev-list", "--count", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        print(f"Total commits: {r2.stdout.strip()}")
    except Exception as e:
        print(f"Git error: {e}")

    # 2. Check aiter version
    print("\n--- AITER VERSION ---")
    try:
        import aiter
        print(f"aiter version: {getattr(aiter, '__version__', 'no __version__')}")
    except Exception as e:
        print(f"aiter import error: {e}")

    # 3. FlyDSL binaries
    print("\n--- FLYDSL BINARIES ---")
    flydsl_files = glob.glob("/home/runner/aiter/hsa/gfx950/**/*flydsl*", recursive=True)
    print(f"FlyDSL binary count: {len(flydsl_files)}")
    for f in flydsl_files[:10]:
        print(f"  {f}")

    # 4. qseqlen dispatch in mla.py
    print("\n--- MLA QSEQLEN DISPATCH ---")
    try:
        r = subprocess.run(["grep", "-n", "qseqlen", "/home/runner/aiter/aiter/mla.py"],
                          capture_output=True, text=True, timeout=5)
        if r.stdout:
            print(r.stdout[:500])
        else:
            print("No qseqlen references in mla.py")
    except Exception as e:
        print(f"Error: {e}")

    # 5. MLA .co files
    print("\n--- MLA KERNEL FILES ---")
    mla_files = glob.glob("/home/runner/aiter/hsa/gfx950/mla*.co")
    print(f"MLA .co count: {len(mla_files)}")
    for f in sorted(mla_files):
        print(f"  {os.path.basename(f)}")

    # 6. PR #2261 configs
    print("\n--- GEMM CONFIG FILES ---")
    cfg_files = glob.glob("/home/runner/aiter/aiter/ops/triton/configs/gemm/*gfx950*fp4*")
    print(f"gfx950 FP4 config count: {len(cfg_files)}")
    for f in sorted(cfg_files)[:10]:
        print(f"  {os.path.basename(f)}")

    # 7. Key file keyword checks
    print("\n--- KEY FILE CHECKS ---")
    checks = [
        ("/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py", "waves_per_eu"),
        ("/home/runner/aiter/aiter/fused_moe.py", "flydsl"),
        ("/home/runner/aiter/aiter/mla.py", "qseqlen_fold"),
    ]
    for filepath, keyword in checks:
        try:
            r = subprocess.run(["grep", "-c", keyword, filepath],
                              capture_output=True, text=True, timeout=5)
            count = r.stdout.strip()
            print(f"  {os.path.basename(filepath)}: '{keyword}' appears {count} times")
        except Exception as e:
            print(f"  {os.path.basename(filepath)}: error: {e}")

    # 8. Total .co file count
    print("\n--- TOTAL .CO FILES ---")
    all_co = glob.glob("/home/runner/aiter/hsa/gfx950/**/*.co", recursive=True)
    print(f"Total .co files: {len(all_co)}")

    # 9. fused_moe new features
    print("\n--- FUSED_MOE NEW FEATURES ---")
    try:
        r = subprocess.run(["grep", "-nE", "flydsl|fly_dsl|FlyDSL|block_size_m|blockPerCu",
                           "/home/runner/aiter/aiter/fused_moe.py"],
                          capture_output=True, text=True, timeout=5)
        if r.stdout:
            print(r.stdout[:500])
        else:
            print("No new features found in fused_moe.py")
    except Exception as e:
        print(f"Error: {e}")

    # 10. ROCm version
    print("\n--- ROCM VERSION ---")
    try:
        r = subprocess.run(["cat", "/opt/rocm/.info/version"], capture_output=True, text=True, timeout=5)
        print(f"ROCm: {r.stdout.strip()}")
    except:
        print("Could not read ROCm version")

    print("\n" + "=" * 60)
    print("PROBE COMPLETE")
    print("=" * 60)

    # Return reference result
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    m, k = A.shape
    n = B.shape[0]
    scale_raw = B_scale_sh.clone()
    s = scale_raw.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale_unshuf = s.view(sm, sn)

    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, B_q.view(torch.uint8), scale_unshuf, dtype=torch.bfloat16, y=out)
    return out

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Probe runner environment for Triton upgrade feasibility.
Check: internet access, Python version, Triton version, torch version.
Also probe for petit-kernel availability.
"""
import os
import sys
import subprocess
from task import input_t, output_t

_probed = False


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Check versions
    import torch
    import triton
    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"HIP: {torch.version.hip}")
    print(f"CUDA arch list: {torch.cuda.get_arch_list()}")

    # 2. Check internet access
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "5",
             "https://repo.radeon.com/rocm/manylinux/rocm-rel-7.2.1/"],
            capture_output=True, text=True, timeout=10
        )
        print(f"Internet (radeon repo): HTTP {result.stdout}")
    except Exception as e:
        print(f"Internet check failed: {e}")

    # 3. Check pip install feasibility
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "5",
             "https://pypi.org/simple/"],
            capture_output=True, text=True, timeout=10
        )
        print(f"Internet (pypi): HTTP {result.stdout}")
    except Exception as e:
        print(f"PyPI check failed: {e}")

    # 4. Check GitHub access (for petit-kernel)
    try:
        result = subprocess.run(
            ["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
             "--connect-timeout", "5",
             "https://github.com/causalflow-ai/petit-kernel"],
            capture_output=True, text=True, timeout=10
        )
        print(f"Internet (github): HTTP {result.stdout}")
    except Exception as e:
        print(f"GitHub check failed: {e}")

    # 5. Check available disk space
    try:
        result = subprocess.run(["df", "-h", "/tmp"], capture_output=True, text=True, timeout=5)
        print(f"Disk: {result.stdout.strip().split(chr(10))[-1]}")
    except:
        pass

    # 6. Check if hipModuleLoad is available
    try:
        from ctypes import cdll
        hip = cdll.LoadLibrary("libamdhip64.so")
        print(f"libamdhip64.so loaded: {hip}")
        # Check for hipLibraryLoadFromFile (7.2 API)
        if hasattr(hip, 'hipLibraryLoadFromFile'):
            print("hipLibraryLoadFromFile: AVAILABLE (7.2)")
        else:
            print("hipLibraryLoadFromFile: NOT available (7.1)")
    except Exception as e:
        print(f"HIP library check: {e}")

    # 7. Check for petit-kernel or similar pre-installed
    try:
        result = subprocess.run(
            ["pip", "list"], capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if any(kw in line.lower() for kw in ['petit', 'hipblaslt', 'composable']):
                print(f"  Found: {line}")
    except:
        pass


def _unshuffle_e8m0(scale_sh):
    import torch
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    import torch
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16)

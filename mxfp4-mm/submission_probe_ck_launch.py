#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe: Read aiter CK kernel launch code to understand hipModuleLoad interface."""
import torch, os, glob
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # 1. List all .co files in f4gemm directory
    print("=== .co kernel files ===")
    co_dir = "/home/runner/aiter/hsa/gfx950/f4gemm/"
    if os.path.exists(co_dir):
        cos = sorted(os.listdir(co_dir))
        for f in cos:
            print(f"  {f}")
    else:
        print(f"  {co_dir} not found")

    # 2. Find the C++ source that launches these kernels
    print("\n=== Searching for gemm_a4w4 C++ source ===")
    search_paths = [
        "/home/runner/aiter/aiter/jit/build/module_gemm_a4w4_asm/",
        "/home/runner/aiter/csrc/",
        "/home/runner/aiter/aiter/",
        "/home/runner/aiter/hsa/",
    ]
    for sp in search_paths:
        if os.path.exists(sp):
            for root, dirs, files in os.walk(sp):
                for f in files:
                    if any(ext in f for ext in ['.cpp', '.cu', '.hip', '.py', '.h']):
                        fp = os.path.join(root, f)
                        try:
                            content = open(fp).read()
                            if 'hipModuleLoad' in content or 'hipModuleLaunchKernel' in content or 'f4gemm' in content:
                                print(f"\n  FOUND: {fp} ({len(content)} bytes)")
                                # Print relevant sections
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if any(kw in line for kw in ['hipModuleLoad', 'hipModuleLaunchKernel', 'f4gemm', 'hipFunction_t', 'launch', 'grid', 'block', 'args']):
                                        start = max(0, i-2)
                                        end = min(len(lines), i+8)
                                        print(f"    Lines {start+1}-{end}:")
                                        for j in range(start, end):
                                            print(f"      {j+1}: {lines[j][:200]}")
                                        print()
                        except:
                            pass

    # 3. Read the codegen.py that generates kernel configs
    print("\n=== codegen.py (kernel launch config) ===")
    codegen = "/home/runner/aiter/hsa/codegen.py"
    if os.path.exists(codegen):
        content = open(codegen).read()
        lines = content.split('\n')
        # Find launch-related code
        for i, line in enumerate(lines):
            if any(kw in line for kw in ['hipModule', 'launch', 'grid', 'block', 'f4gemm', 'def ', 'class ']):
                start = max(0, i-1)
                end = min(len(lines), i+5)
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j][:200]}")
                print()

    # 4. Find the Python wrapper for gemm_a4w4_asm
    print("\n=== gemm_a4w4 Python source ===")
    for pyf in ["/home/runner/aiter/aiter/ops/hip/gemm_a4w4.py",
                "/home/runner/aiter/aiter/ops/gemm_a4w4.py"]:
        if os.path.exists(pyf):
            content = open(pyf).read()
            print(f"  {pyf}:")
            print(content[:3000])

    # 5. Check the jit module source
    print("\n=== JIT module source for a4w4 ===")
    jit_dir = "/home/runner/aiter/aiter/jit/build/module_gemm_a4w4_asm/"
    if os.path.exists(jit_dir):
        for f in os.listdir(jit_dir):
            fp = os.path.join(jit_dir, f)
            if f.endswith(('.cpp', '.cu', '.hip')):
                content = open(fp).read()
                print(f"\n  {f} ({len(content)} bytes):")
                print(content[:4000])

    # Use reference for output
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

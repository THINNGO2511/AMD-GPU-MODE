"""Focused probe: .co files, git log, _get_config source, preshuffle kernel source"""
import os, glob, subprocess

# 1. ALL .co kernel files
for d in ["/home/runner/aiter/hsa/gfx950/f4gemm/", "/home/runner/aiter/hsa/gfx950/mla/",
          "/home/runner/aiter/hsa/gfx950/"]:
    cos = sorted(glob.glob(d + "*.co"))
    if cos:
        print(f"\n[.co in {d}] {len(cos)} files:")
        for c in cos:
            print(f"  {os.path.basename(c)} ({os.path.getsize(c)}B)")

# 2. MoE .co files
for pattern in ["*moe*", "*fmoe*", "*2stage*"]:
    cos = glob.glob(f"/home/runner/aiter/hsa/gfx950/**/{pattern}.co", recursive=True)
    if cos:
        print(f"\n[{pattern} .co] {len(cos)} files:")
        for c in sorted(cos)[:20]:
            print(f"  {c}")

# 3. Git log
try:
    r = subprocess.run(["git", "-C", "/home/runner/aiter", "log", "--oneline", "-15"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[GIT LOG]\n{r.stdout}")
except Exception as e:
    print(f"[GIT] {e}")

# 4. _get_config source (the actual Triton kernel file)
for path in ["/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py"]:
    try:
        with open(path) as f:
            lines = f.readlines()
        print(f"\n[FILE] {path} ({len(lines)} lines)")
        # Find _get_config function
        for i, line in enumerate(lines):
            if "_get_config" in line or "autotune" in line.lower() or "OPTIMIZE_EPILOGUE" in line:
                start = max(0, i-2)
                end = min(len(lines), i+15)
                print(f"  --- line {start+1}-{end} ---")
                for j in range(start, end):
                    print(f"  {j+1}: {lines[j]}", end='')
    except Exception as e:
        print(f"[FILE] {path}: {e}")

# 5. Versions
import torch, triton
print(f"\n[VERSIONS] torch={torch.__version__} triton={triton.__version__} hip={torch.version.hip}")

# 6. Read the actual kernel config selection (_get_config)
try:
    r = subprocess.run(["grep", "-rn", "def _get_config", "/home/runner/aiter/aiter/ops/triton/"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[_get_config locations]\n{r.stdout}")
except: pass

# 7. Check for A16WFP4 specific JSON configs (not just AFP4WFP4)
a16_configs = glob.glob("/home/runner/aiter/aiter/ops/triton/configs/gemm/*A16WFP4*")
print(f"\n[A16WFP4 configs] {len(a16_configs)} files:")
for f in sorted(a16_configs):
    print(f"  {os.path.basename(f)}")

# 8. Preshuffle files
preshuffle = glob.glob("/home/runner/aiter/**/*preshuffle*", recursive=True)
print(f"\n[preshuffle files] {len(preshuffle)} files:")
for f in sorted(preshuffle)[:15]:
    print(f"  {f}")

# 9. Check dsv3 CSV for E=33 entries
try:
    with open("/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv") as f:
        lines = f.readlines()
    print(f"\n[dsv3 CSV] {len(lines)} lines")
    # Print header + E=33 entries
    print(lines[0].strip())
    for line in lines:
        if ",33," in line or "E=33" in line.lower():
            print(line.strip())
    # Also print E=257 d=256 entries (first few)
    count = 0
    for line in lines:
        if ",257," in line and ",256," in line:
            print(line.strip())
            count += 1
            if count >= 5:
                break
except Exception as e:
    print(f"[dsv3 CSV] {e}")

# Actual GEMM to pass test
import torch
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data):
    A = data['A']
    B_q = data['B_q'].view(torch.uint8)
    B_scale = _unshuffle_e8m0(data['B_scale_sh'])
    M, K = A.shape
    N = B_q.shape[0]
    output = torch.empty(M, N, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, B_q, B_scale, dtype=torch.bfloat16, y=output)
    return output

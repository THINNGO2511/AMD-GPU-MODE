"""MoE source code probe — reads fused_moe source, CSV configs, CK kernel names"""
import os, glob

def _read_file(path, max_lines=200):
    try:
        with open(path) as f:
            lines = f.readlines()
        print(f"\n{'='*60}\n[FILE] {path} ({len(lines)} lines)\n{'='*60}")
        for i, line in enumerate(lines[:max_lines]):
            print(line, end='')
        if len(lines) > max_lines:
            print(f"\n... truncated ({len(lines) - max_lines} more lines)")
    except Exception as e:
        print(f"[FILE] {path}: {e}")

# Phase 2A: fused_moe.py full source
_read_file("/home/runner/aiter/aiter/fused_moe.py", 400)

# Phase 2B: tuned CSV files
_read_file("/home/runner/aiter/aiter/configs/tuned_fmoe.csv", 50)
_read_file("/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv", 100)

# Phase 2C: CK MoE kernels
cos = sorted(glob.glob("/home/runner/aiter/hsa/gfx950/*moe*/*.co") +
             glob.glob("/home/runner/aiter/hsa/gfx950/fmoe*/*.co"))
if not cos:
    cos = sorted(glob.glob("/home/runner/aiter/hsa/gfx950/**/*moe*.co", recursive=True))
print(f"\n[MoE .co files] {len(cos)} found:")
for c in cos[:30]:
    print(f"  {c} ({os.path.getsize(c)} bytes)")

# Phase 2D: FlyDSL
flydsl = sorted(glob.glob("/home/runner/aiter/**/*flydsl*", recursive=True) +
                glob.glob("/home/runner/aiter/**/*fly_dsl*", recursive=True))
print(f"\n[FlyDSL files] {len(flydsl)} found:")
for f in flydsl[:20]:
    print(f"  {f}")

# Phase 2E: env vars in fused_moe
import subprocess
try:
    r = subprocess.run(["grep", "-n", "environ\\|getenv\\|os.env\\|AITER",
                        "/home/runner/aiter/aiter/fused_moe.py"],
                       capture_output=True, text=True, timeout=5)
    print(f"\n[ENV VARS in fused_moe.py]\n{r.stdout}")
except: pass

# Phase 2C extra: search for CK stage config files
for pattern in ["ck_moe", "2stages", "fmoe_2stage"]:
    matches = glob.glob(f"/home/runner/aiter/hsa/gfx950/**/*{pattern}*", recursive=True)
    if matches:
        print(f"\n[{pattern}] {len(matches)} files:")
        for m in matches[:10]:
            print(f"  {m}")

# Now do the actual MoE to pass the test
import torch
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

def custom_kernel(data):
    return fused_moe(
        data['hidden_states'], data['w1'], data['w2'],
        data['topk_weights'], data['topk_ids'],
        w1_scale=data['w1_scale'], w2_scale=data['w2_scale'],
        activation=ActivationType.Silu,
    )

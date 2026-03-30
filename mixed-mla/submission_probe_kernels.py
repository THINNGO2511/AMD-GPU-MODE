#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""Probe: List all MLA kernel files and CSV configs on the runner."""
import os, sys, glob, torch
from task import input_t, output_t

# Print all MLA kernel files
mla_dir = "/home/runner/aiter/hsa/gfx950/mla/"
print(f"\n=== MLA kernel directory: {mla_dir} ===")
if os.path.exists(mla_dir):
    for f in sorted(os.listdir(mla_dir)):
        size = os.path.getsize(os.path.join(mla_dir, f))
        print(f"  {f} ({size} bytes)")
else:
    print("  DIR NOT FOUND")

# Check for page-related kernels
print("\n=== Searching for page/pg kernels ===")
for root, dirs, files in os.walk("/home/runner/aiter/hsa/gfx950/"):
    for f in files:
        if "page" in f.lower() or "_pg" in f.lower():
            print(f"  {os.path.join(root, f)}")

# Print the MLA ASM CSV dispatch table
csv_path = "/home/runner/aiter/hsa/gfx950/mla/mla_asm.csv"
if os.path.exists(csv_path):
    print(f"\n=== mla_asm.csv ===")
    with open(csv_path) as f:
        print(f.read())
else:
    # Try alternate locations
    for p in glob.glob("/home/runner/aiter/**/*mla*csv", recursive=True):
        print(f"\n=== {p} ===")
        with open(p) as f:
            print(f.read()[:2000])

# Check aiter version
print("\n=== aiter version ===")
try:
    import aiter
    print(dir(aiter))
    if hasattr(aiter, '__version__'):
        print(f"Version: {aiter.__version__}")
except Exception as e:
    print(f"Error: {e}")

# Check for experimental features
print("\n=== Experimental flags ===")
for env in ["AITER_ENABLE_EXPERIMENTAL", "AITER_MLA_MODE"]:
    print(f"  {env}={os.environ.get(env, 'NOT SET')}")

# Check mla.py source for dispatch logic
mla_py = "/home/runner/aiter/aiter/mla.py"
if os.path.exists(mla_py):
    print(f"\n=== mla.py dispatch functions ===")
    with open(mla_py) as f:
        content = f.read()
    # Print function signatures
    for line in content.split('\n'):
        if 'def ' in line and ('mla' in line.lower() or 'decode' in line.lower()):
            print(f"  {line.strip()}")
    # Print page_size handling
    for i, line in enumerate(content.split('\n')):
        if 'page' in line.lower():
            print(f"  L{i+1}: {line.rstrip()}")

# Now do the actual kernel work (use reference approach)
import aiter
from aiter import dtypes as aiter_dtypes
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1
from aiter.mla import mla_decode_fwd

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data
    bs = config["batch_size"]
    nq = config["num_heads"]
    nkv = config["num_kv_heads"]
    dq = config["qk_head_dim"]
    dv = config["v_head_dim"]
    sm_scale = config["sm_scale"]
    kv_seq_len = config["kv_seq_len"]
    
    kv_fp8, kv_scale = kv_data["fp8"]
    output = torch.empty(q.shape[0], nq, dv, dtype=torch.bfloat16, device=q.device)
    
    mla_decode_fwd(q, kv_fp8, qo_indptr, kv_indptr,
                   output=output, sm_scale=sm_scale,
                   kv_scale=kv_scale, page_size=1,
                   num_kv_splits=16)
    return output

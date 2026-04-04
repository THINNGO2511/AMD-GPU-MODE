"""Deep source probe: MLA dispatch logic, preshuffle kernel, config loader, MoE 2stage_cfgs, _page kernels"""
import os, glob, subprocess

def _read(path, max_lines=150):
    try:
        with open(path) as f:
            lines = f.readlines()
        print(f"\n[FILE] {path} ({len(lines)} lines)")
        for i, line in enumerate(lines[:max_lines]):
            print(f"{i+1}: {line}", end='')
        if len(lines) > max_lines:
            print(f"\n... ({len(lines)-max_lines} more)")
    except Exception as e:
        print(f"[ERR] {path}: {e}")

def _grep(pattern, path, max=30):
    try:
        r = subprocess.run(["grep", "-n", pattern, path], capture_output=True, text=True, timeout=5)
        lines = r.stdout.strip().split('\n')[:max]
        print(f"\n[GREP '{pattern}' in {os.path.basename(path)}]")
        for l in lines:
            if l: print(f"  {l}")
    except Exception as e:
        print(f"[GREP ERR] {e}")

# 2A: MLA qseqlen dispatch — read the ASM kernel loader (codegen)
print("=" * 60)
print("SECTION 2A: MLA kernel dispatch / qseqlen selection")
print("=" * 60)
_grep("qseqlen\\|max_seqlen_q\\|qseq", "/home/runner/aiter/aiter/mla.py")
# Read the ASM kernel loading code
for path in glob.glob("/home/runner/aiter/hsa/*.py"):
    _grep("mla.*qseqlen\\|qseq.*mla\\|stage1.*asm", path)

# Check codegen.py for MLA kernel selection
_grep("mla\\|qseqlen\\|stage1_asm", "/home/runner/aiter/hsa/codegen.py")

# 2B: Preshuffle function in gemm_a16wfp4
print("\n" + "=" * 60)
print("SECTION 2B: Preshuffle kernel source")
print("=" * 60)
_grep("def.*preshuffle\\|preshuffle_kernel\\|_PRESHUFFLED", "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py")
# Read the Triton kernel file for preshuffle
_read("/home/runner/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py", 120)

# 2C: PRESHUFFLED JSON configs
print("\n" + "=" * 60)
print("SECTION 2C: JSON configs")
print("=" * 60)
for f in ["gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json",
          "gfx950-GEMM-A16WFP4_PRESHUFFLED.json",
          "gfx950-GEMM-A16WFP4.json"]:
    _read(f"/home/runner/aiter/aiter/ops/triton/configs/gemm/{f}", 80)

# 2D: Config loader
print("\n" + "=" * 60)
print("SECTION 2D: Config loader")
print("=" * 60)
try:
    r = subprocess.run(["grep", "-rn", "def get_gemm_config\\|load_config\\|json.load\\|config_dir",
                        "/home/runner/aiter/aiter/ops/triton/"],
                       capture_output=True, text=True, timeout=5)
    print(r.stdout[:3000])
except: pass

# 2E: MoE get_2stage_cfgs
print("\n" + "=" * 60)
print("SECTION 2E: MoE get_2stage_cfgs")
print("=" * 60)
_grep("def get_2stage\\|run_1stage\\|1stage\\|one_stage", "/home/runner/aiter/aiter/fused_moe.py")

# Read the get_2stage_cfgs function
try:
    with open("/home/runner/aiter/aiter/fused_moe.py") as f:
        lines = f.readlines()
    in_func = False
    count = 0
    for i, line in enumerate(lines):
        if "def get_2stage_cfgs" in line:
            in_func = True
        if in_func:
            print(f"  {i+1}: {line}", end='')
            count += 1
            if count > 80:
                break
            if count > 3 and line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                break
except: pass

# 2F: _page kernel refs
print("\n" + "=" * 60)
print("SECTION 2F: _page kernel refs in mla.py")
print("=" * 60)
_grep("_page\\|page_table\\|native_page\\|kPageSize", "/home/runner/aiter/aiter/mla.py")

# 2G: Write permissions
print("\n" + "=" * 60)
print("SECTION 2G: Write permissions")
print("=" * 60)
cfg_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm/"
try:
    test_path = os.path.join(cfg_dir, "_test_write.json")
    with open(test_path, 'w') as f:
        f.write('{"test": true}')
    os.remove(test_path)
    print(f"CAN WRITE to {cfg_dir}")
except Exception as e:
    print(f"CANNOT WRITE to {cfg_dir}: {e}")

# Actual MoE to pass test
import torch
from aiter.fused_moe import fused_moe
from aiter import ActivationType

def custom_kernel(data):
    return fused_moe(
        data['hidden_states'], data['w1'], data['w2'],
        data['topk_weights'], data['topk_ids'],
        w1_scale=data['w1_scale'], w2_scale=data['w2_scale'],
        activation=ActivationType.Silu,
    )

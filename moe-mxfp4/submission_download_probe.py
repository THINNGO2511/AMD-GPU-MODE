#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe: Investigate CK kernel loading architecture.
Goals:
1. Find where module_moe_ck2stages .so is cached
2. Check if fmoe_2stages .co files are actually used by FP4 path
3. Determine if we can download newer kernels from GitHub
4. Measure JIT compilation time
5. Check internet access and wget/curl availability
"""
import os
import sys
import time
import glob
import subprocess
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_probed = False


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    P = lambda msg: print(msg, file=sys.stderr)

    # ==========================================
    # 1. Find cached .so module files
    # ==========================================
    P("\n=== 1. MODULE .SO FILES ===")
    for search_dir in [
        os.path.expanduser("~/.aiter/"),
        "/home/runner/aiter/aiter/jit/",
        "/home/runner/aiter/",
    ]:
        if os.path.isdir(search_dir):
            for root, dirs, files in os.walk(search_dir):
                for f in files:
                    if f.endswith('.so') and 'moe' in f.lower():
                        fp = os.path.join(root, f)
                        sz = os.path.getsize(fp)
                        mt = os.path.getmtime(fp)
                        P(f"  .so: {fp} ({sz} bytes, mtime={time.ctime(mt)})")

    # ==========================================
    # 2. Check what's in fmoe_2stages .co dir
    # ==========================================
    P("\n=== 2. FMOE_2STAGES .CO FILES ===")
    co_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    if os.path.isdir(co_dir):
        files = sorted(os.listdir(co_dir))
        P(f"  Total files: {len(files)}")
        fp4_files = [f for f in files if 'FP4' in f or 'fp4' in f]
        P(f"  FP4 files: {len(fp4_files)}")
        for f in fp4_files[:20]:
            P(f"    {f}")
        non_fp4 = [f for f in files if 'FP4' not in f and 'fp4' not in f][:10]
        P(f"  Non-FP4 sample:")
        for f in non_fp4:
            P(f"    {f}")
    else:
        P(f"  DIR NOT FOUND: {co_dir}")

    # Also check fmoe/silu and fmoe/gelu for FP4
    for subdir in ["/home/runner/aiter/hsa/gfx950/fmoe/silu/",
                   "/home/runner/aiter/hsa/gfx950/fmoe/gelu/",
                   "/home/runner/aiter/hsa/gfx950/fmoe/"]:
        if os.path.isdir(subdir):
            files = sorted(os.listdir(subdir))
            fp4 = [f for f in files if 'FP4' in f or 'fp4' in f or 'mxfp4' in f]
            P(f"  {subdir}: {len(files)} total, {len(fp4)} FP4")
            for f in fp4[:5]:
                P(f"    {f}")

    # ==========================================
    # 3. Check CK module architecture - how kernels load
    # ==========================================
    P("\n=== 3. CK MODULE LOADING ARCHITECTURE ===")
    try:
        import importlib
        # Check if the CK module is already loaded
        for name in sorted(sys.modules.keys()):
            if 'moe_ck2stages' in name or 'module_moe' in name:
                mod = sys.modules[name]
                P(f"  Loaded module: {name}")
                P(f"    File: {getattr(mod, '__file__', 'N/A')}")
                P(f"    Dir: {sorted(dir(mod))[:20]}")
    except Exception as e:
        P(f"  Module check error: {e}")

    # Check JIT dir and AITER env vars
    P("\n=== 4. AITER ENV AND JIT CONFIG ===")
    for var in ['AITER_JIT_DIR', 'AITER_ASM_DIR', 'AITER_REBUILD',
                'AITER_META_DIR', 'CK_DIR', 'PYTORCH_ROCM_ARCH',
                'AITER_CONFIG_FMOE', 'AITER_ROOT_DIR']:
        P(f"  {var}={os.environ.get(var, '<not set>')}")

    try:
        from aiter.jit.core import get_user_jit_dir, get_asm_dir, AITER_CSRC_DIR
        jit_dir = get_user_jit_dir()
        asm_dir = get_asm_dir()
        P(f"  get_user_jit_dir() = {jit_dir}")
        P(f"  get_asm_dir() = {asm_dir}")
        P(f"  AITER_CSRC_DIR = {AITER_CSRC_DIR}")

        # List all .so in JIT dir
        if os.path.isdir(jit_dir):
            sos = [f for f in os.listdir(jit_dir) if f.endswith('.so')]
            P(f"  JIT dir .so files: {len(sos)}")
            for f in sorted(sos):
                fp = os.path.join(jit_dir, f)
                sz = os.path.getsize(fp)
                mt = os.path.getmtime(fp)
                P(f"    {f} ({sz} bytes, {time.ctime(mt)})")
    except Exception as e:
        P(f"  JIT core import error: {e}")

    # ==========================================
    # 5. Check internet access
    # ==========================================
    P("\n=== 5. INTERNET ACCESS ===")
    try:
        result = subprocess.run(
            ['curl', '-sI', '--connect-timeout', '5', 'https://api.github.com'],
            capture_output=True, text=True, timeout=10
        )
        P(f"  curl github.com: rc={result.returncode}")
        for line in result.stdout.split('\n')[:5]:
            P(f"    {line}")
    except Exception as e:
        P(f"  curl error: {e}")

    try:
        result = subprocess.run(
            ['wget', '--version'],
            capture_output=True, text=True, timeout=5
        )
        P(f"  wget available: {'GNU Wget' in result.stdout}")
    except Exception as e:
        P(f"  wget check: {e}")

    # ==========================================
    # 6. Check if hipModuleLoad is available
    # ==========================================
    P("\n=== 6. HIP MODULE LOADING ===")
    try:
        # Check ctypes availability for HIP
        import ctypes
        hip_lib = None
        for lib_path in ['/opt/rocm/lib/libamdhip64.so',
                         '/opt/rocm/hip/lib/libamdhip64.so']:
            if os.path.exists(lib_path):
                try:
                    hip_lib = ctypes.CDLL(lib_path)
                    P(f"  libamdhip64.so loaded from {lib_path}")
                    # Check for hipModuleLoad symbol
                    if hasattr(hip_lib, 'hipModuleLoad'):
                        P(f"  hipModuleLoad: AVAILABLE")
                    if hasattr(hip_lib, 'hipModuleLoadData'):
                        P(f"  hipModuleLoadData: AVAILABLE")
                    if hasattr(hip_lib, 'hipModuleGetFunction'):
                        P(f"  hipModuleGetFunction: AVAILABLE")
                    break
                except Exception as e:
                    P(f"  Failed to load {lib_path}: {e}")
    except Exception as e:
        P(f"  HIP ctypes error: {e}")

    # ==========================================
    # 7. Check the CK module's kernel registry
    # ==========================================
    P("\n=== 7. CK KERNEL REGISTRY ===")
    try:
        # Try to list all available CK kernel names
        # The C++ lookup table should have them
        all_ops = sorted(dir(torch.ops.aiter))
        moe_ops = [a for a in all_ops if 'moe' in a.lower() or 'ck_moe' in a.lower()]
        P(f"  torch.ops.aiter MoE ops: {moe_ops}")

        # Try calling with invalid kernel name to see error message
        # (might reveal available kernels)
        P(f"\n  Attempting to trigger kernel list via invalid name...")
        try:
            # Dummy call to see what error we get
            dummy = torch.zeros(1, device='cuda')
            result = aiter.ck_moe_stage1_fwd(
                dummy, dummy, dummy, dummy, dummy, dummy, dummy,
                kernelName="INVALID_PROBE_KERNEL",
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                dtype=torch.bfloat16,
                splitk=0,
                use_non_temporal_load=False,
            )
        except Exception as e:
            err_msg = str(e)
            P(f"  Error (may reveal kernels): {err_msg[:500]}")
    except Exception as e:
        P(f"  Registry probe error: {e}")

    # ==========================================
    # 8. Check gen_instances.py on runner
    # ==========================================
    P("\n=== 8. GEN_INSTANCES CHECK ===")
    gen_path = None
    for candidate in [
        "/home/runner/aiter/csrc/ck_gemm_moe_2stages_codegen/gen_instances.py",
        "/home/runner/aiter/aiter/jit/core.py",
    ]:
        if os.path.isfile(candidate):
            P(f"  Found: {candidate}")
            gen_path = candidate
            with open(candidate) as f:
                content = f.read()
            # Search for .co loading
            for keyword in ['hipModuleLoad', '.co', 'MOE_STAGE2_ASM_DIR',
                           'code_object', 'kernel_path']:
                if keyword in content:
                    P(f"  '{keyword}' FOUND in {os.path.basename(candidate)}")
                    # Show context
                    for i, line in enumerate(content.split('\n')):
                        if keyword in line:
                            P(f"    L{i+1}: {line.strip()[:150]}")

    # ==========================================
    # 9. Measure time to import/load CK module
    # ==========================================
    P("\n=== 9. CK MODULE LOAD TIMING ===")
    try:
        from aiter.jit.core import get_module
        t0 = time.time()
        # This should already be cached
        mod = get_module('module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2_')
        t1 = time.time()
        P(f"  get_module time: {t1-t0:.3f}s")
        P(f"  Module file: {getattr(mod, '__file__', 'N/A')}")
        P(f"  Module dir: {sorted(dir(mod))[:30]}")
    except Exception as e:
        P(f"  get_module error: {e}")

    # Try alternative module name
    try:
        for mod_name in ['module_moe_ck2stages', 'module_moe_asm']:
            try:
                t0 = time.time()
                mod = get_module(mod_name)
                t1 = time.time()
                P(f"  {mod_name}: loaded in {t1-t0:.3f}s, file={getattr(mod, '__file__', 'N/A')}")
            except Exception as e:
                P(f"  {mod_name}: {str(e)[:100]}")
    except:
        pass

    # ==========================================
    # 10. Test downloading a file from GitHub
    # ==========================================
    P("\n=== 10. DOWNLOAD TEST ===")
    try:
        # Try to download the README (small file) to test
        test_url = "https://raw.githubusercontent.com/ROCm/aiter/main/README.md"
        result = subprocess.run(
            ['curl', '-sL', '--connect-timeout', '5', '-o', '/tmp/test_download.md', test_url],
            capture_output=True, text=True, timeout=15
        )
        if os.path.exists('/tmp/test_download.md'):
            sz = os.path.getsize('/tmp/test_download.md')
            P(f"  Download test: SUCCESS ({sz} bytes)")
            os.unlink('/tmp/test_download.md')
        else:
            P(f"  Download test: FAILED (no file)")
            P(f"  stderr: {result.stderr[:200]}")
    except Exception as e:
        P(f"  Download test error: {e}")

    P("\n=== PROBE COMPLETE ===")


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    fm.use_nt = lambda token, topk, expert: False

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
    STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    _patch()
    _probe()
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

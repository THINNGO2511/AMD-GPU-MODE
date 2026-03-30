#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Probe: dump the actual gemm_a16wfp4 config loading mechanism.
Find where configs are loaded from and what function selects them.
"""
from task import input_t, output_t
import torch
import os, sys, json, inspect

_probed = False


def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Probe gemm_a16wfp4 module
    try:
        from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as mod
        print(f"\n[PROBE] gemm_a16wfp4 module: {mod.__file__}")
        print(f"[PROBE] dir: {[x for x in dir(mod) if not x.startswith('__')]}")

        # Check for config-related functions
        for name in dir(mod):
            obj = getattr(mod, name)
            if callable(obj) and 'config' in name.lower():
                print(f"[PROBE] config func: {name}, sig: {inspect.signature(obj)}")
    except Exception as e:
        print(f"[PROBE] gemm_a16wfp4 import error: {e}")

    # 2. Probe config utils
    try:
        from aiter.ops.triton.utils import gemm_config_utils as gcu
        print(f"\n[PROBE] gemm_config_utils: {gcu.__file__}")
        print(f"[PROBE] dir: {[x for x in dir(gcu) if not x.startswith('__')]}")

        if hasattr(gcu, 'get_gemm_config'):
            sig = inspect.signature(gcu.get_gemm_config)
            print(f"[PROBE] get_gemm_config sig: {sig}")
            src = inspect.getsource(gcu.get_gemm_config)
            print(f"[PROBE] get_gemm_config source (first 500 chars):\n{src[:500]}")
    except Exception as e:
        print(f"[PROBE] config utils error: {e}")

    # 3. Probe config directories
    try:
        import aiter
        aiter_dir = os.path.dirname(aiter.__file__)
        config_dirs = [
            os.path.join(aiter_dir, 'ops', 'triton', 'configs', 'gemm'),
            os.path.join(aiter_dir, 'configs'),
        ]
        for d in config_dirs:
            if os.path.isdir(d):
                files = os.listdir(d)
                fp4_files = [f for f in files if 'FP4' in f.upper() or 'fp4' in f or 'a16w' in f.lower()]
                print(f"\n[PROBE] Config dir {d}: {len(files)} files, FP4-related: {fp4_files[:10]}")
                # Read one config file
                for cf in fp4_files[:2]:
                    fpath = os.path.join(d, cf)
                    try:
                        with open(fpath) as fh:
                            data = json.load(fh)
                            if isinstance(data, list):
                                print(f"[PROBE] {cf}: {len(data)} entries, first: {data[0] if data else 'empty'}")
                            elif isinstance(data, dict):
                                print(f"[PROBE] {cf}: keys={list(data.keys())[:5]}")
                    except:
                        print(f"[PROBE] {cf}: read error")
    except Exception as e:
        print(f"[PROBE] dir scan error: {e}")

    # 4. Probe gemm_afp4wfp4 config loading
    try:
        from aiter.ops.triton.gemm.basic import gemm_afp4wfp4 as mod2
        print(f"\n[PROBE] gemm_afp4wfp4 module: {mod2.__file__}")
        for name in dir(mod2):
            obj = getattr(mod2, name)
            if callable(obj) and ('config' in name.lower() or 'get' in name.lower()):
                try:
                    print(f"[PROBE] func: {name}, sig: {inspect.signature(obj)}")
                except:
                    print(f"[PROBE] func: {name} (no sig)")
    except Exception as e:
        print(f"[PROBE] gemm_afp4wfp4 error: {e}")

    # 5. Check env vars
    for key in sorted(os.environ.keys()):
        if 'AITER' in key or 'TRITON' in key or 'CONFIG' in key:
            print(f"[PROBE] env: {key}={os.environ[key]}")


def custom_kernel(data: input_t) -> output_t:
    _probe()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    A, B, B_q, B_shuffle, B_scale_sh = data
    # Standard path — just produce correct results
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    A_u8 = A_fp4.view(torch.uint8)
    # Unshuffle B scales
    s = B_scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)
    return gemm_afp4wfp4(A_u8, B_q.view(torch.uint8), A_scale, s, dtype=torch.bfloat16)

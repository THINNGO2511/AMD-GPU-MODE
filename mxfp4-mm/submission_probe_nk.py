#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM — Probe: list all gfx950 FP4 configs + dump _get_config source + test configs."""
from task import input_t, output_t
import torch, os, json, inspect

_probed = False

def _probe():
    global _probed
    if _probed: return
    _probed = True

    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    fp4_files = sorted([f for f in os.listdir(config_dir) if 'gfx950' in f and ('FP4' in f or 'AFP4' in f or 'PREQUANT' in f or 'a16wfp4' in f.lower())])
    print(f"\n[PROBE] gfx950 FP4 config files ({len(fp4_files)}):")
    for f in fp4_files:
        try:
            with open(os.path.join(config_dir, f)) as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    # Show first entry
                    first = data[keys[0]] if keys else {}
                    print(f"  {f}: keys={keys}, first_val={first}")
                elif isinstance(data, list):
                    print(f"  {f}: {len(data)} entries, [0]={data[0] if data else 'empty'}")
        except Exception as e:
            print(f"  {f}: error={e}")

    # Dump _get_config source
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import _get_config
        src = inspect.getsource(_get_config)
        print(f"\n[PROBE] _get_config source:\n{src[:2000]}")
    except Exception as e:
        print(f"\n[PROBE] _get_config source error: {e}")

    # Test what _get_config returns for our sizes
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import _get_config
        for m, n, k in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
            cfg = _get_config(m, n, k)
            print(f"[PROBE] _get_config(M={m},N={n},K={k}): {cfg}")
    except Exception as e:
        print(f"[PROBE] _get_config test error: {e}")

def custom_kernel(data: input_t) -> output_t:
    _probe()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    A, B, B_q, B_shuffle, B_scale_sh = data
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    s = B_scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)
    return gemm_afp4wfp4(A_fp4.view(torch.uint8), B_q.view(torch.uint8), A_scale, s, dtype=torch.bfloat16)

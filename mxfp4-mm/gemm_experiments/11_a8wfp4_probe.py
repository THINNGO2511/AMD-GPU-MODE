#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 11: Probe + test gemm_a8wfp4 (NEW DISCOVERY)
gemm_a8wfp4 takes fp8 A + fp4 B. fp8 A = 50% less bandwidth than bf16 A.
This could be significantly faster for memory-bound shapes.

Steps:
1. Probe the API signature, config, and requirements
2. Try to call it with our benchmark data
3. Fall back to a16wfp4 if it fails
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False
_a8wfp4_works = None  # None = untested, True/False = result
_a8wfp4_func = None
_y_cache = {}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe():
    global _probed, _a8wfp4_func
    if _probed:
        return
    _probed = True

    # Probe gemm_a8wfp4 module
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        _a8wfp4_func = gemm_a8wfp4
        import inspect
        try:
            sig = inspect.signature(gemm_a8wfp4)
            print("PROBE:A8WFP4_SIG:%s" % sig)
        except Exception:
            print("PROBE:A8WFP4_SIG:could_not_inspect")

        # Try to read source
        try:
            src = inspect.getsource(gemm_a8wfp4)
            lines = src.split('\n')
            print("PROBE:A8WFP4_SRC_LINES:%d" % len(lines))
            for line in lines[:40]:
                print("PROBE:A8WFP4_SRC:%s" % line.rstrip())
        except Exception as e:
            print("PROBE:A8WFP4_SRC_ERROR:%s" % e)
    except ImportError as e:
        print("PROBE:A8WFP4_IMPORT_ERROR:%s" % e)

    # Check set_use_gemm_splitk_bf16
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import set_use_gemm_splitk_bf16
        print("PROBE:A8WFP4_HAS_SPLITK_BF16:True")
    except ImportError:
        pass

    # Read the A8WFP4 config file
    import json, os
    cfg_path = "/home/runner/aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-A8WFP4.json"
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                cfg_data = json.load(f)
            print("PROBE:A8WFP4_CONFIG_KEYS:%s" % list(cfg_data.keys())[:20])
            # Print first few entries
            items = list(cfg_data.items())[:5]
            for k, v in items:
                print("PROBE:A8WFP4_CONFIG_ENTRY:%s=%s" % (k, v))
        except Exception as e:
            print("PROBE:A8WFP4_CONFIG_ERROR:%s" % e)
    else:
        print("PROBE:A8WFP4_CONFIG:NOT_FOUND")

    # Probe deepgemm
    try:
        from aiter import deepgemm
        import inspect
        print("PROBE:DEEPGEMM_TYPE:%s" % type(deepgemm))
        if callable(deepgemm):
            try:
                sig = inspect.signature(deepgemm)
                print("PROBE:DEEPGEMM_SIG:%s" % sig)
            except Exception:
                print("PROBE:DEEPGEMM_DIR:%s" % [x for x in dir(deepgemm) if not x.startswith('_')])
        else:
            print("PROBE:DEEPGEMM_DIR:%s" % [x for x in dir(deepgemm) if not x.startswith('_')])
    except Exception as e:
        print("PROBE:DEEPGEMM_ERROR:%s" % e)

    try:
        from aiter import deepgemm_ck
        import inspect
        print("PROBE:DEEPGEMM_CK_TYPE:%s" % type(deepgemm_ck))
        if callable(deepgemm_ck):
            try:
                sig = inspect.signature(deepgemm_ck)
                print("PROBE:DEEPGEMM_CK_SIG:%s" % sig)
            except Exception:
                pass
        else:
            print("PROBE:DEEPGEMM_CK_DIR:%s" % [x for x in dir(deepgemm_ck) if not x.startswith('_')])
    except Exception as e:
        print("PROBE:DEEPGEMM_CK_ERROR:%s" % e)

    # Probe per_1x32_f8_scale_f8_quant (might be needed for a8wfp4 A quant)
    try:
        from aiter import per_1x32_f8_scale_f8_quant
        import inspect
        sig = inspect.signature(per_1x32_f8_scale_f8_quant)
        print("PROBE:F8_QUANT_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:F8_QUANT_ERROR:%s" % e)

    # Also probe dynamic_per_token_scaled_quant
    try:
        from aiter import dynamic_per_token_scaled_quant
        import inspect
        sig = inspect.signature(dynamic_per_token_scaled_quant)
        print("PROBE:PERTOKEN_QUANT_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:PERTOKEN_QUANT_ERROR:%s" % e)

    # Probe gemm_a16wfp4_preshuffle (retry)
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        import inspect
        sig = inspect.signature(gemm_a16wfp4_preshuffle)
        print("PROBE:A16WFP4_PRESHUFFLE_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:A16WFP4_PRESHUFFLE_ERROR:%s" % e)

    # List f4gemm .co files
    import glob as globmod
    f4gemm_dir = "/home/runner/aiter/hsa/gfx950/f4gemm"
    import os
    if os.path.isdir(f4gemm_dir):
        files = sorted(os.listdir(f4gemm_dir))
        print("PROBE:F4GEMM_CO_COUNT:%d" % len(files))
        for f in files:
            print("PROBE:F4GEMM_CO:%s" % f)
    else:
        print("PROBE:F4GEMM_DIR:NOT_FOUND")

    print("PROBE:DONE")


def _try_a8wfp4(A, B_q_u8, B_scale_raw, m, n, k):
    """Try gemm_a8wfp4. Returns output tensor or None if it fails."""
    global _a8wfp4_works, _a8wfp4_func
    if _a8wfp4_works is False:
        return None
    if _a8wfp4_func is None:
        try:
            from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
            _a8wfp4_func = gemm_a8wfp4
        except ImportError:
            _a8wfp4_works = False
            return None

    try:
        # Try calling with same signature as a16wfp4 but with fp8 A
        # First quantize A to fp8
        from aiter import dynamic_per_token_scaled_quant

        # Per-token quant: A_fp8 (m, k) fp8, A_scale (m, 1) fp32
        A_fp8, A_scale = dynamic_per_token_scaled_quant(A)
        print("PROBE:A8WFP4_A_QUANT_SHAPE:%s %s" % (A_fp8.shape, A_scale.shape))
        print("PROBE:A8WFP4_A_QUANT_DTYPE:%s %s" % (A_fp8.dtype, A_scale.dtype))

        result = _a8wfp4_func(A_fp8, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
        _a8wfp4_works = True
        print("PROBE:A8WFP4_SUCCESS:shape=%s dtype=%s m=%d n=%d k=%d" %
              (result.shape, result.dtype, m, n, k))
        return result
    except TypeError as e:
        print("PROBE:A8WFP4_TYPEERROR:%s" % e)
        # Maybe different signature - try with A_scale
        try:
            result = _a8wfp4_func(A_fp8, B_q_u8, A_scale, B_scale_raw, dtype=torch.bfloat16)
            _a8wfp4_works = True
            print("PROBE:A8WFP4_SUCCESS_V2:shape=%s" % (result.shape,))
            return result
        except Exception as e2:
            print("PROBE:A8WFP4_V2_ERROR:%s" % e2)
    except Exception as e:
        print("PROBE:A8WFP4_ERROR:%s %s" % (type(e).__name__, e))

    _a8wfp4_works = False
    return None


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _probe()

    # Try a8wfp4 first (only for non-K=1536 shapes)
    if k != 1536 and _a8wfp4_works is not False:
        result = _try_a8wfp4(A, _bq_u8, _bscale_raw, m, n, k)
        if result is not None:
            return result

    # Fallback to standard path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
    } if k == 7168 else None

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                       y=_y_cache[key], config=cfg)

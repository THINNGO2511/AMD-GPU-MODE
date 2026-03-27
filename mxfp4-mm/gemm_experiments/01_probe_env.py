#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 01: Environment Probe
Discover new APIs, kernels, and library versions on the runner.
Produces correct GEMM output while printing probe data to stdout.
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe():
    global _probed
    if _probed:
        return
    _probed = True
    import sys

    # aiter version and top-level API
    try:
        import aiter
        print("PROBE:AITER_VERSION:%s" % getattr(aiter, '__version__', 'unknown'))
        print("PROBE:AITER_DIR:%s" % [x for x in dir(aiter) if not x.startswith('_')])
    except Exception as e:
        print("PROBE:AITER_IMPORT_ERROR:%s" % e)

    # Check for new GEMM kernels
    try:
        import aiter.ops.triton.gemm.basic as gb
        print("PROBE:GEMM_BASIC_DIR:%s" % [x for x in dir(gb) if not x.startswith('_')])
    except Exception as e:
        print("PROBE:GEMM_BASIC_ERROR:%s" % e)

    # Check for gemm_a8wfp4 or other new variants
    for name in ['gemm_a8wfp4', 'gemm_a16wfp4_v2', 'gemm_fp4', 'gemm_mxfp4',
                 'gemm_a4w4_v2', 'gemm_scaled', 'gemm_fused']:
        try:
            mod = __import__('aiter.ops.triton.gemm.basic.%s' % name, fromlist=[name])
            print("PROBE:FOUND_KERNEL:%s dir=%s" % (name, [x for x in dir(mod) if not x.startswith('_')]))
        except ImportError:
            pass

    # Check for CK/ASM GEMM variants
    try:
        from aiter import gemm_a4w4, gemm_a4w4_blockscale_tune
        print("PROBE:CK_A4W4:available")
        # Check tune function signature
        import inspect
        sig = inspect.signature(gemm_a4w4_blockscale_tune)
        print("PROBE:CK_TUNE_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:CK_A4W4_ERROR:%s" % e)

    # List .co files for GEMM
    import glob as globmod
    co_patterns = [
        "/home/runner/aiter/hsa/gfx950/gemm*.co",
        "/home/runner/aiter/hsa/gfx950/*fp4*gemm*.co",
        "/home/runner/aiter/hsa/gfx950/*mxfp4*.co",
    ]
    all_co = set()
    for pat in co_patterns:
        all_co.update(globmod.glob(pat))
    co_list = sorted(all_co)
    print("PROBE:GEMM_CO_COUNT:%d" % len(co_list))
    for f in co_list[:30]:
        print("PROBE:CO_FILE:%s" % os.path.basename(f))

    # Check gemm_a16wfp4 source for new features
    try:
        from aiter.ops.triton.gemm.basic import gemm_a16wfp4 as m
        print("PROBE:A16WFP4_DIR:%s" % [x for x in dir(m) if not x.startswith('_')])
        # Check if there's a tune/config function
        for attr in ['get_config', 'get_autotune_config', 'tune', 'best_config',
                     'CONFIGS', 'DEFAULT_CONFIG', 'autotune_configs']:
            if hasattr(m, attr):
                print("PROBE:A16WFP4_HAS:%s" % attr)
    except Exception as e:
        print("PROBE:A16WFP4_ERROR:%s" % e)

    # Check for hipBLASLt or rocBLAS FP4 support
    try:
        import hipblas
        print("PROBE:HIPBLAS:available")
    except ImportError:
        pass
    try:
        from aiter import hipb_mm, hipb_fp4_mm
        print("PROBE:HIPB_FP4:available")
    except ImportError:
        pass

    # Check Triton version
    try:
        import triton
        print("PROBE:TRITON_VERSION:%s" % triton.__version__)
    except Exception:
        pass

    # Check ROCm version
    try:
        print("PROBE:TORCH_VERSION:%s" % torch.__version__)
        print("PROBE:TORCH_HIP:%s" % torch.version.hip)
    except Exception:
        pass

    # Check available configs for our benchmark shapes
    import os
    cfg_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    if os.path.isdir(cfg_dir):
        cfgs = sorted(os.listdir(cfg_dir))
        fp4_cfgs = [c for c in cfgs if 'fp4' in c.lower() or 'FP4' in c]
        print("PROBE:CONFIG_FILES_TOTAL:%d" % len(cfgs))
        print("PROBE:CONFIG_FILES_FP4:%d" % len(fp4_cfgs))
        for c in fp4_cfgs[:20]:
            print("PROBE:CFG:%s" % c)

    # Check for new quant functions
    try:
        from aiter.ops.triton import quant
        print("PROBE:QUANT_DIR:%s" % [x for x in dir(quant) if not x.startswith('_')])
    except Exception:
        pass

    # Check for torch.compile or inductor support
    try:
        compiled = torch.compile is not None
        print("PROBE:TORCH_COMPILE:available")
    except Exception:
        pass

    print("PROBE:DONE")


import os


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

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
        "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
    } if k == 7168 else None

    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)

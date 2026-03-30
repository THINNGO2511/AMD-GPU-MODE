#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Experiment 09: Deep probe of CK GEMM capabilities
Dump all available CK kernel configs, .co file names, CSV entries,
and gemm_a4w4 internals. Maybe there are new kernels or config options
we haven't discovered.

Also probes for any hip/rocBLAS MXFP4 GEMM support.
"""
from task import input_t, output_t
import torch
import os

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

    import glob as globmod

    # List ALL .co files in gfx950 directory
    co_dir = "/home/runner/aiter/hsa/gfx950"
    if os.path.isdir(co_dir):
        all_co = sorted(os.listdir(co_dir))
        gemm_co = [f for f in all_co if 'gemm' in f.lower() or 'fp4' in f.lower()
                    or 'mm' in f.lower() or 'matmul' in f.lower()]
        print("PROBE:TOTAL_CO_FILES:%d" % len(all_co))
        print("PROBE:GEMM_CO_FILES:%d" % len(gemm_co))
        for f in gemm_co:
            print("PROBE:GEMM_CO:%s" % f)

    # Read CK GEMM CSV configs
    csv_paths = globmod.glob("/home/runner/aiter/aiter/configs/**/*gemm*.csv", recursive=True)
    csv_paths += globmod.glob("/home/runner/aiter/aiter/configs/**/*fp4*.csv", recursive=True)
    csv_paths += globmod.glob("/home/runner/aiter/aiter/configs/**/*mm*.csv", recursive=True)
    csv_paths = list(set(csv_paths))
    for csv_path in csv_paths[:5]:
        print("PROBE:CSV_FILE:%s" % csv_path)
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            print("PROBE:CSV_LINES:%d" % len(lines))
            # Print header and first few rows
            for line in lines[:3]:
                print("PROBE:CSV_DATA:%s" % line.strip()[:200])
            # Print rows matching our shapes
            for line in lines:
                for m_val in ['4,', '16,', '32,', '64,', '256,']:
                    if m_val in line:
                        print("PROBE:CSV_MATCH:%s" % line.strip()[:200])
                        break
        except Exception as e:
            print("PROBE:CSV_ERROR:%s %s" % (csv_path, e))

    # Check gemm_a4w4 internals
    try:
        from aiter import gemm_a4w4
        import inspect
        sig = inspect.signature(gemm_a4w4)
        print("PROBE:A4W4_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:A4W4_ERROR:%s" % e)

    # Check gemm_a4w4_blockscale_tune
    try:
        from aiter import gemm_a4w4_blockscale_tune
        import inspect
        sig = inspect.signature(gemm_a4w4_blockscale_tune)
        print("PROBE:A4W4_TUNE_SIG:%s" % sig)
    except Exception as e:
        print("PROBE:A4W4_TUNE_ERROR:%s" % e)

    # Check for gemm_op_a4w4 source
    try:
        from aiter.ops.ck import gemm_op_a4w4
        print("PROBE:GEMM_OP_A4W4_DIR:%s" % [x for x in dir(gemm_op_a4w4) if not x.startswith('_')])
        # Check for get_GEMM_config
        if hasattr(gemm_op_a4w4, 'get_GEMM_config'):
            import inspect
            src = inspect.getsource(gemm_op_a4w4.get_GEMM_config)
            print("PROBE:GET_GEMM_CONFIG_SRC_LINES:%d" % len(src.split('\n')))
            for line in src.split('\n')[:15]:
                print("PROBE:GET_GEMM_CONFIG:%s" % line.rstrip())
    except Exception as e:
        print("PROBE:GEMM_OP_ERROR:%s" % e)

    # Check for any new top-level GEMM functions
    try:
        import aiter
        gemm_funcs = [x for x in dir(aiter) if 'gemm' in x.lower() or 'mm' in x.lower()
                      or 'matmul' in x.lower()]
        print("PROBE:AITER_GEMM_FUNCS:%s" % gemm_funcs)
    except Exception:
        pass

    # Check for hipblaslt
    try:
        import hipblaslt
        print("PROBE:HIPBLASLT:available")
        print("PROBE:HIPBLASLT_DIR:%s" % [x for x in dir(hipblaslt) if not x.startswith('_')])
    except ImportError:
        print("PROBE:HIPBLASLT:not_found")

    # Check torch._C for FP4 ops
    try:
        fp4_ops = [x for x in dir(torch.ops) if 'fp4' in str(x).lower() or 'mxfp' in str(x).lower()]
        print("PROBE:TORCH_FP4_OPS:%s" % fp4_ops)
    except Exception:
        pass

    # Check aiter.ops for new modules
    try:
        import aiter.ops as aops
        print("PROBE:AITER_OPS_DIR:%s" % [x for x in dir(aops) if not x.startswith('_')])
    except Exception:
        pass

    # Check Triton gemm directory for new kernel files
    triton_gemm_dir = "/home/runner/aiter/aiter/ops/triton/gemm/basic"
    if os.path.isdir(triton_gemm_dir):
        files = sorted(os.listdir(triton_gemm_dir))
        print("PROBE:TRITON_GEMM_FILES:%s" % files)

    print("PROBE:DONE")


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

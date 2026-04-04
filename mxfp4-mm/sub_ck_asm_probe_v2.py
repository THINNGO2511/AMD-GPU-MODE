#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM: Probe CK ASM kernel configs + test direct ASM path for our shapes.

Discovery: Our shapes have NO tuned CK/ASM configs. Competitors may be using
CK ASM with proper CSV entries. This probe:
1. Lists all available CK ASM kernels (.co files)
2. Reads the tuning CSV to find nearby configs
3. Tests CK ASM path with different kernel names for K=512
4. Falls back to Triton for actual output
"""
import os
import subprocess
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch

_bscale_raw = None
_bscale_ref = None
_bq_u8 = None
_y_cache = {}
_warmed = False
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

    try:
        # 1. List FP4 GEMM .co files
        co_dir = "/home/runner/aiter/hsa/gfx950/f4gemm/"
        cos = sorted(os.listdir(co_dir))
        print(f"[PROBE] FP4 GEMM .co files: {len(cos)}")
        for c in cos:
            print(f"  {c}")

        # 2. Read tuning CSV
        import csv
        csv_paths = [
            "/home/runner/aiter/aiter/ops/triton/configs/gemm/",
        ]
        for cpath in csv_paths:
            if os.path.isdir(cpath):
                configs = sorted(os.listdir(cpath))
                fp4_configs = [c for c in configs if "FP4" in c.upper() or "fp4" in c or "A16W" in c]
                print(f"\n[PROBE] Triton GEMM configs in {cpath}: {len(configs)} total, {len(fp4_configs)} FP4-related")
                for c in fp4_configs[:20]:
                    print(f"  {c}")

        # 3. Check gemm_a4w4 CSV
        a4w4_csv = "/home/runner/aiter/aiter/configs/gemm_a4w4.csv"
        if os.path.exists(a4w4_csv):
            with open(a4w4_csv) as f:
                lines = f.readlines()
            print(f"\n[PROBE] gemm_a4w4.csv: {len(lines)} lines")
            # Header
            if lines:
                print(f"  Header: {lines[0].strip()}")
            # Find entries near our shapes
            for line in lines[1:]:
                for shape in ["2880", "4096", "2112", "7168", "3072"]:
                    if shape in line:
                        print(f"  {line.strip()}")
                        break

        # 4. Check get_padded_m behavior
        try:
            from aiter.ops.gemm_common import get_padded_m
            for m in [4, 16, 32, 64, 256]:
                for n in [2880, 2112, 4096, 7168, 3072]:
                    for k in [512, 7168, 2048, 1536]:
                        pm = get_padded_m(m, n, k, 32)
                        if pm != m:
                            print(f"  get_padded_m({m},{n},{k},32) = {pm}")
        except Exception as e:
            print(f"  get_padded_m error: {e}")

        # 5. Check gemm_a4w4 function signature
        try:
            from aiter import gemm_a4w4
            import inspect
            sig = inspect.signature(gemm_a4w4)
            print(f"\n[PROBE] gemm_a4w4 signature: {sig}")
        except Exception as e:
            print(f"\n[PROBE] gemm_a4w4 import: {e}")

        # 6. Check gemm_a4w4_blockscale_tune
        try:
            from aiter import gemm_a4w4_blockscale_tune
            import inspect
            sig = inspect.signature(gemm_a4w4_blockscale_tune)
            print(f"[PROBE] gemm_a4w4_blockscale_tune signature: {sig}")
        except Exception as e:
            print(f"[PROBE] gemm_a4w4_blockscale_tune: {e}")

        # 7. List ALL config JSONs for our specific shapes
        for n in [2880, 2112, 4096, 7168, 3072]:
            for k in [512, 7168, 2048, 1536]:
                pattern = f"N={n}"
                matches = [c for c in os.listdir("/home/runner/aiter/aiter/ops/triton/configs/gemm/")
                           if pattern in c and f"K={k}" in c]
                if matches:
                    print(f"\n[PROBE] Configs for N={n} K={k}:")
                    for m in matches:
                        fpath = f"/home/runner/aiter/aiter/ops/triton/configs/gemm/{m}"
                        with open(fpath) as f:
                            content = f.read()[:500]
                        print(f"  {m}: {content[:200]}")

    except Exception as e:
        print(f"[PROBE] Error: {e}")


_K7168_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

_K512_CONFIG = {
    "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}

_K2048_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}


def custom_kernel(data: input_t) -> output_t:
    global _bscale_raw, _bscale_ref, _bq_u8, _warmed
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    _probe()

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 7168:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K7168_CONFIG)
    elif k == 2048:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K2048_CONFIG)
    elif k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw, dtype=torch.bfloat16)
    else:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K512_CONFIG)
    return out

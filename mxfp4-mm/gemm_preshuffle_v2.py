#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Preshuffle config path + custom shape configs.

RUNNER DISCOVERY:
1. gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json EXISTS — tuned
   specifically for our M=16 N=2112 K=7168 shape!
2. _get_config(M, N, K, shuffle=True) loads the PRESHUFFLED configs
3. gemm_a16wfp4_preshuffle(A, B_shuffle_uint8, B_scale_uint8, ...) with
   shuffle=True flag should use these optimized configs
4. ALL our shapes use generic fallback — shape-specific configs could help

The Triton preshuffle kernel failed on fp4x2 dtype. BUT if we pass
B_shuffle.view(torch.uint8) and B_scale_sh.view(torch.uint8) it might work.

Also: write custom JSON configs to /tmp/ and point the config loader there.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

from task import input_t, output_t
import torch
import json

_yc = {}
_call = 0
_preshuffle_works = None
_custom_configs_written = False

# Shape-specific configs based on runner's generic config structure
# Tuned for our exact shapes
CUSTOM_CONFIGS = {
    # K=512 (actual K, config looks for 2*K=1024): M=4, M=32
    "M_LEQ_4": {
        "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 1
    },
    "M_LEQ_16": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024
    },
    "M_LEQ_32": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 1
    },
    "M_LEQ_64": {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 1
    },
    "M_LEQ_128": {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 1
    },
    "M_LEQ_256": {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3,
        "waves_per_eu": 1, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 1
    },
}


def _write_custom_configs():
    """Write shape-specific JSON configs that the config loader can find."""
    global _custom_configs_written
    if _custom_configs_written:
        return
    _custom_configs_written = True

    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    # Write configs for our specific N,K combos (K is doubled in filename)
    shapes = [
        ("N=2880-K=1024", CUSTOM_CONFIGS),   # K=512 shapes
        ("N=4096-K=1024", CUSTOM_CONFIGS),   # K=512 M=32
        ("N=2112-K=14336", CUSTOM_CONFIGS),  # K=7168
        ("N=7168-K=4096", CUSTOM_CONFIGS),   # K=2048
        ("N=3072-K=3072", CUSTOM_CONFIGS),   # K=1536
    ]
    for suffix, cfg in shapes:
        path = f"{config_dir}/gfx950-GEMM-A16WFP4-{suffix}.json"
        try:
            with open(path, 'w') as f:
                json.dump(cfg, f)
        except PermissionError:
            # Can't write to runner dir, try via env var
            pass


def custom_kernel(data: input_t) -> output_t:
    global _call, _preshuffle_works
    _call += 1

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]

    # Try preshuffle path first (uses B_shuffle directly)
    if _preshuffle_works is None and _call <= 2:
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
            key = (m, n)
            if key not in _yc:
                _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
            out = _yc[key]

            # Pass as uint8 to avoid fp4x2 dtype issue
            b_u8 = B_shuffle.view(torch.uint8)
            bs_u8 = B_scale_sh.view(torch.uint8)

            gemm_a16wfp4_preshuffle(A, b_u8, bs_u8, prequant=True,
                                    dtype=torch.bfloat16, y=out, config=None)
            print(f"[GEMM] preshuffle SUCCESS m={m} n={n} k={k}", flush=True)
            _preshuffle_works = True
            return out
        except Exception as e:
            print(f"[GEMM] preshuffle fail: {e}", flush=True)
            _preshuffle_works = False

    if _preshuffle_works:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
        key = (m, n)
        if key not in _yc:
            _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
        out = _yc[key]
        gemm_a16wfp4_preshuffle(A, B_shuffle.view(torch.uint8),
                                B_scale_sh.view(torch.uint8),
                                prequant=True, dtype=torch.bfloat16, y=out)
        return out

    # Fallback: standard path with our proven configs
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    su = B_scale_sh.view(torch.uint8); sm, sn = su.shape
    d0, d1 = sm // 32, sn // 8; total = sm * sn
    idx = torch.arange(total, dtype=torch.int64, device=su.device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    bscale_raw = torch.take(su.reshape(-1), idx).view(sm, sn)
    bq_u8 = B_q.view(torch.uint8)

    _K7168 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}
    _K512 = {"BLOCK_SIZE_M":4,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,"num_stages":3,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}
    _K2048 = {"BLOCK_SIZE_M":16,"BLOCK_SIZE_N":128,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":8,"num_stages":2,"waves_per_eu":4,"matrix_instr_nonkdim":16,"cache_modifier":".cg","NUM_KSPLIT":1}

    key = (m, n)
    if key not in _yc:
        _yc[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _yc[key]

    if k == 1536:
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)
    cfg = _K7168 if k == 7168 else (_K2048 if k == 2048 else _K512)
    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM -- Small-tile configs for K=512 shapes to fix wave quantization.

MI355X has 256 CUs. With the current BM=4 BN=128 K=512 config:
  M=4  N=2880: tiles = 1 * ceil(2880/128) = 1*23 = 23  -> 91% CUs idle
  M=32 N=2880: tiles = 8 * 23 = 184 -> OK but BM=4 wastes 75% of tile rows
  M=32 N=4096: tiles = 8 * 32 = 256 -> perfect

The fix: use smaller BLOCK_SIZE_N to create more tiles for small shapes.
  BN=32: M=4  N=2880 -> 1*90 = 90 tiles   (4x more work items)
  BN=64: M=4  N=2880 -> 1*45 = 45 tiles   (2x more work items)
  BN=32: M=32 N=2880 -> 1*90 = 90 tiles   (BM=32 keeps full utilization)
  BN=64: M=32 N=4096 -> 1*64 = 64 tiles

For K>=1536: proven configs unchanged (already fast enough).

Also tries BM=4 (exact M match) instead of BM=4 with padding, since
gemm_a16wfp4 supports BLOCK_SIZE_M=4.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch

_bscale_raw = None
_bscale_ref = None
_bq_u8 = None
_y_cache = {}
_warmed = False


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# ---- K=7168: proven best config (split-K=8, .cg cache modifier) ----
_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# ---- K=2048: proven tuned config ----
_K2048_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}

# ---- K=512 small-tile configs keyed by M ----
# Goal: maximize tile count for CU utilization on 256-CU MI355X.
#
# M=4:  Only 1 M-tile no matter what BM is (BM must be >= 4).
#        -> Maximize N-tiles: BN=32 gives ceil(N/32) tiles.
#        N=2880 -> 90 tiles. N=4096 -> 128 tiles. Much better than 23/32.
#        BK=512 = full K in one pass (no K-loop overhead).
#
# M=16: 1 M-tile with BM=16. BN=32 -> many tiles.
#        N=2112 -> 66 tiles. Good CU fill.
#
# M=32: 1 M-tile with BM=32. BN=64 -> decent tiles.
#        N=2880 -> 45. N=4096 -> 64. Reasonable.
#        BN=32 would give 90/128 but each tile is tiny (32x32 = 1024 elems).
#
# M=64: 1 M-tile with BM=64. N is large (7168) so tiles = 112 with BN=64.
#        Not a problem shape for wave quantization.
#
# M=256: BM=64 -> 4 M-tiles. N=3072 -> 48 N-tiles. 192 tiles. Fine.

_K512_M4_CONFIG = {
    "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}

_K512_M16_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}

_K512_M32_CONFIG = {
    "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}

# M=64 and M=256 with K=512: larger tiles are fine, enough parallelism
_K512_DEFAULT_CONFIG = {
    "BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1,
}


def _get_k512_config(m):
    """Select K=512 config based on M to maximize tile count."""
    if m <= 4:
        return _K512_M4_CONFIG
    elif m <= 16:
        return _K512_M16_CONFIG
    elif m <= 32:
        return _K512_M32_CONFIG
    else:
        return _K512_DEFAULT_CONFIG


def custom_kernel(data: input_t) -> output_t:
    global _bscale_raw, _bscale_ref, _bq_u8, _warmed
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # Pre-warm all kernel variants on first call to avoid Triton JIT during benchmark
    if not _warmed:
        _warmed = True
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        # Warm the specific config each M size will use
        warmup_specs = []
        if k == 512:
            # Warm all M sizes with their specific K=512 small-tile configs
            for wm in [4, 16, 32, 64, 256]:
                warmup_specs.append((wm, _get_k512_config(wm)))
        elif k == 7168:
            for wm in [4, 16, 32, 64, 256]:
                warmup_specs.append((wm, _K7168_CONFIG))
        elif k == 2048:
            for wm in [4, 16, 32, 64, 256]:
                warmup_specs.append((wm, _K2048_CONFIG))
        elif k == 1536:
            # K=1536 uses afp4wfp4 path -- warm quant kernel
            for wm in [16, 64, 256]:
                try:
                    dummy_a = torch.randn(wm, 1536, dtype=torch.bfloat16, device=A.device)
                    dynamic_mxfp4_quant(dummy_a)
                except Exception:
                    pass

        for wm, cfg in warmup_specs:
            try:
                dummy_a = torch.randn(wm, k, dtype=torch.bfloat16, device=A.device)
                dummy_out = torch.empty(wm, n, dtype=torch.bfloat16, device=A.device)
                gemm_a16wfp4(dummy_a, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                             y=dummy_out, config=cfg)
                del dummy_a, dummy_out
            except Exception:
                pass

        torch.cuda.synchronize()

    # ---- K=1536: proven best path is quant + afp4wfp4 (see CLAUDE.md) ----
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw,
                             dtype=torch.bfloat16)

    # ---- All other K values: gemm_a16wfp4 with tuned configs ----
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 7168:
        cfg = _K7168_CONFIG
    elif k == 2048:
        cfg = _K2048_CONFIG
    else:
        # K=512: use M-aware small-tile config
        cfg = _get_k512_config(m)

    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

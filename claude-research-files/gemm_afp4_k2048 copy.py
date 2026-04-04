#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Try afp4wfp4 (separate quant+GEMM) for K=2048 in addition to K=1536.
Rationale: afp4wfp4 has 69 tuned gfx950 FP4 config files.
These configs may be better-tuned than a16wfp4 defaults for certain shapes.

The tradeoff: 2 kernel launches (quant + GEMM) vs 1 fused (a16wfp4).
For K=1536 this was a clear win. For K=2048, it's untested.

K=512 stays with a16wfp4 (quant overhead would dominate the short kernel).
K=7168 stays with a16wfp4 + tuned config.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

from task import input_t, output_t
import torch

_gather_cache = {}
_bscale_ref = None
_bq_u8 = None
_bscale_raw = None
_scale_shape = None
_y_cache = {}
_warmed = False

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

_ALL_SHAPES = [
    (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
    (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
]


def _build_gather_cache(sm, sn, device):
    total = sm * sn
    d0, d1 = sm // 32, sn // 8
    idx = torch.arange(total, dtype=torch.int64, device=device)
    idx = idx.view(d0, d1, 4, 16, 2, 2).permute(0, 5, 3, 1, 4, 2).contiguous().view(-1)
    out_buf = torch.empty(total, dtype=torch.uint8, device=device)
    return idx, out_buf


def _fast_unshuffle(scale_sh_u8_flat, sm, sn):
    gather_idx, out_buf = _gather_cache[(sm, sn)]
    torch.take(scale_sh_u8_flat, gather_idx, out=out_buf)
    return out_buf.view(sm, sn)


def _full_prewarm(device):
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4

    for m, n, k in _ALL_SHAPES:
        try:
            dummy_a = torch.randn(m, k, dtype=torch.bfloat16, device=device)
            dummy_bq = torch.zeros(n, k // 2, dtype=torch.uint8, device=device)
            if k in (1536, 2048):
                # afp4wfp4 path: prewarm quant + GEMM
                af, asc = dynamic_mxfp4_quant(dummy_a)
                dummy_bs = torch.full((n, k // 32), 127, dtype=torch.uint8, device=device)
                gemm_afp4wfp4(af.view(torch.uint8), dummy_bq, asc, dummy_bs,
                              dtype=torch.bfloat16)
            else:
                # a16wfp4 path
                pad_n = ((n + 31) // 32) * 32
                dummy_bs = torch.full((pad_n, k // 32), 127, dtype=torch.uint8, device=device)
                dummy_out = torch.empty(m, n, dtype=torch.bfloat16, device=device)
                cfg = _K7168_CONFIG if k == 7168 else _K512_CONFIG
                gemm_a16wfp4(dummy_a, dummy_bq, dummy_bs, dtype=torch.bfloat16,
                             y=dummy_out, config=cfg)
            del dummy_a
        except Exception:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bq_u8, _bscale_raw, _scale_shape

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        s_u8 = B_scale_sh.view(torch.uint8)
        sm, sn = s_u8.shape
        _scale_shape = (sm, sn)
        if _scale_shape not in _gather_cache:
            _gather_cache[_scale_shape] = _build_gather_cache(sm, sn, B_scale_sh.device)
        _bscale_raw = _fast_unshuffle(s_u8.reshape(-1), sm, sn)
        _bq_u8 = B_q.view(torch.uint8)

    _full_prewarm(A.device)

    # K=1536 and K=2048: use separate quant + afp4wfp4 (tuned configs)
    if k in (1536, 2048):
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw,
                             dtype=torch.bfloat16)

    # K=512, K=7168: use a16wfp4 (fused quant)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 7168:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K7168_CONFIG)
    else:
        gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=_K512_CONFIG)
    return out

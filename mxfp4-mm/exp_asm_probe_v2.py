"""
GEMM ASM Probe v2: Read CSV config, understand kernel naming, try direct ASM calls.
Focus on understanding the gemm_a4w4_asm API before attempting to use it.
Falls back to working Triton path for correctness.
"""
from task import input_t, output_t
import torch
import sys
import os

_probed = False
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


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

    P = lambda *a: print(f"PROBE: {' '.join(str(x) for x in a)}", file=sys.stderr)

    try:
        import aiter

        # 1. Read the f4gemm CSV
        csv_path = "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4.csv"
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                lines = f.readlines()
            P(f"CSV has {len(lines)} lines")
            for line in lines[:15]:
                P(f"CSV: {line.strip()}")
            if len(lines) > 15:
                P(f"CSV: ... {len(lines)-15} more lines")

        # 2. Check get_padded_m for our shapes
        shapes = [(4,2880,512), (16,2112,7168), (32,4096,512),
                  (32,2880,512), (64,7168,2048), (256,3072,1536)]
        for m, n, k in shapes:
            try:
                pm = aiter.get_padded_m(m, n, k, 0)
                P(f"get_padded_m({m},{n},{k},0) = {pm}")
            except Exception as e:
                P(f"get_padded_m({m},{n},{k}) error: {e}")

        # 3. Check get_GEMM_config for our shapes
        try:
            for m, n, k in shapes:
                cfg = aiter.get_GEMM_config(m, n, k)
                P(f"get_GEMM_config({m},{n},{k}) = {cfg}")
        except Exception as e:
            P(f"get_GEMM_config error: {e}")

        # 4. Try get_GEMM_config_with_quant_type
        try:
            from aiter import QuantType
            for m, n, k in shapes[:2]:
                cfg = aiter.get_GEMM_config_with_quant_type(m, n, k, QuantType.per_1x32)
                P(f"get_GEMM_config_with_quant({m},{n},{k}) = {cfg}")
        except Exception as e:
            P(f"get_GEMM_config_with_quant error: {e}")

        # 5. Check deepgemm source
        try:
            import inspect
            src = inspect.getsource(aiter.deepgemm)
            P(f"deepgemm source ({len(src)} chars):")
            for line in src.split('\n')[:30]:
                P(f"  {line}")
        except Exception as e:
            P(f"deepgemm source error: {e}")

        # 6. Check gemm_a4w4_asm JIT type hints
        try:
            src = inspect.getsource(aiter.gemm_a4w4_asm)
            P(f"gemm_a4w4_asm source ({len(src)} chars):")
            for line in src.split('\n')[:20]:
                P(f"  {line}")
        except Exception as e:
            P(f"gemm_a4w4_asm source: {e}")

    except Exception as e:
        P(f"Probe error: {e}")


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

    # Use working Triton path for correctness
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
    cfg = _K7168_CONFIG if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                        y=_y_cache[key], config=cfg)

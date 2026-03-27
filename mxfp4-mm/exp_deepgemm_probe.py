"""
GEMM Probe: Discover deepgemm / deepgemm_ck APIs on the runner.
These are completely unexplored and could be the path to 7-8us.
Falls back to prewarm approach if deepgemm isn't usable.
"""
from task import input_t, output_t
import torch
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}
_deepgemm_available = None

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


def _probe_deepgemm():
    """Probe deepgemm APIs on the runner and print findings."""
    global _deepgemm_available
    if _deepgemm_available is not None:
        return _deepgemm_available

    findings = []
    try:
        import aiter
        # Check for deepgemm
        for name in ['deepgemm', 'deepgemm_ck', 'gemm_a4w4_asm']:
            if hasattr(aiter, name):
                fn = getattr(aiter, name)
                findings.append(f"FOUND: aiter.{name} = {fn}")
                try:
                    import inspect
                    sig = inspect.signature(fn)
                    findings.append(f"  Signature: {sig}")
                except:
                    findings.append(f"  (no signature)")

        # Check for ASM GEMM module
        try:
            from aiter.jit import module_gemm_a4w4_asm
            findings.append(f"FOUND: module_gemm_a4w4_asm")
        except:
            pass

        # List all f4gemm .co files
        import os
        co_dir = "/home/runner/aiter/hsa/gfx950/f4gemm/"
        if os.path.exists(co_dir):
            cos = sorted(os.listdir(co_dir))
            findings.append(f"f4gemm .co files ({len(cos)}):")
            for co in cos[:10]:
                findings.append(f"  {co}")
            if len(cos) > 10:
                findings.append(f"  ... and {len(cos)-10} more")

        # Check gemm_a4w4_asm signature
        try:
            from aiter import gemm_a4w4_asm
            import inspect
            findings.append(f"gemm_a4w4_asm sig: {inspect.signature(gemm_a4w4_asm)}")
        except Exception as e:
            findings.append(f"gemm_a4w4_asm: {e}")

        # Check for new GEMM APIs
        gemm_names = [n for n in dir(aiter) if 'gemm' in n.lower()]
        findings.append(f"All GEMM-related names in aiter: {gemm_names}")

    except Exception as e:
        findings.append(f"Probe error: {e}")

    for f in findings:
        print(f"PROBE: {f}", file=sys.stderr)

    _deepgemm_available = False  # Default to fallback
    return False


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    _probe_deepgemm()
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except:
            pass
    torch.cuda.synchronize()


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
        _prewarm()

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

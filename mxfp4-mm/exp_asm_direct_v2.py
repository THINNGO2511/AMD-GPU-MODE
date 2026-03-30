"""
GEMM: Hybrid Triton (a16wfp4) + ASM direct for shapes with tuned configs.
Probe found tuned ASM configs for:
  (64,7168,2048) → kernel 32x128, 6.8μs claimed
  (256,3072,1536) → kernel 32x128, 6.2μs claimed

For these shapes, try gemm_a4w4_asm directly with the tuned kernel.
For other shapes, use gemm_a16wfp4 (proven fastest).
"""
from task import input_t, output_t
import torch
import sys

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_warmed = False
_y_cache = {}

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

# ASM kernel name for shapes with tuned config
ASM_KERNEL_32x128 = "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _prewarm():
    global _warmed
    if _warmed:
        return
    _warmed = True
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    for wm in (4, 16, 32, 64, 256):
        try:
            wA = torch.randn((wm, 1536), dtype=torch.bfloat16, device='cuda')
            dynamic_mxfp4_quant(wA)
        except:
            pass

    # Also warm ASM path for shapes that use it
    import aiter
    from aiter import dtypes
    from aiter.utility.fp4_utils import e8m0_shuffle
    from aiter.ops.shuffle import shuffle_weight
    for m, n, k in [(64, 7168, 2048), (256, 3072, 1536)]:
        try:
            A = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')
            B = torch.randn((n, k), dtype=torch.bfloat16, device='cuda')
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_scale_sh = e8m0_shuffle(A_scale)
            B_fp4, B_scale = dynamic_mxfp4_quant(B)
            B_scale_sh = e8m0_shuffle(B_scale)
            B_shuf = shuffle_weight(B_fp4.view(dtypes.fp4x2), layout=(16, 16))
            out = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

            aiter.gemm_a4w4_asm(
                A_fp4.view(dtypes.fp4x2), B_shuf,
                A_scale_sh.view(dtypes.fp8_e8m0),
                B_scale_sh.view(dtypes.fp8_e8m0),
                out, ASM_KERNEL_32x128,
                bpreshuffle=True, log2_k_split=0)
            print(f"ASM_WARM: ({m},{n},{k}) OK", file=sys.stderr)
        except Exception as e:
            print(f"ASM_WARM: ({m},{n},{k}) ERR: {e}", file=sys.stderr)

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

    # For shapes with tuned ASM configs, try ASM path
    # (64,7168,2048) and (256,3072,1536) have 6-7μs tuned kernels
    if (m == 64 and k == 2048) or (m == 256 and k == 1536):
        try:
            import aiter
            from aiter import dtypes
            from aiter.ops.triton.quant import dynamic_mxfp4_quant
            from aiter.utility.fp4_utils import e8m0_shuffle

            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            A_scale_sh = e8m0_shuffle(A_scale)

            key = (m, n)
            if key not in _y_cache:
                _y_cache[key] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

            aiter.gemm_a4w4_asm(
                A_fp4.view(dtypes.fp4x2), B_shuffle,
                A_scale_sh.view(dtypes.fp8_e8m0), B_scale_sh,
                _y_cache[key], ASM_KERNEL_32x128,
                bpreshuffle=True, log2_k_split=0)
            return _y_cache[key]
        except Exception as e:
            print(f"ASM_FALLBACK: {e}", file=sys.stderr)
            # Fall through to Triton path

    # Default: Triton path (fastest for most shapes)
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

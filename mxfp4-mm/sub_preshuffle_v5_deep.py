import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
import torch
import inspect
from task import input_t, output_t

_y_cache = {}
_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _probe(A, B_q, B_shuffle, B_scale_sh, m, n, k):
    global _probed
    if _probed:
        return
    _probed = True

    # Get the FULL kernel source (the Triton JIT function)
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle_
        src = inspect.getsource(gemm_a16wfp4_preshuffle_)
        print(f"[PS5] gemm_a16wfp4_preshuffle_ source ({len(src)} chars):")
        # Print first 3000 chars to see stride handling
        print(src[:3000])
    except Exception as e:
        print(f"[PS5] preshuffle_ source error: {e}")

    # Also get _get_config source
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import _get_config
        src2 = inspect.getsource(_get_config)
        print(f"\n[PS5] _get_config source ({len(src2)} chars):")
        print(src2[:1500])
    except Exception as e:
        print(f"[PS5] _get_config error: {e}")

    # Try preshuffle with B_shuffle in ORIGINAL shape (no reshape)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
    b_sh_u8 = B_shuffle.view(torch.uint8)
    b_sc_u8 = B_scale_sh.view(torch.uint8)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)

    # Attempt: original shape, no reshape
    try:
        result = gemm_a16wfp4_preshuffle(A, b_sh_u8, b_sc_u8, prequant=True, dtype=torch.bfloat16, y=out)
        print(f"[PS5] Original shape SUCCESS! out sample: {out[0,:5]}")
    except Exception as e:
        print(f"[PS5] Original shape: {str(e)[:300]}")

    # Print shapes for debugging
    print(f"\n[PS5] Shapes:")
    print(f"  A: {A.shape} {A.dtype}")
    print(f"  B_shuffle u8: {b_sh_u8.shape} strides={b_sh_u8.stride()}")
    print(f"  B_scale_sh u8: {b_sc_u8.shape} strides={b_sc_u8.stride()}")

    # The kernel does N, K = w.shape; N *= 16; K //= 16
    # For b_sh_u8 (2880, 256): N=2880*16=46080, K=256//16=16
    # We need N=2880, K=256 (packed fp4x2 for K=512)
    # So w must have shape (2880/16, 256*16) = (180, 4096)
    # OR the kernel interprets the data differently

    # What if we need to also reshape scales?
    # B_scale_sh u8: (3072, 16). If kernel does similar transform...
    # N_s, K_s = scale.shape; N_s *= 16; K_s //= 16
    # (3072, 16) -> N_s=3072*16=49152, K_s=16//16=1
    # That doesn't make sense for scales

    # Try: pass B_shuffle.view(torch.uint8) with shape (N//16, K_packed*16)
    # AND scales with shape (N_padded//16, K_scale*16)
    try:
        w_reshaped = b_sh_u8.reshape(n // 16, -1)
        s_reshaped = b_sc_u8.reshape(b_sc_u8.shape[0] // 16, -1)
        print(f"  w_reshaped: {w_reshaped.shape}")
        print(f"  s_reshaped: {s_reshaped.shape}")
        result = gemm_a16wfp4_preshuffle(A, w_reshaped, s_reshaped, prequant=True, dtype=torch.bfloat16, y=out)
        print(f"[PS5] Both reshaped SUCCESS! out sample: {out[0,:5]}")
    except Exception as e:
        print(f"[PS5] Both reshaped: {str(e)[:300]}")

    # Try: w reshaped, scales NOT reshaped
    try:
        result = gemm_a16wfp4_preshuffle(A, w_reshaped, b_sc_u8, prequant=True, dtype=torch.bfloat16, y=out)
        print(f"[PS5] W reshaped only SUCCESS! out sample: {out[0,:5]}")
    except Exception as e:
        print(f"[PS5] W reshaped only: {str(e)[:300]}")


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    _probe(A, B_q, B_shuffle, B_scale_sh, m, n, k)

    # Standard fallback
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)

    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)
    return out

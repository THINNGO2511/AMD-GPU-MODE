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

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle

    # Get source code
    try:
        src = inspect.getsource(gemm_a16wfp4_preshuffle)
        print(f"[PS3] gemm_a16wfp4_preshuffle source ({len(src)} chars):")
        print(src[:2000])
        if len(src) > 2000:
            print(f"... ({len(src)-2000} more chars)")
    except Exception as e:
        print(f"[PS3] getsource error: {e}")

    # Get signature
    try:
        sig = inspect.signature(gemm_a16wfp4_preshuffle)
        print(f"\n[PS3] Signature: {sig}")
    except:
        pass

    # Try calling with various arg combos
    b_sh = B_shuffle.view(torch.uint8)
    b_sc = B_scale_sh.view(torch.uint8)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)

    # Attempt 1: (A, B_shuffle, B_scale_sh, dtype, y)
    try:
        gemm_a16wfp4_preshuffle(A, b_sh, b_sc, dtype=torch.bfloat16, y=out)
        print(f"[PS3] Attempt 1 SUCCESS: (A, B_sh_u8, B_sc_u8, dtype, y)")
        print(f"[PS3] Output sample: {out[0,:5]}")
    except Exception as e:
        print(f"[PS3] Attempt 1 failed: {str(e)[:300]}")

    # Attempt 2: B_scale_sh as fp8_e8m0 (not uint8)
    try:
        gemm_a16wfp4_preshuffle(A, b_sh, B_scale_sh, dtype=torch.bfloat16, y=out)
        print(f"[PS3] Attempt 2 SUCCESS: (A, B_sh_u8, B_sc_raw, dtype, y)")
    except Exception as e:
        print(f"[PS3] Attempt 2 failed: {str(e)[:300]}")

    # Attempt 3: B_shuffle without view
    try:
        gemm_a16wfp4_preshuffle(A, B_shuffle, B_scale_sh, dtype=torch.bfloat16, y=out)
        print(f"[PS3] Attempt 3 SUCCESS: (A, B_shuffle_raw, B_sc_raw, dtype, y)")
    except Exception as e:
        print(f"[PS3] Attempt 3 failed: {str(e)[:300]}")

    # Attempt 4: with config
    try:
        cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 1,
               "waves_per_eu": 1, "matrix_instr_nonkdim": 16}
        gemm_a16wfp4_preshuffle(A, b_sh, b_sc, dtype=torch.bfloat16, y=out, config=cfg)
        print(f"[PS3] Attempt 4 SUCCESS: with config")
    except Exception as e:
        print(f"[PS3] Attempt 4 failed: {str(e)[:300]}")

    # Check B_scale_sh shape and strides
    print(f"\n[PS3] B_scale_sh shape={B_scale_sh.shape} stride={B_scale_sh.stride()} dtype={B_scale_sh.dtype}")
    print(f"[PS3] B_shuffle shape={B_shuffle.shape} stride={B_shuffle.stride()} dtype={B_shuffle.dtype}")
    print(f"[PS3] B_q shape={B_q.shape} stride={B_q.stride()} dtype={B_q.dtype}")


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    _probe(A, B_q, B_shuffle, B_scale_sh, m, n, k)

    # Fallback to standard
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

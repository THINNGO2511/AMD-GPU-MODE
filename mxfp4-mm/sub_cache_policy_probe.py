#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM — Probe if tl.load supports eviction_policy on AMD.
NVIDIA GEMV winners ALL use per-operand cache policies:
- Weight (large, streamed): no_allocate / evict_first
- Activation (small, reused): evict_last
This is THE key optimization they all share.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

from task import input_t, output_t
import torch
import triton
import triton.language as tl

# Probe: does tl.load accept eviction_policy on AMD?
_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    import inspect
    sig = inspect.signature(tl.load)
    print(f"tl.load signature: {sig}")
    params = list(sig.parameters.keys())
    print(f"tl.load params: {params}")

    has_eviction = 'eviction_policy' in params
    print(f"Has eviction_policy: {has_eviction}")

    if has_eviction:
        # Test if it actually works on AMD
        try:
            @triton.jit
            def _test_kernel(ptr, out_ptr, N, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = offs < N
                # Try evict_first (streaming, don't cache)
                x = tl.load(ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
                tl.store(out_ptr + offs, x, mask=mask)

            a = torch.randn(1024, device="cuda", dtype=torch.float32)
            b = torch.empty_like(a)
            _test_kernel[(4,)](a, b, 1024, BLOCK=256)
            print(f"eviction_policy='evict_first': WORKS on AMD!")

            @triton.jit
            def _test_kernel2(ptr, out_ptr, N, BLOCK: tl.constexpr):
                offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
                mask = offs < N
                x = tl.load(ptr + offs, mask=mask, other=0.0, eviction_policy="evict_last")
                tl.store(out_ptr + offs, x, mask=mask)

            _test_kernel2[(4,)](a, b, 1024, BLOCK=256)
            print(f"eviction_policy='evict_last': WORKS on AMD!")

        except Exception as e:
            print(f"eviction_policy test failed: {e}")

    # Also check cache_modifier in triton config
    print(f"\nTriton version: {triton.__version__}")


_bscale_raw = None
_bscale_ref = None
_bq_u8 = None
_y_cache = {}

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    global _bscale_raw, _bscale_ref, _bq_u8
    _probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    _K7168_CONFIG = {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
    }

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    cfg = _K7168_CONFIG if k == 7168 else None

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq_u8, asc, _bscale_raw, dtype=torch.bfloat16)

    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM: Call CK ASM gemm_a4w4 directly with pre-quantized A.
The CK ASM kernels produce CORRECT results but the 3-launch overhead
(quant 12μs + shuffle + GEMM) kills performance.
Idea: pre-quantize A ONCE during warmup, cache the quantized tensor,
and call only the ASM GEMM kernel on subsequent calls.

BUT: A changes every call. So we must quantize A each time.
The key insight: gemm_a4w4 includes A quantization INSIDE.
Can we call it with lower overhead?

Actually, gemm_a16wfp4 does A quant on-the-fly inside the Triton kernel.
Let's time the components individually to understand the overhead.
"""
import torch
import time
from task import input_t, output_t

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_warmed = False
_timed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}

_K2048_CONFIG = {
    "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
    "waves_per_eu": 4, "matrix_instr_nonkdim": 16,
    "cache_modifier": ".cg", "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 4096,
}

def _time_components(A, bq_u8, bscale_raw, m, n, k):
    """Time individual components to understand overhead."""
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.utility.fp4_utils import e8m0_shuffle

    # Warmup
    for _ in range(3):
        if k == 1536:
            A_fp4, A_scale = dynamic_mxfp4_quant(A)
            gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)
        else:
            cfg = _K7168_CONFIG if k == 7168 else (_K2048_CONFIG if k == 2048 else None)
            out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
            gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)

    torch.cuda.synchronize()

    # Time a16wfp4 (fused quant+gemm)
    N_REPS = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    if k != 1536:
        cfg = _K7168_CONFIG if k == 7168 else (_K2048_CONFIG if k == 2048 else None)
        out = torch.empty(m, n, dtype=torch.bfloat16, device='cuda')
        start.record()
        for _ in range(N_REPS):
            gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
        end.record()
        torch.cuda.synchronize()
        t_a16wfp4 = start.elapsed_time(end) * 1000 / N_REPS  # μs
        print(f"  a16wfp4: {t_a16wfp4:.1f} μs", flush=True)

    # Time quant alone
    start.record()
    for _ in range(N_REPS):
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
    end.record()
    torch.cuda.synchronize()
    t_quant = start.elapsed_time(end) * 1000 / N_REPS
    print(f"  quant alone: {t_quant:.1f} μs", flush=True)

    # Time afp4wfp4 alone (with pre-quantized A)
    A_fp4, A_scale = dynamic_mxfp4_quant(A)
    start.record()
    for _ in range(N_REPS):
        gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)
    end.record()
    torch.cuda.synchronize()
    t_afp4 = start.elapsed_time(end) * 1000 / N_REPS
    print(f"  afp4wfp4 (pre-quantized): {t_afp4:.1f} μs", flush=True)
    print(f"  quant+afp4wfp4: {t_quant + t_afp4:.1f} μs", flush=True)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8, _warmed, _timed

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    # Prewarm
    if not _warmed:
        _warmed = True
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        for wm, wn, wk in [(4,2880,512),(16,2112,7168),(32,4096,512),(32,2880,512),(64,7168,2048),(256,3072,1536)]:
            try:
                da = torch.randn(wm, wk, dtype=torch.bfloat16, device='cuda')
                if wk == 1536:
                    af, asc = dynamic_mxfp4_quant(da)
                    db = torch.zeros(wn, wk//2, dtype=torch.uint8, device='cuda')
                    ds = torch.full((wn, wk//32), 127, dtype=torch.uint8, device='cuda')
                    gemm_afp4wfp4(af.view(torch.uint8), db, asc, ds, dtype=torch.bfloat16)
                else:
                    db = torch.zeros(wn, wk//2, dtype=torch.uint8, device='cuda')
                    pn = ((wn+31)//32)*32
                    ds = torch.full((pn, wk//32), 127, dtype=torch.uint8, device='cuda')
                    dout = torch.empty(wm, wn, dtype=torch.bfloat16, device='cuda')
                    cfg = _K7168_CONFIG if wk==7168 else (_K2048_CONFIG if wk==2048 else None)
                    gemm_a16wfp4(da, db, ds, dtype=torch.bfloat16, y=dout, config=cfg)
            except:
                pass

    # Time components on first real call
    if not _timed:
        _timed = True
        print(f"\n=== Timing M={m}, N={n}, K={k} ===", flush=True)
        _time_components(A, _bq_u8, _bscale_raw, m, n, k)

    # Use best path
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]
    cfg = _K7168_CONFIG if k == 7168 else (_K2048_CONFIG if k == 2048 else None)
    gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

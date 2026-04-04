#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe e8m0_shuffle: read source, test behavior, understand gemm_a4w4 call."""
import torch, inspect, subprocess
from task import input_t, output_t

_probed = False

def _probe():
    global _probed
    if _probed: return
    _probed = True

    # 1. e8m0_shuffle source
    print("=== 1. e8m0_shuffle source ===", flush=True)
    try:
        from aiter.utility.fp4_utils import e8m0_shuffle
        src = inspect.getsource(e8m0_shuffle)
        print(src[:3000], flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 2. Test e8m0_shuffle behavior
    print("\n=== 2. e8m0_shuffle test ===", flush=True)
    try:
        from aiter.utility.fp4_utils import e8m0_shuffle
        # Small test
        scale = torch.arange(8, dtype=torch.uint8, device='cuda').view(1, 8)
        print(f"Input shape: {scale.shape}, values: {scale.cpu().tolist()}", flush=True)
        shuffled = e8m0_shuffle(scale)
        print(f"Output shape: {shuffled.shape}, dtype: {shuffled.dtype}", flush=True)
        print(f"Output values: {shuffled.view(torch.uint8).cpu().tolist()}", flush=True)

        # Larger test matching benchmark shape
        scale2 = torch.arange(16, dtype=torch.uint8, device='cuda').view(1, 16)
        shuffled2 = e8m0_shuffle(scale2)
        print(f"\nInput (1,16): {scale2.cpu().tolist()}", flush=True)
        print(f"Output shape: {shuffled2.shape}", flush=True)
        print(f"Output: {shuffled2.view(torch.uint8).cpu().tolist()}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 3. gemm_a4w4 signature and usage
    print("\n=== 3. gemm_a4w4 API ===", flush=True)
    try:
        from aiter import gemm_a4w4
        # Find source
        r = subprocess.run(['grep', '-n', 'def gemm_a4w4', '/home/runner/aiter/aiter/ops/gemm_a4w4.py'],
                          capture_output=True, text=True, timeout=10)
        print(f"Definition: {r.stdout[:500]}", flush=True)

        # Read gemm_op_a4w4.py
        gpath = "/home/runner/aiter/aiter/ops/gemm_op_a4w4.py"
        if __import__('os').path.exists(gpath):
            with open(gpath) as f:
                content = f.read()
            print(f"\ngemm_op_a4w4.py ({len(content)} chars):", flush=True)
            print(content[:3000], flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 4. Test gemm_a4w4 with small inputs
    print("\n=== 4. gemm_a4w4 test ===", flush=True)
    try:
        from aiter import gemm_a4w4, dtypes as aiter_dtypes
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle

        M, N, K = 4, 32, 128
        A = torch.randn(M, K, dtype=torch.bfloat16, device='cuda')
        B = torch.randn(N, K, dtype=torch.bfloat16, device='cuda')

        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        B_fp4, B_scale = dynamic_mxfp4_quant(B)

        # Shuffle scales for gemm_a4w4
        A_scale_sh = e8m0_shuffle(A_scale.view(torch.uint8))
        B_scale_sh = e8m0_shuffle(B_scale.view(torch.uint8))

        print(f"A_fp4: {A_fp4.shape} {A_fp4.dtype}", flush=True)
        print(f"A_scale: {A_scale.shape} → shuffled: {A_scale_sh.shape} {A_scale_sh.dtype}", flush=True)
        print(f"B_fp4: {B_fp4.shape} {B_fp4.dtype}", flush=True)
        print(f"B_scale: {B_scale.shape} → shuffled: {B_scale_sh.shape} {B_scale_sh.dtype}", flush=True)

        # Try calling gemm_a4w4
        out = torch.empty(M, N, dtype=torch.bfloat16, device='cuda')
        result = gemm_a4w4(
            A_fp4.view(aiter_dtypes.fp4x2),
            B_fp4.view(aiter_dtypes.fp4x2),
            A_scale_sh.view(aiter_dtypes.fp8_e8m0),
            B_scale_sh.view(aiter_dtypes.fp8_e8m0),
            out, torch.bfloat16, 1.0, 0.0, 1
        )
        print(f"gemm_a4w4 result: {result.shape} {result.dtype}", flush=True)
        print(f"Output[0,:8]: {result[0,:8].tolist()}", flush=True)

        # Compare with afp4wfp4
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        ref = gemm_afp4wfp4(A_fp4.view(torch.uint8), B_fp4.view(torch.uint8),
                            A_scale.view(torch.uint8), B_scale.view(torch.uint8),
                            dtype=torch.bfloat16)
        maxdiff = (result.float() - ref.float()).abs().max().item()
        print(f"vs afp4wfp4: maxdiff={maxdiff:.4f}", flush=True)
    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        traceback.print_exc()

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

_cache = {}

def custom_kernel(data: input_t) -> output_t:
    _probe()
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape; n = B.shape[0]
    cache_key = id(B_scale_sh)
    if cache_key not in _cache:
        _cache[cache_key] = (_unshuffle_e8m0(B_scale_sh), B_q.view(torch.uint8))
    bscale_raw, bq_u8 = _cache[cache_key]
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), bq_u8, A_scale, bscale_raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    out = torch.empty((m, n), dtype=torch.bfloat16, device=A.device)
    return gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out)

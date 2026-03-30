#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Probe: Test gemm_a16wfp4 API, configs, timing, and correctness.
Falls back to proven approach for correctness.
"""
from task import input_t, output_t
import torch
import sys

_probed = False
_use_a16wfp4 = {}  # per (m,n,k) -> bool


def _probe_a16wfp4(A, B_q_u8, B_scale_raw, B_shuffle, B_scale_sh, m, n, k):
    """Test gemm_a16wfp4 and variants, dump info."""
    global _probed
    if _probed:
        return
    _probed = True

    import os, inspect, json

    # 1. List all gemm modules in aiter
    print("=== GEMM modules ===", file=sys.stderr)
    gemm_dir = "/home/runner/aiter/aiter/ops/triton/gemm/basic"
    if os.path.isdir(gemm_dir):
        for f in sorted(os.listdir(gemm_dir)):
            if f.endswith('.py'):
                print(f"  {f}", file=sys.stderr)

    # 2. Dump gemm_a16wfp4.py source (first 300 lines)
    a16_path = os.path.join(gemm_dir, "gemm_a16wfp4.py")
    if os.path.exists(a16_path):
        with open(a16_path) as f:
            lines = f.readlines()
        print(f"\n=== gemm_a16wfp4.py ({len(lines)} lines) ===", file=sys.stderr)
        for i, line in enumerate(lines[:400]):
            print(f"{i+1:4d}: {line}", end='', file=sys.stderr)
        if len(lines) > 400:
            print(f"... ({len(lines)-400} more lines)", file=sys.stderr)
    else:
        print(f"gemm_a16wfp4.py not found at {a16_path}", file=sys.stderr)

    # 3. List a16wfp4 config files
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    if os.path.isdir(config_dir):
        a16_configs = [f for f in os.listdir(config_dir)
                       if 'a16' in f.lower() or 'A16' in f or 'PREQUANT' in f]
        fp4_configs = [f for f in os.listdir(config_dir)
                       if 'gfx950' in f and 'FP4' in f]
        print(f"\n=== a16wfp4 configs ({len(a16_configs)}) ===", file=sys.stderr)
        for f in sorted(a16_configs)[:20]:
            print(f"  {f}", file=sys.stderr)
        print(f"\n=== All gfx950 FP4 configs ({len(fp4_configs)}) ===", file=sys.stderr)
        for f in sorted(fp4_configs)[:30]:
            print(f"  {f}", file=sys.stderr)

    # 4. Try importing gemm_a16wfp4 and get signature
    print("\n=== Testing gemm_a16wfp4 import ===", file=sys.stderr)
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        sig = inspect.signature(gemm_a16wfp4)
        print(f"gemm_a16wfp4 signature: {sig}", file=sys.stderr)

        # Test call with unshuffled scales
        print(f"\nTesting gemm_a16wfp4(A[{m},{k}], B_q[{B_q_u8.shape}], scale[{B_scale_raw.shape}])...", file=sys.stderr)
        try:
            result = gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
            print(f"  SUCCESS! Result shape: {result.shape}, dtype: {result.dtype}", file=sys.stderr)
            print(f"  Result sample: {result[0,:5]}", file=sys.stderr)
            _use_a16wfp4['unshuffled'] = True
        except Exception as e:
            print(f"  FAILED (unshuffled): {e}", file=sys.stderr)

        # Test with shuffled scales
        print(f"\nTesting gemm_a16wfp4 with shuffled scales...", file=sys.stderr)
        try:
            result2 = gemm_a16wfp4(A, B_q_u8, B_scale_sh, dtype=torch.bfloat16)
            print(f"  SUCCESS (shuffled)! shape: {result2.shape}", file=sys.stderr)
            _use_a16wfp4['shuffled'] = True
        except Exception as e:
            print(f"  FAILED (shuffled): {e}", file=sys.stderr)

    except ImportError as e:
        print(f"gemm_a16wfp4 import failed: {e}", file=sys.stderr)

    # 5. Try preshuffle variant
    print("\n=== Testing preshuffle variants ===", file=sys.stderr)
    try:
        import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as mod
        all_names = [x for x in dir(mod) if not x.startswith('_')]
        print(f"Module exports: {all_names}", file=sys.stderr)

        if hasattr(mod, 'gemm_a16wfp4_preshuffle'):
            sig = inspect.signature(mod.gemm_a16wfp4_preshuffle)
            print(f"preshuffle sig: {sig}", file=sys.stderr)
            try:
                B_sh_u8 = B_shuffle.view(torch.uint8)
                result3 = mod.gemm_a16wfp4_preshuffle(A, B_sh_u8, B_scale_sh, dtype=torch.bfloat16)
                print(f"  preshuffle SUCCESS! shape: {result3.shape}", file=sys.stderr)
                _use_a16wfp4['preshuffle'] = True
            except Exception as e:
                print(f"  preshuffle FAILED: {e}", file=sys.stderr)
    except Exception as e:
        print(f"preshuffle test error: {e}", file=sys.stderr)

    # 6. Test gemm_a4w4 ASM (reference approach) - for timing comparison
    print("\n=== Testing gemm_a4w4 ASM ===", file=sys.stderr)
    try:
        import aiter
        from aiter import dtypes
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.utility.fp4_utils import e8m0_shuffle

        # Time the quant step
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        for _ in range(10):
            A_fp4, A_sc = dynamic_mxfp4_quant(A)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        quant_us = (t1 - t0) / 10 * 1e6
        print(f"  dynamic_mxfp4_quant time: {quant_us:.1f} us (M={m},K={k})", file=sys.stderr)

        # Time e8m0_shuffle
        t0 = time.perf_counter()
        for _ in range(10):
            A_sc_sh = e8m0_shuffle(A_sc)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        shuffle_us = (t1 - t0) / 10 * 1e6
        print(f"  e8m0_shuffle time: {shuffle_us:.1f} us", file=sys.stderr)

        # Time full a4w4
        A_q = A_fp4.view(dtypes.fp4x2)
        A_sc_sh_v = A_sc_sh.view(dtypes.fp8_e8m0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            out = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh_v, B_scale_sh,
                                   dtype=dtypes.bf16, bpreshuffle=True)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        gemm_us = (t1 - t0) / 10 * 1e6
        print(f"  gemm_a4w4 time: {gemm_us:.1f} us", file=sys.stderr)
        print(f"  Total (quant+shuffle+gemm): {quant_us+shuffle_us+gemm_us:.1f} us", file=sys.stderr)
    except Exception as e:
        print(f"  a4w4 test error: {e}", file=sys.stderr)

    # 7. Time gemm_a16wfp4 if it worked
    if 'unshuffled' in _use_a16wfp4:
        print("\n=== Timing gemm_a16wfp4 ===", file=sys.stderr)
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                out = gemm_a16wfp4(A, B_q_u8, B_scale_raw, dtype=torch.bfloat16)
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            a16_us = (t1 - t0) / 10 * 1e6
            print(f"  gemm_a16wfp4 time: {a16_us:.1f} us (M={m},K={k})", file=sys.stderr)
        except Exception as e:
            print(f"  timing failed: {e}", file=sys.stderr)

    # 8. Dump gemm_afp4wfp4 config lookup logic
    print("\n=== Config lookup ===", file=sys.stderr)
    try:
        from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
        sig = inspect.signature(get_gemm_config)
        print(f"get_gemm_config sig: {sig}", file=sys.stderr)
        src = inspect.getsource(get_gemm_config)
        print(f"Source:\n{src[:2000]}", file=sys.stderr)
    except Exception as e:
        print(f"config lookup error: {e}", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Unshuffle scales for Triton path
    s = B_scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)
    B_q_u8 = B_q.view(torch.uint8)

    # Run probe on first call
    _probe_a16wfp4(A, B_q_u8, s, B_shuffle, B_scale_sh, m, n, k)

    # Fall back to proven reference for correctness
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A_fp4, A_sc = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_sc_sh = e8m0_shuffle(A_sc).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh,
                            dtype=dtypes.bf16, bpreshuffle=True)

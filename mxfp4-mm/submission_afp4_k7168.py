#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Test quant+afp4wfp4 for K=7168 (M=16, N=2112).
afp4wfp4 has 69 pre-tuned per-N-K configs that may be faster.
Quant overhead for M=16 is small (~2us).
If GEMM is fast enough, total may beat a16wfp4 (14.8us).
"""
from task import input_t, output_t
import torch
import time

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_probed = False

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _probe(A, bq, bscale, m, n, k, B_shuffle=None, B_scale_sh=None):
    global _probed
    if _probed:
        return
    _probed = True

    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    print(f"\n=== afp4wfp4 vs a16wfp4 for M={m},N={n},K={k} ===", flush=True)

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    # Time a16wfp4 default (no custom config = auto-config)
    for _ in range(5):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y)
    torch.cuda.synchronize()
    a16_default = (time.perf_counter() - t0) / 50 * 1e6
    print(f"  a16wfp4 default:   {a16_default:7.1f}us", flush=True)

    # Time a16wfp4 with BM=8 KS=8
    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}
    for _ in range(5):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        gemm_a16wfp4(A, bq, bscale, dtype=torch.bfloat16, y=y, config=cfg)
    torch.cuda.synchronize()
    a16_ks8 = (time.perf_counter() - t0) / 50 * 1e6
    print(f"  a16wfp4 KS=8:      {a16_ks8:7.1f}us", flush=True)

    # Time quant only
    for _ in range(5):
        dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        A_fp4, A_sc = dynamic_mxfp4_quant(A)
    torch.cuda.synchronize()
    quant_us = (time.perf_counter() - t0) / 50 * 1e6
    print(f"  quant only:        {quant_us:7.1f}us", flush=True)

    # Time afp4wfp4 GEMM only
    A_fp4, A_sc = dynamic_mxfp4_quant(A)
    A_u8 = A_fp4.view(torch.uint8)
    for _ in range(5):
        gemm_afp4wfp4(A_u8, bq, A_sc, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        gemm_afp4wfp4(A_u8, bq, A_sc, bscale, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    afp4_gemm = (time.perf_counter() - t0) / 50 * 1e6
    print(f"  afp4wfp4 GEMM:     {afp4_gemm:7.1f}us", flush=True)
    print(f"  afp4wfp4 total:    {quant_us + afp4_gemm:7.1f}us", flush=True)

    # Check config files
    import os, json
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    for target_n in ['2112', '7168', '3072', '2880', '4096']:
        for f in sorted(os.listdir(config_dir)):
            if 'gfx950' in f and 'FP4' in f and target_n in f:
                with open(os.path.join(config_dir, f)) as fh:
                    data = json.load(fh)
                print(f"\n  {f}:", flush=True)
                if isinstance(data, list):
                    for entry in data[:3]:
                        print(f"    {entry}", flush=True)
                elif isinstance(data, dict):
                    for key in list(data.keys())[:3]:
                        print(f"    {key}: {data[key]}", flush=True)

    # Test gemm_a8wfp4 (FP8 A, FP4 B — half A bandwidth)
    print(f"\n=== gemm_a8wfp4 test ===", flush=True)
    try:
        from aiter.ops.triton.gemm.basic.gemm_a8wfp4 import gemm_a8wfp4
        import inspect
        sig = inspect.signature(gemm_a8wfp4)
        print(f"  signature: {sig}", flush=True)
        # Try calling it
        A_fp8 = A.to(torch.float8_e4m3fnuz)
        result = gemm_a8wfp4(A_fp8, bq, bscale, dtype=torch.bfloat16)
        print(f"  SUCCESS! shape={result.shape}", flush=True)
        # Time it
        for _ in range(5):
            gemm_a8wfp4(A_fp8, bq, bscale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            gemm_a8wfp4(A_fp8, bq, bscale, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        a8_us = (time.perf_counter() - t0) / 50 * 1e6
        print(f"  gemm_a8wfp4: {a8_us:.1f}us", flush=True)
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)

    # Test ASM gemm_a4w4
    if B_shuffle is not None and B_scale_sh is not None:
        print(f"\n=== ASM gemm_a4w4 test ===", flush=True)
        try:
            import aiter
            from aiter import dtypes
            from aiter.utility.fp4_utils import e8m0_shuffle
            A_fp4_q, A_sc = dynamic_mxfp4_quant(A)
            A_q = A_fp4_q.view(dtypes.fp4x2)
            A_sc_sh = e8m0_shuffle(A_sc).view(dtypes.fp8_e8m0)
            B_scale_sh_v = B_scale_sh.view(dtypes.fp8_e8m0)
            for split in [None, 0, 1, 2, 3]:
                try:
                    for _ in range(5):
                        out = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh_v,
                                               dtype=dtypes.bf16, bpreshuffle=True,
                                               log2_k_split=split)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(30):
                        out = aiter.gemm_a4w4(A_q, B_shuffle, A_sc_sh, B_scale_sh_v,
                                               dtype=dtypes.bf16, bpreshuffle=True,
                                               log2_k_split=split)
                    torch.cuda.synchronize()
                    asm_us = (time.perf_counter() - t0) / 30 * 1e6
                    total = quant_us + asm_us + 2  # ~2us for e8m0_shuffle
                    print(f"  ASM split={split}: {asm_us:.1f}us (total~{total:.1f}us)", flush=True)
                except Exception as e:
                    print(f"  ASM split={split}: FAIL {e}", flush=True)
        except Exception as e:
            print(f"  ASM setup FAILED: {e}", flush=True)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    if k == 7168:
        _probe(A, _bq_u8, _bscale_raw, m, n, k, B_shuffle, B_scale_sh)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None

    y = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, y=y, config=cfg)

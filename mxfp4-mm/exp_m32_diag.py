#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Fast diagnostic: Test custom kernel on M=32 K=512 shapes ONLY.
One config (BM32_BN64 = winner from M=4 test). Minimal JIT.
"""
from task import input_t, output_t
import torch
import sys
import time
import triton
import triton.language as tl

_ref = None
_raw = None
_sh = None
_bq = None
_tested = set()

def P(m):
    print(f"T: {m}", file=sys.stderr)

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)

def _shuffle(s):
    s = s.clone().view(torch.uint8); sm, sn = s.shape
    return s.view(sm//32,2,16,sn//8,2,4,1).permute(0,3,5,2,4,1,6).contiguous().view(sm//32,sn*32)

@triton.jit
def _k(a, b, c, asc, bsc, M, N, K,
       sa0, sa1, sb0, sb1, sc0, sc1, sas0, sas1, sbs0, sbs1,
       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
       mfma: tl.constexpr):
    SG: tl.constexpr = 32
    pid = tl.program_id(0)
    npn = tl.cdiv(N, BN)
    pm = pid // npn
    pn = pid % npn
    nki = tl.cdiv(K, BK)
    ok = tl.arange(0, BK // 2)
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    ap = a + om[:, None] * sa0 + ok[None, :] * sa1
    bp = b + ok[:, None] * sb0 + on[None, :] * sb1
    oasm = (pm * (BM // 32) + tl.arange(0, BM // 32)) % (M // 32)
    oasn = (pn * (BN // 32) + tl.arange(0, BN // 32)) % (N // 32)
    oks = tl.arange(0, BK // SG * 32)
    asp = asc + oasm[:, None] * sas0 + oks[None, :] * sas1
    bsp = bsc + oasn[:, None] * sbs0 + oks[None, :] * sbs1
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(0, nki):
        av = tl.load(ap)
        bv = tl.load(bp)
        asv = tl.load(asp).reshape(BM//32, BK//SG//8, 4, 16, 2, 2, 1).permute(0,5,3,1,4,2,6).reshape(BM, BK//SG)
        bsv = tl.load(bsp).reshape(BN//32, BK//SG//8, 4, 16, 2, 2, 1).permute(0,5,3,1,4,2,6).reshape(BN, BK//SG)
        acc += tl.dot_scaled(av, asv, "e2m1", bv, bsv, "e2m1")
        ap += (BK // 2) * sa1
        bp += (BK // 2) * sb0
        asp += BK * sas1
        bsp += BK * sbs1
    cv = acc.to(tl.bfloat16)
    ocm = pm * BM + tl.arange(0, BM).to(tl.int64)
    ocn = pn * BN + tl.arange(0, BN).to(tl.int64)
    mk = (ocm[:, None] < M) & (ocn[None, :] < N)
    tl.store(c + ocm[:, None] * sc0 + ocn[None, :] * sc1, cv, mask=mk)

def _run(afp4, asc, bq, bsc, m, n, k, bm, bn):
    C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
    g = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    _k[g](afp4, bq, C, asc, bsc, m, n, k,
          afp4.stride(0), afp4.stride(1), bq.stride(1), bq.stride(0),
          C.stride(0), C.stride(1), asc.stride(0), asc.stride(1),
          bsc.stride(0), bsc.stride(1), BM=bm, BN=bn, BK=256, mfma=16,
          num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    return C

def _test(A, bq, braw, bsh, m, n, k):
    sk = (m, n, k)
    if sk in _tested or k != 512:
        return
    _tested.add(sk)
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    afp4, asc = dynamic_mxfp4_quant(A)
    pad = (32 - asc.shape[0] % 32) % 32
    if pad > 0:
        asc_p = torch.nn.functional.pad(asc.view(torch.uint8), (0,0,0,pad), value=127)
        afp4_p = torch.nn.functional.pad(afp4.view(torch.uint8), (0,0,0,pad), value=0)
    else:
        asc_p = asc; afp4_p = afp4.view(torch.uint8)
    asc_sh = _shuffle(asc_p)
    mp = afp4_p.shape[0]
    NI = 20
    # aiter
    for _ in range(5): gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(NI): gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    da = (time.time()-t0)/NI*1e6
    # custom BM32_BN64
    for _ in range(3): _run(afp4_p, asc_sh, bq, bsh, mp, n, k, 32, 64)
    torch.cuda.synchronize()
    Cc = _run(afp4_p, asc_sh, bq, bsh, mp, n, k, 32, 64)
    torch.cuda.synchronize()
    Cr = gemm_a16wfp4(A, bq, braw, dtype=torch.bfloat16)
    err = (Cc[:m,:n].float()-Cr.float()).abs().max().item()
    t0 = time.time()
    for _ in range(NI): _run(afp4_p, asc_sh, bq, bsh, mp, n, k, 32, 64)
    torch.cuda.synchronize()
    dc = (time.time()-t0)/NI*1e6
    P(f"M{m}N{n}K{k}: custom={dc:.1f}us aiter={da:.1f}us x{da/dc:.2f} err={err:.6f}")

_K7 = {"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,
       "num_warps":4,"num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,
       "cache_modifier":None,"NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024}

def custom_kernel(data: input_t) -> output_t:
    global _ref, _raw, _sh, _bq
    A, B, Bq, Bs, Bss = data
    m, k = A.shape; n = B.shape[0]
    if _ref is not Bss:
        _ref = Bss; _raw = _unshuffle_e8m0(Bss); _bq = Bq.view(torch.uint8)
        _sh = _shuffle(_raw)
    _test(A, _bq, _raw, _sh, m, n, k)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), _bq, asc, _raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    return gemm_a16wfp4(A, _bq, _raw, dtype=torch.bfloat16, config=_K7 if k==7168 else None)

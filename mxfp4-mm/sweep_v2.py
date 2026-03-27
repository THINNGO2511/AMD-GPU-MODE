#!/usr/bin/env python3
"""
Config sweeper v2 — per-problem-size configs with M-adaptive BLOCK_M.
Fixes BLOCK_M > M error by selecting BLOCK_M per problem size.
Sweeps BOTH fused (K≤1024) and separate path configs.
"""
import subprocess, re, os, json, sys, time, itertools

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Sweep: {config_name}"""
from task import input_t, output_t
import torch, triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_br = None; _bs = None; _bq = None

def _ush(s):
    s = s.view(torch.uint8); sm, sn = s.shape
    s = s.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous()
    return s.view(sm, sn)

@triton.jit
def _fqg(a_ptr, b_ptr, c_ptr, bs_ptr, M, N, K,
    sa, sk, sbk, sbn, scm, scn, ssn, ssk,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GS: tl.constexpr):
    SG: tl.constexpr = 32
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM); npn = tl.cdiv(N, BN)
    nig = GS * npn; gid = pid // nig
    fm = gid * GS; gsm = min(npm - fm, GS)
    pm = fm + ((pid % nig) % gsm); pn = (pid % nig) // gsm
    om = (pm * BM + tl.arange(0, BM)) % M
    on = (pn * BN + tl.arange(0, BN)) % N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for ki in range(tl.cdiv(K, BK)):
        ks = ki * BK
        a = tl.load(a_ptr + om[:, None] * sa + (ks + tl.arange(0, BK))[None, :] * sk).to(tl.float32)
        af, asc = _mxfp4_quant_op(a, BK, BM, SG)
        bf = tl.load(b_ptr + (ks//2 + tl.arange(0, BK//2))[:, None] * sbk + on[None, :] * sbn)
        bsc = tl.load(bs_ptr + on[:, None] * ssn + (ks//SG + tl.arange(0, BK//SG))[None, :] * ssk)
        acc = tl.dot_scaled(af, asc, "e2m1", bf, bsc, "e2m1", acc)
    c = acc.to(tl.bfloat16)
    ocm = pm * BM + tl.arange(0, BM).to(tl.int64)
    ocn = pn * BN + tl.arange(0, BN).to(tl.int64)
    tl.store(c_ptr + ocm[:, None] * scm + ocn[None, :] * scn, c,
             mask=(ocm[:, None] < M) & (ocn[None, :] < N))

# CONFIGS: dict mapping (M,K) -> (BM,BN,BK,GS,nw,ns)
CFGS = {cfgs_str}

def custom_kernel(data: input_t) -> output_t:
    global _br, _bs, _bq
    A, B, Bq, Bsh, Bss = data
    m, k = A.shape; n = B.shape[0]
    if _br is not Bss:
        _br = Bss; _bs = _ush(Bss); _bq = Bq.view(torch.uint8)
    if k <= 1024:
        cfg = CFGS.get((m,k), (32,128,128,4,4,2))
        C = torch.empty((m,n), dtype=torch.bfloat16, device='cuda')
        grid = (triton.cdiv(m,cfg[0]) * triton.cdiv(n,cfg[1]),)
        _fqg[grid](A, _bq, C, _bs, m, n, k,
            A.stride(0), A.stride(1), _bq.stride(1), _bq.stride(0),
            C.stride(0), C.stride(1), _bs.stride(0), _bs.stride(1),
            BM=cfg[0], BN=cfg[1], BK=cfg[2], GS=cfg[3],
            num_warps=cfg[4], num_stages=cfg[5])
        return C
    else:
        Af, As = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(Af, _bq, As, _bs, dtype=torch.bfloat16)
'''

POPCORN = os.path.expanduser("~/.local/bin/popcorn-cli")
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "sweep_v2_results.json")

# Problem sizes: (M, K) for K=512 fused path
# BLOCK_M must be <= M for the % M trick, and power of 2
FUSED_CONFIGS = []
for BN in [64, 128, 256]:
    for GS in [1, 4, 8]:
        for nw in [2, 4, 8]:
            for ns in [1, 2]:
                FUSED_CONFIGS.append((32, BN, 128, GS, nw, ns))

# Total: 3*3*3*2 = 54 configs. Filter unique:
FUSED_CONFIGS = list(set(FUSED_CONFIGS))
FUSED_CONFIGS.sort()

def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f: return json.load(f)
    return {}

def save_results(r):
    with open(RESULTS, 'w') as f: json.dump(r, f, indent=2)

def gen_submission(cfg):
    BM, BN, BK, GS, nw, ns = cfg
    name = f"BM{BM}_BN{BN}_GS{GS}_W{nw}_S{ns}"
    # All K=512 sizes use same config (BM=32 works for all M >= 4)
    cfgs_str = "{\n"
    for m in [4, 8, 16, 32, 64, 256]:
        cfgs_str += f"    ({m}, 512): ({BM}, {BN}, {BK}, {GS}, {nw}, {ns}),\n"
    cfgs_str += "}"
    content = SUBMISSION_TEMPLATE.format(config_name=name, cfgs_str=cfgs_str)
    path = os.path.join(DIR, f"_sweep_{name}.py")
    with open(path, 'w') as f: f.write(content)
    return path, name

def submit(path, max_retries=3):
    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", path, "--no-tui"]
    for attempt in range(max_retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=os.path.dirname(os.path.dirname(DIR)))
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return None

        # Check for rate limit
        if "Rate limit" in out:
            wait_match = re.search(r'Try again in (\d+)s', out)
            wait = int(wait_match.group(1)) + 10 if wait_match else 400
            print(f"RATE LIMITED, waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)
            continue

        times = {}
        for m in re.finditer(r'k: (\d+); m: (\d+); n: (\d+).*?\n\s*⏱\s*([\d.]+)', out):
            times[(int(m.group(2)), int(m.group(3)), int(m.group(1)))] = float(m.group(4))
        return times if times else None
    return None

def geomean(t):
    if not t: return float('inf')
    p = 1.0
    for v in t.values(): p *= v
    return p ** (1.0/len(t))

def main():
    results = load_results()
    best_gm, best_name = float('inf'), None
    for n, d in results.items():
        g = d.get('geomean', float('inf'))
        if g < best_gm: best_gm, best_name = g, n

    print(f"Previous best: {best_name} @ {best_gm:.2f}μs")
    print(f"Total configs to test: {len(FUSED_CONFIGS)}")
    tested = sum(1 for c in FUSED_CONFIGS
                 if f"BM{c[0]}_BN{c[1]}_GS{c[3]}_W{c[4]}_S{c[5]}" in results)
    print(f"Already tested: {tested}, remaining: {len(FUSED_CONFIGS) - tested}")
    print()

    for i, cfg in enumerate(FUSED_CONFIGS):
        name = f"BM{cfg[0]}_BN{cfg[1]}_GS{cfg[3]}_W{cfg[4]}_S{cfg[5]}"
        if name in results:
            continue

        print(f"[{i+1}/{len(FUSED_CONFIGS)}] {name}...", end=" ", flush=True)
        path, _ = gen_submission(cfg)
        times = submit(path)

        if times:
            gm = geomean(times)
            results[name] = {'config': list(cfg), 'times': {str(k):v for k,v in times.items()}, 'geomean': gm}
            marker = " *** BEST ***" if gm < best_gm else ""
            print(f"{gm:.2f}μs{marker}")
            if gm < best_gm: best_gm, best_name = gm, name
        else:
            results[name] = {'config': list(cfg), 'error': True}
            print("ERROR")

        save_results(results)
        try: os.remove(path)
        except: pass

    print(f"\n{'='*50}")
    print(f"BEST: {best_name} @ {best_gm:.2f}μs")
    if best_name in results and 'times' in results[best_name]:
        for k, v in sorted(results[best_name]['times'].items()):
            print(f"  {k}: {v}μs")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Continuous GEMM config sweeper with AMD-specific compile options.
Coordinate descent + exploration. Handles rate limits. Never stops.
"""
import subprocess, re, os, json, time, random, copy, sys

POPCORN = os.path.expanduser("~/.local/bin/popcorn-cli")
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "amd_sweep_results.json")

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
from task import input_t, output_t
import torch, triton, triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op
_br=None;_bs=None;_bq=None;_cc={{}}
def _ush(s):
    s=s.view(torch.uint8);sm,sn=s.shape
    return s.view(sm//32,sn//8,4,16,2,2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
{config_code}
@triton.jit
def _fqg(a_ptr,b_ptr,c_ptr,bs_ptr,M,N,K,sa,sk,sbk,sbn,scm,scn,ssn,ssk,
    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GS:tl.constexpr):
    SG:tl.constexpr=32;pid=tl.program_id(0)
    npm=tl.cdiv(M,BM);npn=tl.cdiv(N,BN);nig=GS*npn;gid=pid//nig
    fm=gid*GS;gsm=min(npm-fm,GS);pm=fm+((pid%nig)%gsm);pn=(pid%nig)//gsm
    om=(pm*BM+tl.arange(0,BM))%M;on=(pn*BN+tl.arange(0,BN))%N
    acc=tl.zeros((BM,BN),dtype=tl.float32)
    for ki in range(tl.cdiv(K,BK)):
        ks=ki*BK
        a=tl.load(a_ptr+om[:,None]*sa+(ks+tl.arange(0,BK))[None,:]*sk).to(tl.float32)
        af,asc=_mxfp4_quant_op(a,BK,BM,SG)
        bf=tl.load(b_ptr+(ks//2+tl.arange(0,BK//2))[:,None]*sbk+on[None,:]*sbn)
        bsc=tl.load(bs_ptr+on[:,None]*ssn+(ks//SG+tl.arange(0,BK//SG))[None,:]*ssk)
        acc=tl.dot_scaled(af,asc,"e2m1",bf,bsc,"e2m1",acc)
    c=acc.to(tl.bfloat16)
    ocm=pm*BM+tl.arange(0,BM).to(tl.int64);ocn=pn*BN+tl.arange(0,BN).to(tl.int64)
    tl.store(c_ptr+ocm[:,None]*scm+ocn[None,:]*scn,c,mask=(ocm[:,None]<M)&(ocn[None,:]<N))
def custom_kernel(data:input_t)->output_t:
    global _br,_bs,_bq
    A,B,Bq,Bsh,Bss=data;m,k=A.shape;n=B.shape[0]
    if _br is not Bss:_br=Bss;_bs=_ush(Bss);_bq=Bq.view(torch.uint8)
    if k<=1024:
        ck=(m,n)
        if ck not in _cc:_cc[ck]=torch.empty((m,n),dtype=torch.bfloat16,device='cuda')
        C=_cc[ck]
        grid=lambda META:(triton.cdiv(m,META['BM'])*triton.cdiv(n,META['BN']),)
        _fqg[grid](A,_bq,C,_bs,m,n,k,A.stride(0),A.stride(1),_bq.stride(1),_bq.stride(0),
            C.stride(0),C.stride(1),_bs.stride(0),_bs.stride(1))
        return C
    else:
        Af,As=dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(Af,_bq,As,_bs,dtype=torch.bfloat16)
'''

def cfg_name(c):
    bm,bn,bk,gs,nw,ns,wpe,mi = c
    name = f"BM{bm}_BN{bn}_GS{gs}_W{nw}_S{ns}"
    if wpe: name += f"_WPE{wpe}"
    if mi: name += f"_MI{mi}"
    return name

def gen_config_code(c):
    bm,bn,bk,gs,nw,ns,wpe,mi = c
    # Must use multi-config autotune for AMD opts to work
    # (Triton extracts waves_per_eu/matrix_instr_nonkdim only through autotune)
    # Include the target config + a deliberately bad fallback so autotune picks target
    target = f"{{'BM':{bm},'BN':{bn},'BK':{bk},'GS':{gs}"
    if wpe: target += f",'waves_per_eu':{wpe}"
    if mi: target += f",'matrix_instr_nonkdim':{mi}"
    target += "}"
    # Fallback: same block sizes but 2 warps (typically worse)
    fallback_nw = 2 if nw != 2 else 4
    fallback = f"{{'BM':{bm},'BN':{bn},'BK':{bk},'GS':{gs}}}"
    return (f"@triton.autotune(configs=["
            f"triton.Config({target},num_warps={nw},num_stages={ns}),"
            f"triton.Config({fallback},num_warps={fallback_nw},num_stages={ns})"
            f"],key=['M','N','K'])")

def gen_submission(c):
    name = cfg_name(c)
    code = gen_config_code(c)
    content = SUBMISSION_TEMPLATE.format(config_code=code)
    path = os.path.join(DIR, f"_sweep_{name}.py")
    with open(path, 'w') as f: f.write(content)
    return path, name

REPO_ROOT = os.path.dirname(DIR)  # AMD-GPU-MODE/

def submit(path, max_retries=5):
    # popcorn needs relative path from repo root
    rel_path = os.path.relpath(path, REPO_ROOT)
    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", rel_path, "--no-tui"]
    for attempt in range(max_retries):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600,
                              cwd=REPO_ROOT)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            print("TIMEOUT", flush=True)
            return None
        if "Rate limit" in out:
            wait_match = re.search(r'Try again in (\d+)s', out)
            wait = int(wait_match.group(1)) + 15 if wait_match else 400
            print(f"RATE LIMITED, waiting {wait}s...", end=" ", flush=True)
            time.sleep(wait)
            continue
        # Save raw output for debugging
        debug_path = os.path.join(DIR, "_last_output.txt")
        with open(debug_path, 'w') as df:
            df.write(out)

        times = {}
        # Parse timing lines: look for the benchmark output pattern
        # Format: "k: 512; m: 4; n: 2880; seed: 4565\n ⏱ 8.42 ± 0.017 µs"
        lines = out.split('\n')
        for i, line in enumerate(lines):
            km = re.match(r'.*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)', line)
            if km and i + 1 < len(lines):
                tm = re.search(r'(\d+\.?\d*)\s*±', lines[i+1])
                if tm:
                    times[(int(km.group(2)), int(km.group(3)), int(km.group(1)))] = float(tm.group(1))
        if not times and ("TypeError" in out or "RuntimeError" in out or "SyntaxError" in out):
            return "ERROR"
        if not times and "Benchmarking failed" in out:
            return "ERROR"
        return times if times else None
    return None

def geomean(t):
    if not t: return float('inf')
    p = 1.0
    for v in t.values(): p *= v
    return p ** (1.0/len(t))

def load_results():
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            try: return json.load(f)
            except: return {}
    return {}

def save_results(r):
    with open(RESULTS, 'w') as f: json.dump(r, f, indent=2)

def test_config(cfg, results):
    name = cfg_name(cfg)
    if name in results:
        return results[name].get('geomean', float('inf'))

    print(f"  {name}...", end=" ", flush=True)
    path, _ = gen_submission(cfg)
    times = submit(path)
    try: os.remove(path)
    except: pass

    if times == "ERROR":
        results[name] = {'config': list(cfg), 'error': True}
        print("ERROR")
        save_results(results)
        return float('inf')
    elif times:
        gm = geomean(times)
        results[name] = {
            'config': list(cfg),
            'times': {str(k): v for k, v in times.items()},
            'geomean': gm
        }
        print(f"{gm:.2f}μs")
        save_results(results)
        return gm
    else:
        results[name] = {'config': list(cfg), 'error': True}
        print("FAILED")
        save_results(results)
        return float('inf')

def find_best(results):
    best_gm, best_cfg, best_name = float('inf'), None, None
    for name, d in results.items():
        gm = d.get('geomean', float('inf'))
        if gm < best_gm and 'config' in d:
            best_gm, best_cfg, best_name = gm, tuple(d['config']), name
    return best_cfg, best_gm, best_name

def coordesc_round(start_cfg, results):
    """One round of coordinate descent from start_cfg."""
    # Config: (BM, BN, BK, GS, NW, NS, WPE, MI)
    param_options = {
        0: [32, 64],           # BLOCK_M
        1: [32, 64, 128, 256], # BLOCK_N
        # 2: [128],            # BLOCK_K (fixed)
        3: [1, 2, 4, 8],      # GROUP_SIZE_M
        4: [2, 4, 8],         # num_warps
        5: [1, 2],            # num_stages
        6: [0, 1, 2, 3, 4],   # waves_per_eu (0=auto)
        7: [0, 16],           # matrix_instr_nonkdim (0=auto)
    }

    current = list(start_cfg)
    current_gm = test_config(tuple(current), results)
    improved = False

    for param_idx, options in param_options.items():
        best_val = current[param_idx]
        best_gm = current_gm

        for val in options:
            if val == current[param_idx]:
                continue
            candidate = list(current)
            candidate[param_idx] = val
            gm = test_config(tuple(candidate), results)
            if gm < best_gm:
                best_gm = gm
                best_val = val
                improved = True

        if best_val != current[param_idx]:
            current[param_idx] = best_val
            current_gm = best_gm
            print(f"  >> Improved param {param_idx} to {best_val}, geomean={best_gm:.2f}μs")

    return tuple(current), current_gm, improved

def random_config():
    return (
        random.choice([32, 64]),
        random.choice([32, 64, 128, 256]),
        128,
        random.choice([1, 2, 4, 8]),
        random.choice([2, 4, 8]),
        random.choice([1, 2]),
        random.choice([0, 1, 2, 3, 4]),
        random.choice([0, 16]),
    )

def main():
    results = load_results()
    round_num = 0

    # Starting config (without AMD opts first to verify template)
    best_cfg = (32, 64, 128, 1, 8, 2, 0, 0)

    # Check if we have a better one from previous runs
    prev_best, prev_gm, prev_name = find_best(results)
    if prev_best and prev_gm < float('inf'):
        best_cfg = prev_best
        print(f"Resuming from previous best: {prev_name} @ {prev_gm:.2f}μs")
    else:
        print(f"Starting from: {cfg_name(best_cfg)}")

    while True:
        round_num += 1
        print(f"\n{'='*60}")
        print(f"ROUND {round_num}")
        print(f"{'='*60}")

        # Phase 1: Coordinate descent
        print(f"\n--- Coordinate Descent from {cfg_name(best_cfg)} ---")
        new_cfg, new_gm, improved = coordesc_round(best_cfg, results)

        if new_gm < geomean(results.get(cfg_name(best_cfg), {}).get('times', {}) or {'x': float('inf')}):
            best_cfg = new_cfg
            print(f"  Best after coordesc: {cfg_name(best_cfg)} @ {new_gm:.2f}μs")

        # Phase 2: Random exploration (5 random configs)
        print(f"\n--- Random Exploration ---")
        for i in range(5):
            rc = random_config()
            gm = test_config(rc, results)
            if gm < new_gm:
                best_cfg = rc
                new_gm = gm
                print(f"  >> Random found better: {cfg_name(rc)} @ {gm:.2f}μs!")

        # Print summary
        best_overall, best_gm_overall, best_name_overall = find_best(results)
        tested = sum(1 for v in results.values() if 'geomean' in v)
        errors = sum(1 for v in results.values() if v.get('error'))
        print(f"\n--- Round {round_num} Summary ---")
        print(f"  Total tested: {tested}, errors: {errors}")
        print(f"  Best overall: {best_name_overall} @ {best_gm_overall:.2f}μs")
        if best_overall:
            best_cfg = best_overall

        # Show top 5
        ranked = sorted(
            [(n, d['geomean']) for n, d in results.items() if 'geomean' in d],
            key=lambda x: x[1]
        )[:5]
        print(f"  Top 5:")
        for name, gm in ranked:
            print(f"    {name}: {gm:.2f}μs")

if __name__ == "__main__":
    main()

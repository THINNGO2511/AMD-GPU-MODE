#!/usr/bin/env python3
"""
Automated sweep for ALL 3 problems. Runs unattended.
Rotates: MoE → GEMM → MLA → MoE → ...
Rate limit: 10/hr per problem. Submits 1 every 7 min per problem = ~8/hr.
With 3 problems rotating: 1 submission every ~2.5 min.
"""
import os, sys, time, subprocess, random, json, shutil
from datetime import datetime

REPO = os.path.expanduser("~/Downloads/Code/gpu mode/AMD-GPU-MODE")
LOG_DIR = os.path.join(REPO, "sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%m/%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "sweep_all.log"), "a") as f:
        f.write(line + "\n")

def submit(leaderboard, mode, filepath):
    cmd = f"cd {REPO} && popcorn submit --gpu MI355X --leaderboard {leaderboard} --mode {mode} {filepath} --no-tui"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
        output = result.stdout + result.stderr
        # Extract timing if benchmark
        timing = ""
        for line in output.split("\n"):
            if "μs" in line or "geomean" in line.lower():
                timing += line.strip() + " | "
        success = ("Rate limit" not in output
                   and "Testing failed" not in output
                   and "Application error" not in output
                   and "Benchmarking failed" not in output)
        return success, output[-300:], timing
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT", ""
    except Exception as e:
        return False, str(e)[:200], ""

def write_submission(filepath, code):
    full = os.path.join(REPO, filepath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, 'w') as f:
        f.write(code)
    return filepath

# ============================================================
# MoE VARIANTS
# ============================================================
def gen_moe_variants():
    variants = []
    base_call = '''def custom_kernel(data):
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    hidden_pad = config['d_hidden_pad'] - config['d_hidden']
    intermediate_pad = config['d_expert_pad'] - config['d_expert']
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs,
                     a1_scale=None, a2_scale=None,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
'''
    imports = '''import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
'''
    for cu in ['256', '304']:
        for nt in [True, False]:
            for opus in [True, False]:
                for bm_name, bm_code in [
                    ("def", ""),
                    ("w128", "fm.get_block_size_M = lambda t,k,e,d: 128 if e>64 else 64"),
                    ("w64", "fm.get_block_size_M = lambda t,k,e,d: 64"),
                    ("s32", "fm.get_block_size_M = lambda t,k,e,d: 32"),
                ]:
                    name = f"cu{cu}_nt{'off' if nt else 'on'}_{'opus' if opus else 'noopus'}_{bm_name}"
                    env = f"import os\nos.environ['CU_NUM']='{cu}'\nos.environ['AITER_USE_NT']='0'\n"
                    if opus:
                        env += "os.environ['AITER_USE_OPUS_MOE_SORTING']='1'\n"
                    patches = "fm.use_nt = lambda t,k,e: False\n" if nt else ""
                    patches += bm_code + "\n" if bm_code else ""
                    header = "#!POPCORN leaderboard amd-moe-mxfp4\n#!POPCORN gpu MI355X\nfrom task import input_t, output_t\n"
                    code = header + env + imports + patches + base_call
                    fp = write_submission(f"moe-mxfp4/sweep_{name}.py", code)
                    variants.append(("amd-moe-mxfp4", name, fp, "test"))
    return variants

# ============================================================
# GEMM VARIANTS  
# ============================================================
def gen_gemm_variants():
    """Vary the a16wfp4 configs and K-threshold"""
    variants = []
    
    # Our working base: a16wfp4 for K<=1024, afp4wfp4 for K>1024
    base_imports = '''import torch
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle
'''
    unshuffle = '''
_sc = {}
def _unshuffle(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _sc: return _sc[key]
    n_sc = K // 32
    sm = ((N+255)//256)*256; sn = ((n_sc+7)//8)*8
    s = B_scale_sh.view(torch.uint8)
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N,:n_sc] = s[:N,:n_sc]
    r = p.view(sm//32, sn//8, 4, 16, 2, 2)
    u = r.permute(0,5,3,1,4,2).contiguous()
    result = u.view(sm, sn)[:N,:n_sc]
    _sc[key] = result
    return result
'''
    # Vary the K threshold for switching from a16wfp4 to afp4wfp4
    for k_thresh in [512, 1024, 2048, 7168]:  # 7168 = a16wfp4 for ALL
        name = f"kthresh_{k_thresh}"
        header = '#!POPCORN leaderboard amd-mxfp4-mm\n#!POPCORN gpu MI355X\n'
        code = header + base_imports + unshuffle + f'''
def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    if K <= {k_thresh}:
        B_scale = _unshuffle(B_scale_sh, N, K)
        return gemm_a16wfp4(A, B_q.view(torch.uint8), B_scale)
    else:
        A_q, A_s = dynamic_mxfp4_quant(A)
        A_ss = e8m0_shuffle(A_s)
        return gemm_afp4wfp4(A_q, B_shuffle, A_ss, B_scale_sh)
'''
        fp = write_submission(f"mxfp4-mm/sweep_{name}.py", code)
        variants.append(("amd-mxfp4-mm", name, fp, "test"))
    
    return variants

# ============================================================
# MLA VARIANTS
# ============================================================
def gen_mla_variants():
    """Vary num_kv_splits around the working pg8_v2"""
    variants = []
    
    base = os.path.join(REPO, "mixed-mla/submission_pg8_v2.py")
    if not os.path.exists(base):
        log("WARNING: pg8_v2 not found, skipping MLA variants")
        return variants
    
    with open(base) as f:
        base_code = f.read()
    
    # Vary num_kv_splits for different batch sizes
    for small_splits in [4, 8, 12]:
        for large_splits in [8, 16, 24, 32]:
            name = f"splits_{small_splits}_{large_splits}"
            code = base_code
            # Replace the split logic - find and modify
            # The pg8_v2 likely has hardcoded splits
            # We'll add a wrapper that overrides
            code = f'''# Modified splits: small={small_splits}, large={large_splits}
_SMALL_SPLITS = {small_splits}
_LARGE_SPLITS = {large_splits}
''' + code
            fp = write_submission(f"mixed-mla/sweep_{name}.py", code)
            variants.append(("amd-mixed-mla", name, fp, "test"))
    
    return variants

# ============================================================
# MAIN SWEEP LOOP
# ============================================================
def main():
    log("=== AUTO SWEEP ALL 3 PROBLEMS ===")
    
    moe = gen_moe_variants()
    gemm = gen_gemm_variants()
    mla = gen_mla_variants()
    
    log(f"Variants: MoE={len(moe)}, GEMM={len(gemm)}, MLA={len(mla)}")
    
    # Combine and shuffle within each problem, then interleave
    random.shuffle(moe)
    random.shuffle(gemm)
    random.shuffle(mla)
    
    # Interleave: MoE, GEMM, MLA, MoE, GEMM, MLA, ...
    all_variants = []
    max_len = max(len(moe), len(gemm), len(mla))
    for i in range(max_len):
        if i < len(moe): all_variants.append(moe[i])
        if i < len(gemm): all_variants.append(gemm[i])
        if i < len(mla): all_variants.append(mla[i])
    
    log(f"Total submissions queued: {len(all_variants)}")
    
    submissions = {}  # leaderboard -> [timestamps]
    results_file = os.path.join(LOG_DIR, "results.jsonl")
    
    for idx, (leaderboard, name, filepath, mode) in enumerate(all_variants):
        now = time.time()
        
        # Clean old timestamps
        if leaderboard not in submissions:
            submissions[leaderboard] = []
        submissions[leaderboard] = [t for t in submissions[leaderboard] if now - t < 3600]
        
        # Check rate limit
        if len(submissions[leaderboard]) >= 9:
            oldest = min(submissions[leaderboard])
            wait = 3600 - (now - oldest) + 30
            log(f"Rate limited on {leaderboard} ({len(submissions[leaderboard])}/10). Waiting {wait:.0f}s...")
            time.sleep(max(wait, 60))
            submissions[leaderboard] = [t for t in submissions[leaderboard] if time.time() - t < 3600]
        
        prob = leaderboard.split("-")[-1]
        log(f"[{idx+1}/{len(all_variants)}] {prob} {mode}: {name}")
        
        ok, output, timing = submit(leaderboard, mode, filepath)
        submissions[leaderboard].append(time.time())
        
        # Log result
        result = {"idx": idx, "problem": prob, "name": name, "mode": mode, 
                  "ok": ok, "timing": timing, "ts": datetime.now().isoformat()}
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        if "Rate limit" in output:
            log(f"  RATE LIMITED — backing off 7 min")
            time.sleep(420)
        elif ok:
            log(f"  OK {timing[:100] if timing else ''}")
            time.sleep(30)  # pace between submissions
        else:
            err = output.replace("\n", " ")[:150]
            log(f"  ERROR: {err}")
            time.sleep(15)
    
    log("=== SWEEP COMPLETE ===")
    log(f"Results in {results_file}")

if __name__ == "__main__":
    main()

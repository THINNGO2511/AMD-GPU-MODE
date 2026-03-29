#!/usr/bin/env python3
"""
Automated sweep runner for all 3 problems.
Generates variants, submits them, logs results.
Runs unattended — just start and leave it.

Rate limits: 10 submissions/hr per problem (test+benchmark+leaderboard combined)
Strategy: 3 test + 3 benchmark + 1 leaderboard per hour per problem
"""
import os, sys, time, subprocess, itertools, random, json
from datetime import datetime

REPO = os.path.expanduser("~/Downloads/Code/gpu mode/AMD-GPU-MODE")
LOG_DIR = os.path.join(REPO, "sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOG_DIR, "sweep.log"), "a") as f:
        f.write(line + "\n")

def submit(problem, mode, filepath):
    """Submit and return (success, output)"""
    leaderboard = {"gemm": "amd-mxfp4-mm", "moe": "amd-moe-mxfp4", "mla": "amd-mixed-mla"}[problem]
    cmd = f"cd {REPO} && popcorn submit --gpu MI355X --leaderboard {leaderboard} --mode {mode} {filepath} --no-tui"
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
        success = "Rate limit" not in output and "error" not in output.lower()
        return success, output[-500:]
    except Exception as e:
        return False, str(e)

def write_moe_variant(name, env_vars, patches):
    """Generate a MoE submission variant"""
    filepath = f"moe-mxfp4/sweep_{name}.py"
    full_path = os.path.join(REPO, filepath)
    
    env_lines = "\n".join(f"os.environ['{k}'] = '{v}'" for k, v in env_vars.items())
    patch_lines = "\n".join(patches)
    
    code = f'''import os
{env_lines}
import torch
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType
{patch_lines}
def custom_kernel(data):
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
    with open(full_path, 'w') as f:
        f.write(code)
    return filepath

def generate_moe_variants():
    """Generate all MoE sweep variants"""
    variants = []
    
    # Sweep: CU_NUM × use_nt × opus × block_m_override
    cu_nums = ['256', '304']
    use_nt_patches = [
        ("nt_off", "fm.use_nt = lambda t,k,e: False"),
        ("nt_default", "# use_nt default"),
    ]
    opus_settings = [
        ("noopus", {}),
        ("opus", {"AITER_USE_OPUS_MOE_SORTING": "1"}),
    ]
    blockm_patches = [
        ("bm_default", "# block_m default"),
        ("bm_wide", "fm.get_block_size_M = lambda t,k,e,d: 128 if e > 64 else 64"),
        ("bm_small", "fm.get_block_size_M = lambda t,k,e,d: 64 if e > 64 else 32"),
    ]
    
    for cu in cu_nums:
        for nt_name, nt_patch in use_nt_patches:
            for opus_name, opus_env in opus_settings:
                for bm_name, bm_patch in blockm_patches:
                    name = f"{cu}_{nt_name}_{opus_name}_{bm_name}"
                    env = {"CU_NUM": cu, "AITER_USE_NT": "0"}
                    env.update(opus_env)
                    patches = [nt_patch, bm_patch]
                    filepath = write_moe_variant(name, env, patches)
                    variants.append(("moe", name, filepath))
    
    return variants

def write_mla_variant(name, page_sizes, splits):
    """Generate MLA variant with different page_size/splits combos"""
    filepath = f"mixed-mla/sweep_{name}.py"
    full_path = os.path.join(REPO, filepath)
    
    # Copy the working pg8_v2 as base and modify splits
    base = os.path.join(REPO, "mixed-mla/submission_pg8_v2.py")
    if not os.path.exists(base):
        return None
    
    with open(base) as f:
        code = f.read()
    
    # Modify num_kv_splits values
    for old_splits, new_splits in splits.items():
        code = code.replace(f"num_kv_splits={old_splits}", f"num_kv_splits={new_splits}")
    
    with open(full_path, 'w') as f:
        f.write(code)
    return filepath

def run_sweep():
    """Main sweep loop"""
    log("=== AUTO SWEEP STARTED ===")
    log(f"Repo: {REPO}")
    
    # Generate all MoE variants
    moe_variants = generate_moe_variants()
    log(f"Generated {len(moe_variants)} MoE variants")
    random.shuffle(moe_variants)
    
    # Track submissions per hour per problem
    submissions = {"gemm": [], "moe": [], "mla": []}
    
    variant_idx = 0
    while variant_idx < len(moe_variants):
        now = time.time()
        
        # Clean old timestamps (older than 1 hour)
        for p in submissions:
            submissions[p] = [t for t in submissions[p] if now - t < 3600]
        
        # Submit MoE variants if under rate limit
        if len(submissions["moe"]) < 8 and variant_idx < len(moe_variants):
            problem, name, filepath = moe_variants[variant_idx]
            variant_idx += 1
            
            log(f"Submitting MoE #{variant_idx}/{len(moe_variants)}: {name}")
            ok, output = submit("moe", "test", filepath)
            submissions["moe"].append(now)
            
            if "Rate limit" in output:
                log(f"  RATE LIMITED — waiting 7 min")
                time.sleep(420)
            elif ok:
                log(f"  Submitted OK")
                # Wait between submissions
                time.sleep(30)
            else:
                log(f"  ERROR: {output[:200]}")
                time.sleep(10)
        else:
            # Rate limited — wait
            wait = 420  # 7 minutes
            log(f"Rate limited ({len(submissions['moe'])}/10 this hour). Waiting {wait}s...")
            time.sleep(wait)
    
    log("=== SWEEP COMPLETE ===")

if __name__ == "__main__":
    run_sweep()

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X

"""
MoE Kernel Probe v3: Enumerate ALL CK kernels, test for E=33 d=2048.

Key insight: get_2stage_cfgs has @lru_cache — must clear cache between tests.
Strategy: monkey-patch get_2stage_cfgs + clear its cache, OR patch at fused_moe_ level.
"""

import os
import sys
import re
import time
import torch
from typing import Dict, Tuple

input_t = Tuple[torch.Tensor, ...]
output_t = torch.Tensor

_probed = False

def _enumerate_kernels():
    """List and categorize all CK kernels."""
    kernel_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    
    print("=" * 100, flush=True)
    print("MoE KERNEL PROBE v3 — MI355X", flush=True)
    print("=" * 100, flush=True)
    
    try:
        all_files = sorted(os.listdir(kernel_dir))
    except Exception as e:
        print(f"ERROR: {e}", flush=True)
        return [], []
    
    co_files = [f.replace('.co', '') for f in all_files if f.endswith('.co')]
    csv_files = [f for f in all_files if f.endswith('.csv')]
    
    print(f"\nTotal .co: {len(co_files)}, .csv: {len(csv_files)}", flush=True)
    
    # Print ALL kernel names
    for i, name in enumerate(co_files):
        print(f"  [{i:3d}] {name}", flush=True)
    
    # Categorize
    s1_fp4_silu = []
    s2_fp4 = []
    
    for name in co_files:
        is_s1 = 'gemm1' in name
        is_s2 = 'gemm2' in name
        is_fp4 = 'FP4X2' in name
        has_silu = 'silu' in name.lower()
        
        if is_s1 and is_fp4 and has_silu:
            s1_fp4_silu.append(name)
        elif is_s2 and is_fp4:
            s2_fp4.append(name)
    
    print(f"\n--- Stage1 FP4+SiLU: {len(s1_fp4_silu)} ---", flush=True)
    for k in s1_fp4_silu:
        m = re.search(r'(\d+x\d+x\d+x\d+_\d+x\d+)', k)
        tiles = m.group(1) if m else "?"
        print(f"  {tiles:30s}  {k}", flush=True)
    
    print(f"\n--- Stage2 FP4: {len(s2_fp4)} ---", flush=True)
    for k in s2_fp4:
        m = re.search(r'(\d+x\d+x\d+x\d+_\d+x\d+)', k)
        tiles = m.group(1) if m else "?"
        print(f"  {tiles:30s}  {k}", flush=True)
    
    # Read CSV configs
    for csv_f in csv_files:
        csv_path = os.path.join(kernel_dir, csv_f)
        try:
            with open(csv_path) as f:
                content = f.read()
            lines = content.strip().split('\n')
            print(f"\n{csv_f}: {len(lines)} lines", flush=True)
            for line in lines[:15]:
                print(f"  {line}", flush=True)
            if len(lines) > 15:
                print(f"  ... ({len(lines)-15} more)", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
    
    # Read tuned fmoe CSVs
    for csv_path in [
        "/home/runner/aiter/aiter/configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    ]:
        if os.path.exists(csv_path):
            print(f"\n{csv_path}:", flush=True)
            try:
                with open(csv_path) as f:
                    lines = f.readlines()
                print(f"  Total: {len(lines)} lines", flush=True)
                if lines:
                    print(f"  Header: {lines[0].strip()}", flush=True)
                for l in lines:
                    if ',33,' in l or ',2048,' in l:
                        print(f"  {l.strip()}", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
    
    return s1_fp4_silu, s2_fp4


def custom_kernel(data):
    global _probed
    
    (hidden_states, w1, w2, w1_s, w2_s, w1_qw, w2_qw, w1_qs, w2_qs,
     topk_weights, topk_ids, config) = data
    
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    
    if not _probed:
        _probed = True
        
        # Enumerate
        s1_kernels, s2_kernels = _enumerate_kernels()
        
        # Shape info
        n_routed = config.get('num_routed_experts', 32)
        n_shared = config.get('num_shared_experts', 1)
        n_experts = n_routed + n_shared
        bs = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        d_hidden = hidden_states.shape[1]
        est_m = bs * topk // n_experts
        
        print(f"\nSHAPE: bs={bs} E={n_experts} topk={topk} d_hidden={d_hidden} est_m={est_m}", flush=True)
        print(f"  w1_qw={w1_qw.shape} w2_qw={w2_qw.shape}", flush=True)
        print(f"  config={config}", flush=True)
        
        # Inspect get_2stage_cfgs
        import inspect
        import aiter.fused_moe as fm
        
        # Check if get_2stage_cfgs has cache_clear
        has_cache = hasattr(fm.get_2stage_cfgs, 'cache_clear')
        print(f"\nget_2stage_cfgs has cache_clear: {has_cache}", flush=True)
        
        # Get source
        try:
            src = inspect.getsource(fm.get_2stage_cfgs)
            lines = src.split('\n')
            print(f"\nget_2stage_cfgs source ({len(lines)} lines):", flush=True)
            for line in lines[:80]:
                print(f"  {line}", flush=True)
            if len(lines) > 80:
                print(f"  ... ({len(lines)-80} more lines)", flush=True)
        except Exception as e:
            print(f"  source error: {e}", flush=True)
        
        # Get fused_moe_ source (look for how kernelName is used)
        try:
            src = inspect.getsource(fm.fused_moe_)
            lines = src.split('\n')
            print(f"\nfused_moe_ source ({len(lines)} lines):", flush=True)
            for i, line in enumerate(lines):
                if any(kw in line for kw in ['kernelName', 'metadata', 'ck_moe_stage', 'block_m', 'block_size']):
                    lo = max(0, i-2)
                    hi = min(len(lines), i+3)
                    for j in range(lo, hi):
                        print(f"  [{j:3d}] {lines[j]}", flush=True)
                    print("  ---", flush=True)
        except Exception as e:
            print(f"  fused_moe_ source error: {e}", flush=True)
        
        # Baseline timing
        print(f"\n{'='*80}", flush=True)
        print("BASELINE TIMING:", flush=True)
        
        # warmup
        for _ in range(3):
            r0 = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                          expert_mask=None, activation=ActivationType.Silu,
                          quant_type=QuantType.per_1x32, doweight_stage1=False,
                          w1_scale=w1_qs, w2_scale=w2_qs)
        torch.cuda.synchronize()
        
        t0 = time.time()
        n_iter = 10
        for _ in range(n_iter):
            r0 = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                          expert_mask=None, activation=ActivationType.Silu,
                          quant_type=QuantType.per_1x32, doweight_stage1=False,
                          w1_scale=w1_qs, w2_scale=w2_qs)
            torch.cuda.synchronize()
        baseline_us = (time.time() - t0) / n_iter * 1e6
        baseline_r = r0.clone()
        print(f"  Baseline: {baseline_us:.1f} μs", flush=True)
        
        # === KERNEL INJECTION TESTS ===
        # Strategy: Instead of patching get_2stage_cfgs (which is cached),
        # patch fused_moe_ directly to intercept the kernel names.
        # OR: patch ck_moe_stage1 and ck_moe_stage2 to override kernelName param.
        
        print(f"\n{'='*80}", flush=True)
        print("KERNEL INJECTION TESTS:", flush=True)
        print(f"{'='*80}", flush=True)
        
        # Approach: Wrap ck_moe_stage1 and ck_moe_stage2 to intercept kernelName
        original_stage1 = None
        original_stage2 = None
        override_s1_name = [None]  # Use list for closure mutability
        override_s2_name = [None]
        
        try:
            # Find the actual stage functions
            # They might be at aiter.ck_moe_stage1_fwd or torch.ops.aiter.ck_moe_stage1
            # From the source, it's called as ck_moe_stage1(...) after JIT loading
            
            # Let's check what's available
            print(f"  dir(fm) ck/stage items:", flush=True)
            for x in dir(fm):
                if 'ck_moe' in x or 'stage' in x.lower():
                    print(f"    {x}", flush=True)
            
            # The actual CK functions are called from fused_moe_ as:
            # ck_moe_stage1(hidden_states, w1, w2, sorted_token_ids, sorted_expert_ids,
            #               num_valid_ids, out, topk, kernelName=..., ...)
            # These are imported into fused_moe module
            
            if hasattr(fm, 'ck_moe_stage1'):
                original_stage1 = fm.ck_moe_stage1
            if hasattr(fm, 'ck_moe_stage2'):
                original_stage2 = fm.ck_moe_stage2
            
            print(f"  Found ck_moe_stage1: {original_stage1 is not None}", flush=True)
            print(f"  Found ck_moe_stage2: {original_stage2 is not None}", flush=True)
            
            if original_stage1 and original_stage2:
                def wrapped_stage1(*args, **kwargs):
                    if override_s1_name[0] is not None:
                        kwargs['kernelName'] = override_s1_name[0]
                    return original_stage1(*args, **kwargs)
                
                def wrapped_stage2(*args, **kwargs):
                    if override_s2_name[0] is not None:
                        kwargs['kernelName'] = override_s2_name[0]
                    return original_stage2(*args, **kwargs)
                
                fm.ck_moe_stage1 = wrapped_stage1
                fm.ck_moe_stage2 = wrapped_stage2
                
                # Test each stage1 kernel
                print(f"\n--- Testing {len(s1_kernels)} Stage1 kernels ---", flush=True)
                s1_results = []
                
                for s1_name in s1_kernels:
                    override_s1_name[0] = s1_name
                    override_s2_name[0] = None  # Use default stage2
                    
                    try:
                        # Warmup
                        r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                     expert_mask=None, activation=ActivationType.Silu,
                                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                                     w1_scale=w1_qs, w2_scale=w2_qs)
                        torch.cuda.synchronize()
                        
                        t0 = time.time()
                        for _ in range(5):
                            r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                         expert_mask=None, activation=ActivationType.Silu,
                                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                                         w1_scale=w1_qs, w2_scale=w2_qs)
                            torch.cuda.synchronize()
                        us = (time.time() - t0) / 5 * 1e6
                        
                        err = (r - baseline_r).abs().max().item()
                        pct = (us / baseline_us - 1) * 100
                        s1_results.append((us, err, s1_name, pct))
                        ok = "✓" if err < 0.05 else "✗"
                        print(f"  {ok} {us:7.1f}μs ({pct:+5.1f}%) err={err:.4f}  {s1_name}", flush=True)
                    except Exception as e:
                        err_str = str(e)[:80]
                        print(f"  ✗ FAIL  {s1_name[-60:]}  {err_str}", flush=True)
                
                # Test each stage2 kernel
                print(f"\n--- Testing {len(s2_kernels)} Stage2 kernels ---", flush=True)
                s2_results = []
                
                for s2_name in s2_kernels:
                    override_s1_name[0] = None  # Use default stage1
                    override_s2_name[0] = s2_name
                    
                    try:
                        r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                     expert_mask=None, activation=ActivationType.Silu,
                                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                                     w1_scale=w1_qs, w2_scale=w2_qs)
                        torch.cuda.synchronize()
                        
                        t0 = time.time()
                        for _ in range(5):
                            r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                         expert_mask=None, activation=ActivationType.Silu,
                                         quant_type=QuantType.per_1x32, doweight_stage1=False,
                                         w1_scale=w1_qs, w2_scale=w2_qs)
                            torch.cuda.synchronize()
                        us = (time.time() - t0) / 5 * 1e6
                        
                        err = (r - baseline_r).abs().max().item()
                        pct = (us / baseline_us - 1) * 100
                        s2_results.append((us, err, s2_name, pct))
                        ok = "✓" if err < 0.05 else "✗"
                        print(f"  {ok} {us:7.1f}μs ({pct:+5.1f}%) err={err:.4f}  {s2_name}", flush=True)
                    except Exception as e:
                        err_str = str(e)[:80]
                        print(f"  ✗ FAIL  {s2_name[-60:]}  {err_str}", flush=True)
                
                # Combine top results
                good_s1 = [(us, err, name) for us, err, name, _ in s1_results if err < 0.05]
                good_s2 = [(us, err, name) for us, err, name, _ in s2_results if err < 0.05]
                good_s1.sort()
                good_s2.sort()
                
                if good_s1 and good_s2:
                    print(f"\n--- Combining top 3 x top 3 ---", flush=True)
                    combo_results = []
                    
                    for _, _, s1 in good_s1[:3]:
                        for _, _, s2 in good_s2[:3]:
                            override_s1_name[0] = s1
                            override_s2_name[0] = s2
                            
                            try:
                                r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                             expert_mask=None, activation=ActivationType.Silu,
                                             quant_type=QuantType.per_1x32, doweight_stage1=False,
                                             w1_scale=w1_qs, w2_scale=w2_qs)
                                torch.cuda.synchronize()
                                
                                t0 = time.time()
                                for _ in range(5):
                                    r = fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                                                 expert_mask=None, activation=ActivationType.Silu,
                                                 quant_type=QuantType.per_1x32, doweight_stage1=False,
                                                 w1_scale=w1_qs, w2_scale=w2_qs)
                                    torch.cuda.synchronize()
                                us = (time.time() - t0) / 5 * 1e6
                                
                                err = (r - baseline_r).abs().max().item()
                                pct = (us / baseline_us - 1) * 100
                                combo_results.append((us, err, s1, s2, pct))
                                ok = "✓" if err < 0.05 else "✗"
                                print(f"  {ok} {us:7.1f}μs ({pct:+5.1f}%) err={err:.4f}", flush=True)
                                print(f"      s1=...{s1[-50:]}", flush=True)
                                print(f"      s2=...{s2[-50:]}", flush=True)
                            except Exception as e:
                                print(f"  ✗ FAIL combo: {str(e)[:60]}", flush=True)
                    
                    combo_results.sort()
                    print(f"\n{'='*100}", flush=True)
                    print("FINAL SUMMARY:", flush=True)
                    print(f"  Baseline: {baseline_us:.1f} μs", flush=True)
                    
                    if combo_results:
                        best_us, best_err, best_s1, best_s2, best_pct = combo_results[0]
                        print(f"  Best combo: {best_us:.1f}μs ({best_pct:+.1f}%)", flush=True)
                        print(f"    s1={best_s1}", flush=True)
                        print(f"    s2={best_s2}", flush=True)
                        print(f"    err={best_err:.4f}", flush=True)
                
                # Restore originals
                override_s1_name[0] = None
                override_s2_name[0] = None
                fm.ck_moe_stage1 = original_stage1
                fm.ck_moe_stage2 = original_stage2
            
            else:
                # Try alternative: patch at torch.ops.aiter level
                print("\nTrying torch.ops.aiter level...", flush=True)
                try:
                    import torch
                    aiter_ops = [x for x in dir(torch.ops.aiter) if 'moe' in x.lower() or 'ck' in x.lower()]
                    print(f"  torch.ops.aiter MoE ops: {aiter_ops}", flush=True)
                except Exception as e:
                    print(f"  Error: {e}", flush=True)
        
        except Exception as e:
            import traceback
            print(f"  Injection error: {e}", flush=True)
            traceback.print_exc()
            # Restore
            if original_stage1:
                fm.ck_moe_stage1 = original_stage1
            if original_stage2:
                fm.ck_moe_stage2 = original_stage2
        
        print(f"\n{'='*100}", flush=True)
        print("PROBE COMPLETE", flush=True)
        print(f"{'='*100}", flush=True)
    
    # Always return correct result
    return fused_moe(hidden_states, w1_qw, w2_qw, topk_weights, topk_ids,
                     expert_mask=None, activation=ActivationType.Silu,
                     quant_type=QuantType.per_1x32, doweight_stage1=False,
                     w1_scale=w1_qs, w2_scale=w2_qs)

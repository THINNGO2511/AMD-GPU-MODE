#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Full probe of MoE quant pipeline: source, signature, call site, sort function."""
import torch, subprocess, inspect, os
from task import input_t, output_t
import aiter.fused_moe as fm
from aiter.fused_moe import fused_moe
from aiter import ActivationType, QuantType

_probed = False

def _probe():
    global _probed
    if _probed: return
    _probed = True

    # 1. Where is fused_dynamic_mxfp4_quant_moe_sort defined?
    print("=== 1. Location ===", flush=True)
    r = subprocess.run(['grep', '-rn', 'def fused_dynamic_mxfp4_quant_moe_sort', '/home/runner/aiter/aiter/'],
                      capture_output=True, text=True, timeout=10)
    print(r.stdout[:1000], flush=True)

    # 2. Full signature and source
    print("\n=== 2. Full source ===", flush=True)
    try:
        obj = getattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort', None)
        if obj is None:
            from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort as obj
        src = inspect.getsource(obj)
        print(src[:4000], flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 3. How it's called in fused_moe_2stages
    print("\n=== 3. Call site in fused_moe_2stages ===", flush=True)
    try:
        from aiter.fused_moe import fused_moe_2stages
        src = inspect.getsource(fused_moe_2stages)
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'fused_dynamic_mxfp4_quant_moe_sort' in line or 'quant_func' in line or 'token_num_quant' in line:
                for j in range(max(0,i-2), min(len(lines),i+5)):
                    print(f"  {j}: {lines[j]}", flush=True)
                print("  ...", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 4. Check for separate sort functions
    print("\n=== 4. Sort functions ===", flush=True)
    for name in ['moe_mxfp4_sort', 'moe_sort', 'moe_sorting_fwd', '_moe_sorting_impl',
                 'moe_sorting_opus_fwd', 'fused_dynamic_mxfp4_quant_moe_sort']:
        obj = getattr(fm, name, None)
        if obj:
            print(f"  fm.{name}: EXISTS ({type(obj).__name__})", flush=True)
            try:
                sig = inspect.signature(obj)
                print(f"    sig: {sig}", flush=True)
            except: pass

    # 5. Read the Triton quant kernel source
    print("\n=== 5. Triton kernel source ===", flush=True)
    try:
        r = subprocess.run(['grep', '-rn', 'def _fused_dynamic_mxfp4_quant_moe_sort_kernel',
                           '/home/runner/aiter/aiter/'],
                          capture_output=True, text=True, timeout=10)
        print(f"Kernel location: {r.stdout[:500]}", flush=True)
        # Read the file
        if r.stdout.strip():
            fpath = r.stdout.split(':')[0]
            with open(fpath) as f:
                content = f.read()
            # Find the kernel function
            idx = content.find('def _fused_dynamic_mxfp4_quant_moe_sort_kernel')
            if idx >= 0:
                print(content[idx:idx+3000], flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

    # 6. What does the quant function return?
    print("\n=== 6. Return type ===", flush=True)
    try:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        test = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
        fp4, scale = dynamic_mxfp4_quant(test)
        print(f"  dynamic_mxfp4_quant returns:", flush=True)
        print(f"    fp4: {fp4.shape} {fp4.dtype}", flush=True)
        print(f"    scale: {scale.shape} {scale.dtype}", flush=True)
    except Exception as e:
        print(f"ERROR: {e}", flush=True)

def custom_kernel(data: input_t) -> output_t:
    _probe()
    (hidden_states, guw, dw, guws, dws, guwsh, dwsh, guwssh, dwssh,
     topk_weights, topk_ids, config) = data
    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(hidden_states, guwsh, dwsh, topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=guwssh, w2_scale=dwssh, a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip)

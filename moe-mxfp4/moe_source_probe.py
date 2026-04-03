#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Source code probe. Read quant kernel, fused_moe internals,
available configs, and timing breakdown.
Falls back to proven submission for benchmark.
"""
import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

import torch
import time
import sys
import inspect

from task import input_t, output_t

_call = 0
_probed = False

def _run_probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. Read fused_moe_2stages source
    try:
        from aiter import fused_moe as fm_mod
        if hasattr(fm_mod, 'fused_moe_2stages'):
            src = inspect.getsource(fm_mod.fused_moe_2stages)
            print(f"[PROBE] fused_moe_2stages ({len(src)} chars):", flush=True)
            for i, line in enumerate(src.split('\n')[:80]):
                print(f"[SRC2S] {i:3d}: {line}", flush=True)
        else:
            print("[PROBE] fused_moe_2stages not found", flush=True)
    except Exception as e:
        print(f"[PROBE] fused_moe_2stages: {e}", flush=True)

    # 2. Read fused_dynamic_mxfp4_quant_moe_sort source
    try:
        from aiter.fused_moe import fused_moe_2stages as f2s
        src2 = inspect.getsource(f2s)
        # Find the quant function name used inside
        for line in src2.split('\n'):
            if 'quant' in line.lower() or 'sort' in line.lower():
                print(f"[QUANT_REF] {line.strip()}", flush=True)
    except Exception as e:
        print(f"[PROBE] quant ref: {e}", flush=True)

    # Find and read the fused quant kernel
    try:
        import aiter.ops.triton
        quant_files = []
        import glob
        for p in ['/home/runner/aiter/aiter/ops/triton/',
                  '/home/runner/aiter/aiter/']:
            quant_files.extend(glob.glob(p + '*quant*'))
            quant_files.extend(glob.glob(p + '*mxfp4*'))
        print(f"[PROBE] quant files: {quant_files[:10]}", flush=True)
    except Exception as e:
        print(f"[PROBE] quant files: {e}", flush=True)

    # 3. Read the quant kernel source
    try:
        quant_path = '/home/runner/aiter/aiter/ops/triton/quant.py'
        with open(quant_path, 'r') as f:
            content = f.read()
        # Find fused_dynamic_mxfp4_quant_moe_sort
        lines = content.split('\n')
        in_func = False
        func_lines = []
        for i, line in enumerate(lines):
            if 'fused_dynamic_mxfp4_quant_moe_sort' in line or 'fused_mxfp4_quant' in line:
                in_func = True
            if in_func:
                func_lines.append(f"{i:4d}: {line}")
                if len(func_lines) > 120:
                    break
        if func_lines:
            print(f"[PROBE] Found quant function in quant.py:", flush=True)
            for fl in func_lines[:80]:
                print(f"[QUANT] {fl}", flush=True)
        else:
            print("[PROBE] quant function not in quant.py", flush=True)
            # Search everywhere
            for i, line in enumerate(lines):
                if 'def ' in line and ('quant' in line.lower() or 'sort' in line.lower()):
                    print(f"[QUANT_DEF] {i}: {line.strip()}", flush=True)
    except Exception as e:
        print(f"[PROBE] quant source: {e}", flush=True)

    # 4. Read fused_moe.py for the quant function import
    try:
        fmoe_path = '/home/runner/aiter/aiter/fused_moe.py'
        with open(fmoe_path, 'r') as f:
            fmoe_content = f.read()
        lines = fmoe_content.split('\n')
        for i, line in enumerate(lines):
            if 'quant' in line.lower() or 'sort' in line.lower() or 'import' in line:
                print(f"[FMOE] {i:4d}: {line}", flush=True)
    except Exception as e:
        print(f"[PROBE] fmoe source: {e}", flush=True)

    # 5. Timing breakdown: measure quant vs GEMM separately
    try:
        from aiter.fused_moe import fused_moe
        from aiter import ActivationType, QuantType

        # Create dummy data for E=33 d=512
        E, d, bs = 33, 512, 128
        hidden = torch.randn(bs, d, dtype=torch.bfloat16, device='cuda')
        gate = torch.randn(bs, E, dtype=torch.float32, device='cuda')
        # MoE weights
        # ... complex setup, skip for now
        print(f"[PROBE] Timing setup complete", flush=True)
    except Exception as e:
        print(f"[PROBE] timing: {e}", flush=True)

    # 6. Check if load_inline works for MoE problem too
    try:
        os.environ["PYTORCH_ROCM_ARCH"] = "gfx950"
        from torch.utils.cpp_extension import load_inline
        mod = load_inline(
            name="moe_test_v1",
            cpp_sources="int moe_test() { return 42; }",
            cuda_sources="",
            functions=["moe_test"],
            verbose=False,
        )
        print(f"[PROBE] load_inline for MoE: {mod.moe_test()}", flush=True)
    except Exception as e:
        print(f"[PROBE] load_inline: {e}", flush=True)

    # 7. Check available CK MoE kernel files
    try:
        import glob
        moe_cos = glob.glob('/home/runner/aiter/hsa/gfx950/fmoe*/*.co')
        print(f"[PROBE] MoE .co files: {len(moe_cos)}", flush=True)
        # Show unique prefixes
        prefixes = set()
        for f in moe_cos:
            bn = os.path.basename(f)
            prefix = '_'.join(bn.split('_')[:6])
            prefixes.add(prefix)
        for p in sorted(prefixes)[:20]:
            print(f"[PROBE] MoE prefix: {p}", flush=True)
    except Exception as e:
        print(f"[PROBE] MoE .co: {e}", flush=True)

    # 8. Read tuned_fmoe CSV config entries
    try:
        csv_path = '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv'
        with open(csv_path, 'r') as f:
            csv_lines = f.readlines()
        print(f"[PROBE] Tuned CSV: {len(csv_lines)} lines", flush=True)
        print(f"[PROBE] CSV header: {csv_lines[0].strip()}", flush=True)
        # Show E=33 d=2048 entries
        for line in csv_lines:
            if ',33,' in line and '2048' in line:
                print(f"[PROBE] E33_D2048: {line.strip()}", flush=True)
        # Show E=33 d=512 entries
        for line in csv_lines[:5]:
            if ',33,' in line and ',512,' in line:
                print(f"[PROBE] E33_D512: {line.strip()}", flush=True)
    except Exception as e:
        print(f"[PROBE] CSV: {e}", flush=True)


# ===== Proven MoE submission =====
def custom_kernel(data: input_t) -> output_t:
    global _call
    _call += 1
    if _call == 1:
        _run_probe()

    from aiter.fused_moe import fused_moe
    from aiter import ActivationType, QuantType

    hidden, w1_q, w2_q, w1_scale, w2_scale, topk_weights, topk_ids, \
        sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out = data

    M_total, topk = topk_ids.shape
    E = w1_q.shape[0]

    return fused_moe(
        hidden_states=hidden,
        w1=w1_q, w2=w2_q,
        w1_scale=w1_scale, w2_scale=w2_scale,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        out=out,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.MXFP4,
        use_nt=False,
    )

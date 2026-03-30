from task import input_t, output_t
import torch
import json
import os

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    print(f"=== SHAPE M={m} N={n} K={k} ===")

    cfg_base = "/home/runner/aiter/aiter/ops/triton/configs/gemm"

    # Read ALL config files matching our benchmark shapes
    targets = [
        # A16WFP4 configs
        f"gfx950-GEMM-A16WFP4.json",
        f"gfx950-GEMM-A16WFP4-N={n}-K={k}.json",
        f"gfx950-GEMM-A16WFP4-N=512-K=7168.json",
        f"gfx950-GEMM-A16WFP4-N=7168-K=2048.json",
        # AFP4WFP4 configs
        f"gfx950-GEMM-AFP4WFP4.json",
        f"gfx950-GEMM-AFP4WFP4-N={n}-K={k}.json",
        f"gfx950-GEMM-AFP4WFP4-N=2112-K=7168.json",
        f"gfx950-GEMM-AFP4WFP4-N=3072-K=1536.json",
        f"gfx950-GEMM-AFP4WFP4-N=7168-K=2048.json",
        f"gfx950-GEMM-AFP4WFP4-N=2880-K=512.json",
        f"gfx950-GEMM-AFP4WFP4-N=4096-K=512.json",
        # Preshuffle configs
        f"gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N={n}-K={k}.json",
        f"gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=2112-K=7168.json",
        f"gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=3072-K=1536.json",
        f"gfx950-GEMM-AFP4WFP4_PRESHUFFLED-N=4096-K=512.json",
        # A16WFP4 preshuffle
        f"gfx950-GEMM-A16WFP4_PRESHUFFLED.json",
        f"gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json",
        # Fused configs
        f"gfx950-FUSED-GEMM-AFP4WFP4-A16W16.json",
        f"gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json",
        # A8WFP4
        f"gfx950-GEMM-A8WFP4.json",
        # GEMM_PREQUANT
        f"gfx950-GEMM_PREQUANT-AFP4WFP4.json",
        f"gfx950-GEMM_PREQUANT-AFP4WFP4-N=512-K=7168.json",
    ]

    for fname in targets:
        fpath = os.path.join(cfg_base, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath) as f:
                    data_json = json.load(f)
                print(f"\n### {fname} ###")
                # Print all M-specific configs
                if isinstance(data_json, dict):
                    for key, val in sorted(data_json.items()):
                        if isinstance(val, dict):
                            print(f"  M={key}: {json.dumps(val)}")
                        else:
                            print(f"  {key}: {val}")
                elif isinstance(data_json, list):
                    for i, entry in enumerate(data_json[:20]):
                        print(f"  [{i}]: {json.dumps(entry)}")
            except Exception as e:
                print(f"  ERROR reading {fname}: {e}")
        else:
            print(f"  (not found: {fname})")

    # Also read gemm_a16wfp4.py config loading logic
    print("\n\n### GEMM_A16WFP4 SOURCE — config loading ###")
    try:
        src = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"
        with open(src) as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['config', 'json', 'load', 'read', 'open', 'path']):
                print(f"  {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"  Error: {e}")

    # Check gemm_afp4wfp4_preshuffle exists and its signature
    print("\n\n### PRESHUFFLE API ###")
    try:
        from aiter.ops.triton.gemm.basic import gemm_afp4wfp4_preshuffle
        import inspect
        sig = inspect.signature(gemm_afp4wfp4_preshuffle.gemm_afp4wfp4_preshuffle)
        print(f"  gemm_afp4wfp4_preshuffle signature: {sig}")
    except Exception as e:
        print(f"  preshuffle import error: {e}")

    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        import inspect
        sig = inspect.signature(gemm_afp4wfp4)
        print(f"  gemm_afp4wfp4 signature: {sig}")
    except Exception as e:
        print(f"  afp4wfp4 import error: {e}")

    # Check deepgemm
    print("\n\n### DEEPGEMM API ###")
    try:
        from aiter.ops.triton.gemm.basic import deepgemm
        import inspect
        sig = inspect.signature(deepgemm.deepgemm)
        print(f"  deepgemm signature: {sig}")
    except Exception as e:
        print(f"  deepgemm error: {e}")

    # Return valid result
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    s = B_scale_sh.clone().view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    scale_unshuf = s.view(sm, sn)
    out = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    gemm_a16wfp4(A, B_q.view(torch.uint8), scale_unshuf, dtype=torch.bfloat16, y=out)
    return out

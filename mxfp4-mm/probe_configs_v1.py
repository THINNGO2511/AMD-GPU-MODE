from task import input_t, output_t
import torch
import os
import glob
import json

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    print("=" * 60)
    print(f"GEMM CONFIG PROBE — shape M={m} N={n} K={k}")
    print("=" * 60)

    # 1. List ALL config directories
    cfg_base = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    print(f"\n--- CONFIG DIRECTORY LISTING ---")
    if os.path.isdir(cfg_base):
        for item in sorted(os.listdir(cfg_base)):
            full = os.path.join(cfg_base, item)
            if os.path.isdir(full):
                files = os.listdir(full)
                print(f"  DIR {item}/ ({len(files)} files)")
                for f in sorted(files)[:5]:
                    print(f"    {f}")
                if len(files) > 5:
                    print(f"    ... and {len(files)-5} more")
            else:
                print(f"  FILE {item}")
    else:
        print(f"  {cfg_base} not found, trying alternatives...")
        for alt in ["/home/runner/aiter/aiter/ops/triton/configs",
                    "/home/runner/aiter/aiter/configs"]:
            if os.path.isdir(alt):
                print(f"  Found: {alt}")
                for item in sorted(os.listdir(alt))[:20]:
                    print(f"    {item}")

    # 2. Find ALL gfx950-related config files
    print(f"\n--- ALL GFX950 CONFIG FILES ---")
    for pattern in ["*gfx950*", "*fp4*", "*950*", "*MI355*"]:
        matches = glob.glob(os.path.join(cfg_base, "**", pattern), recursive=True)
        if matches:
            print(f"Pattern '{pattern}': {len(matches)} files")
            for f in sorted(matches)[:10]:
                print(f"  {os.path.relpath(f, cfg_base)}")

    # 3. Read ALL config files (JSON) that could be relevant
    print(f"\n--- CONFIG FILE CONTENTS ---")
    all_cfgs = glob.glob(os.path.join(cfg_base, "**", "*.json"), recursive=True)
    if not all_cfgs:
        all_cfgs = glob.glob(os.path.join(cfg_base, "**", "*"), recursive=True)
    print(f"Total config files found: {len(all_cfgs)}")

    for cfg_file in sorted(all_cfgs)[:30]:
        if os.path.isfile(cfg_file):
            try:
                with open(cfg_file) as f:
                    content = f.read()
                rel = os.path.relpath(cfg_file, cfg_base)
                if len(content) < 2000:
                    print(f"\n  === {rel} ===")
                    print(f"  {content[:1500]}")
                else:
                    print(f"\n  === {rel} === (truncated, {len(content)} bytes)")
                    print(f"  {content[:1500]}")
            except:
                pass

    # 4. Check how gemm_a16wfp4 reads configs
    print(f"\n--- GEMM_A16WFP4 CONFIG LOADING ---")
    try:
        import subprocess
        r = subprocess.run(["grep", "-n", "config\|CONFIG\|json\|csv\|tune",
                           "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:2000])
    except Exception as e:
        print(f"Error: {e}")

    # 5. Check gemm_afp4wfp4 configs too
    print(f"\n--- GEMM_AFP4WFP4 CONFIG LOADING ---")
    try:
        r = subprocess.run(["grep", "-n", "config\|CONFIG\|json\|csv\|tune",
                           "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py"],
                          capture_output=True, text=True, timeout=5)
        print(r.stdout[:2000])
    except Exception as e:
        print(f"Error: {e}")

    # 6. Check what configs get selected for our benchmark shapes
    print(f"\n--- DEFAULT CONFIG SELECTION ---")
    try:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        # Temporarily capture what config gets used
        shapes = [(4, 7168, 2112), (16, 1536, 3072), (32, 512, 2880),
                  (64, 2048, 3584), (256, 512, 2880), (16, 7168, 2112)]
        for sm, sk, sn in shapes:
            print(f"  Shape M={sm} N={sn} K={sk}: checking...")
    except Exception as e:
        print(f"Error loading gemm_a16wfp4: {e}")

    # 7. Read the actual Triton kernel source for config handling
    print(f"\n--- GEMM_A16WFP4 SOURCE (config section) ---")
    try:
        src_path = "/home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py"
        with open(src_path) as f:
            lines = f.readlines()
        # Print lines mentioning config
        for i, line in enumerate(lines):
            if any(kw in line.lower() for kw in ['config', 'block_size', 'ksplit', 'num_stages', 'waves']):
                print(f"  {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)

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

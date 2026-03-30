#!POPCORN leaderboard amd-mixed-mla
#!POPCORN gpu MI355X
"""
Probe: Dump aiter MLA internals - page64, 3-buffer, a16w8, kernel binaries.
"""
import torch
import sys
import os
import inspect
import importlib
from task import input_t, output_t

def custom_kernel(data: input_t) -> output_t:
    q, kv_data, qo_indptr, kv_indptr, config = data

    print("=" * 80)
    print("PROBE: aiter MLA internals")
    print("=" * 80)

    # 1. Check aiter.mla source
    try:
        import aiter.mla as mla_mod
        print("\n--- aiter.mla source file ---")
        print(inspect.getfile(mla_mod))
        print("\n--- aiter.mla dir() ---")
        print([x for x in dir(mla_mod) if not x.startswith('_')])

        # Check mla_decode_fwd signature
        print("\n--- mla_decode_fwd signature ---")
        sig = inspect.signature(mla_mod.mla_decode_fwd)
        print(sig)

        # Full source of mla_decode_fwd
        print("\n--- mla_decode_fwd source ---")
        try:
            src = inspect.getsource(mla_mod.mla_decode_fwd)
            print(src[:8000])
        except Exception as e:
            print(f"Cannot get source: {e}")

    except Exception as e:
        print(f"Error inspecting aiter.mla: {e}")

    # 2. Check for page64 / 3-buffer related functions
    try:
        print("\n--- Search for page64/3buffer in aiter ---")
        import aiter
        aiter_dir = os.path.dirname(inspect.getfile(aiter))
        print(f"aiter location: {aiter_dir}")

        # List all .py files
        for root, dirs, files in os.walk(aiter_dir):
            for f in files:
                if f.endswith('.py'):
                    fpath = os.path.join(root, f)
                    try:
                        with open(fpath) as fh:
                            content = fh.read()
                            if any(kw in content.lower() for kw in ['page64', '3buffer', 'three_buffer', 'nope', 'rope_buf', 'ds32', 'page_size=64']):
                                print(f"\n  MATCH: {fpath}")
                                # Print matching lines
                                for i, line in enumerate(content.split('\n')):
                                    if any(kw in line.lower() for kw in ['page64', '3buffer', 'three_buffer', 'nope', 'rope_buf', 'ds32', 'page_size=64']):
                                        print(f"    L{i+1}: {line.rstrip()}")
                    except:
                        pass
    except Exception as e:
        print(f"Error searching aiter: {e}")

    # 3. Check kernel binaries
    try:
        print("\n--- Kernel binaries (.co files) ---")
        import aiter
        aiter_dir = os.path.dirname(inspect.getfile(aiter))
        co_files = []
        for root, dirs, files in os.walk(aiter_dir):
            for f in files:
                if f.endswith('.co') and 'mla' in f.lower():
                    co_files.append(os.path.join(root, f))
        co_files.sort()
        for cf in co_files[:50]:
            print(f"  {cf}")
        print(f"\n  Total MLA .co files: {len(co_files)}")
    except Exception as e:
        print(f"Error listing kernels: {e}")

    # 4. Check if mla_decode_fwd accepts page_size > 1
    try:
        print("\n--- mla_decode_fwd full parameter list ---")
        sig = inspect.signature(mla_mod.mla_decode_fwd)
        for name, param in sig.parameters.items():
            print(f"  {name}: {param.default if param.default != inspect.Parameter.empty else 'REQUIRED'}")
    except Exception as e:
        print(f"Error: {e}")

    # 5. Check for other MLA functions
    try:
        print("\n--- All MLA-related functions in aiter ---")
        for name in dir(mla_mod):
            if 'mla' in name.lower() or 'decode' in name.lower() or 'page' in name.lower():
                obj = getattr(mla_mod, name)
                if callable(obj):
                    try:
                        print(f"  {name}{inspect.signature(obj)}")
                    except:
                        print(f"  {name} (no signature)")
    except Exception as e:
        print(f"Error: {e}")

    # 6. Check aiter.ops or aiter.jit_kernels for MLA-related stuff
    try:
        print("\n--- aiter submodules ---")
        for submod_name in ['ops', 'jit_kernels', 'ops.mla', 'jit_kernels.mla']:
            try:
                submod = importlib.import_module(f'aiter.{submod_name}')
                fns = [x for x in dir(submod) if 'mla' in x.lower() or 'decode' in x.lower()]
                if fns:
                    print(f"  aiter.{submod_name}: {fns}")
            except ImportError:
                pass
    except Exception as e:
        print(f"Error: {e}")

    # 7. Read the full mla.py source
    try:
        print("\n--- Full aiter/mla.py source (first 200 lines) ---")
        mla_path = inspect.getfile(mla_mod)
        with open(mla_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines[:200]):
            print(f"{i+1:4d}: {line.rstrip()}")
        print(f"\n  Total lines: {len(lines)}")
    except Exception as e:
        print(f"Error: {e}")

    # 8. Check what kernel gets dispatched for different page sizes
    try:
        print("\n--- Checking mla dispatch logic ---")
        # Look for _get_kernel_name or similar dispatch
        mla_path = inspect.getfile(mla_mod)
        with open(mla_path) as f:
            content = f.read()
        for kw in ['page_size', 'page64', 'kernel_name', 'dispatch', '.co', 'load_module', 'hipModule']:
            matches = [(i+1, l.rstrip()) for i, l in enumerate(content.split('\n')) if kw in l]
            if matches:
                print(f"\n  '{kw}' occurrences:")
                for ln, lt in matches[:10]:
                    print(f"    L{ln}: {lt}")
    except Exception as e:
        print(f"Error: {e}")

    # 9. Check for environment variables or config that affect kernel selection
    try:
        print("\n--- Relevant env vars ---")
        for k, v in sorted(os.environ.items()):
            if any(kw in k.lower() for kw in ['aiter', 'mla', 'hip', 'roc', 'gpu', 'cuda']):
                print(f"  {k}={v}")
    except Exception as e:
        print(f"Error: {e}")

    # 10. Check aiter version and installation
    try:
        print("\n--- aiter version info ---")
        import aiter
        print(f"  aiter version: {getattr(aiter, '__version__', 'unknown')}")
        print(f"  aiter path: {os.path.dirname(inspect.getfile(aiter))}")
        # Check git info
        aiter_root = os.path.dirname(os.path.dirname(inspect.getfile(aiter)))
        git_dir = os.path.join(aiter_root, '.git')
        if os.path.exists(git_dir):
            head_file = os.path.join(git_dir, 'HEAD')
            if os.path.exists(head_file):
                with open(head_file) as f:
                    print(f"  git HEAD: {f.read().strip()}")
    except Exception as e:
        print(f"Error: {e}")

    # 11. Check for any page64 kernel binary specifically
    try:
        print("\n--- Searching for page64 kernel binary ---")
        import aiter
        aiter_dir = os.path.dirname(inspect.getfile(aiter))
        for root, dirs, files in os.walk(aiter_dir):
            for f in files:
                if 'page' in f.lower() and f.endswith('.co'):
                    print(f"  FOUND: {os.path.join(root, f)}")
    except Exception as e:
        print(f"Error: {e}")

    # 12. Get aiter/mla.py lines 200-400
    try:
        print("\n--- aiter/mla.py lines 200-400 ---")
        mla_path = inspect.getfile(mla_mod)
        with open(mla_path) as f:
            lines = f.readlines()
        for i, line in enumerate(lines[200:400], start=201):
            print(f"{i:4d}: {line.rstrip()}")
    except Exception as e:
        print(f"Error: {e}")

    # Produce valid output
    nq = config["num_heads"]
    dv = config["v_head_dim"]
    return torch.zeros((q.shape[0], nq, dv), dtype=torch.bfloat16, device=q.device)

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Kernel Probe: Dump full source of fused_moe, fused_moe_2stages,
quantization, sorting, CK kernel selection, CSV configs.
Goal: find hidden params (blockPerCu, cktile), d=2048 kernel paths, new APIs.
"""
import os
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_dumped = False
_call_count = 0


# ─── Source dump helpers ───
def _dump_file(path, label):
    try:
        if os.path.isfile(path):
            with open(path) as f:
                src = f.read()
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DUMP] {label}: {path} ({len(src)} bytes, {src.count(chr(10))} lines)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            lines = src.split('\n')
            chunk_size = 100
            for i in range(0, len(lines), chunk_size):
                chunk = '\n'.join(lines[i:i+chunk_size])
                print(chunk, file=sys.stderr)
            return src
        else:
            print(f"[DUMP] {label}: FILE NOT FOUND: {path}", file=sys.stderr)
            return None
    except Exception as e:
        print(f"[DUMP] {label}: ERROR: {e}", file=sys.stderr)
        return None


def _list_dir(path, label):
    try:
        if os.path.isdir(path):
            entries = []
            for root, dirs, files in os.walk(path):
                for f in sorted(files):
                    fp = os.path.join(root, f)
                    sz = os.path.getsize(fp)
                    rel = os.path.relpath(fp, path)
                    entries.append((rel, sz))
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"[DIR] {label}: {path} ({len(entries)} files)", file=sys.stderr)
            print(f"{'='*80}", file=sys.stderr)
            for rel, sz in sorted(entries):
                print(f"  {sz:>8} {rel}", file=sys.stderr)
        else:
            print(f"[DIR] {label}: NOT FOUND: {path}", file=sys.stderr)
    except Exception as e:
        print(f"[DIR] {label}: ERROR: {e}", file=sys.stderr)


def _do_dump():
    global _dumped
    if _dumped:
        return
    _dumped = True

    # 1. Core fused_moe.py — the entry point
    _dump_file("/home/runner/aiter/aiter/fused_moe.py", "fused_moe.py (entry)")

    # 2. fused_moe_2stages — the 2-stage CK pipeline
    # Search for it in multiple possible locations
    for candidate in [
        "/home/runner/aiter/aiter/ops/triton/fused_moe_2stages.py",
        "/home/runner/aiter/aiter/fused_moe_2stages.py",
        "/home/runner/aiter/aiter/ops/fused_moe_2stages.py",
    ]:
        _dump_file(candidate, f"fused_moe_2stages ({candidate})")

    # 3. Fused dynamic quant + moe sort kernel
    _dump_file("/home/runner/aiter/aiter/ops/triton/fused_dynamic_mxfp4_quant_moe_sort.py",
               "fused_dynamic_mxfp4_quant_moe_sort.py")

    # 4. MoE sorting implementation
    for candidate in [
        "/home/runner/aiter/aiter/ops/triton/moe_sorting.py",
        "/home/runner/aiter/aiter/moe_sorting.py",
        "/home/runner/aiter/aiter/ops/moe_sorting.py",
    ]:
        _dump_file(candidate, f"moe_sorting ({candidate})")

    # 5. Find ALL MoE-related Python files
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[SEARCH] All MoE-related files in aiter:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    moe_files = []
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py') and ('moe' in fname.lower() or 'fused_moe' in fname.lower()
                                          or 'expert' in fname.lower()):
                fpath = os.path.join(root, fname)
                sz = os.path.getsize(fpath)
                moe_files.append(fpath)
                print(f"  {sz:>8} {fpath}", file=sys.stderr)

    # Dump each MoE file we found
    for fpath in moe_files:
        _dump_file(fpath, f"MoE file: {os.path.basename(fpath)}")

    # 6. List ALL fmoe_2stages .co files
    _list_dir("/home/runner/aiter/hsa/gfx950/fmoe_2stages/", "fmoe_2stages .co kernels")

    # 7. List ALL fmoe_1stage .co files
    _list_dir("/home/runner/aiter/hsa/gfx950/fmoe_1stage/", "fmoe_1stage .co kernels")

    # 8. Any other MoE .co directories
    hsa_dir = "/home/runner/aiter/hsa/gfx950/"
    if os.path.isdir(hsa_dir):
        subdirs = sorted(os.listdir(hsa_dir))
        moe_dirs = [d for d in subdirs if 'moe' in d.lower() or 'fmoe' in d.lower()
                    or 'expert' in d.lower() or 'flydsl' in d.lower()]
        print(f"\n[HSA] MoE-related subdirs: {moe_dirs}", file=sys.stderr)
        for d in moe_dirs:
            _list_dir(os.path.join(hsa_dir, d), f"hsa/{d}")

    # 9. Dump the tuned CSV configs
    for csv_path in [
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    ]:
        _dump_file(csv_path, f"CSV: {os.path.basename(csv_path)}")

    # 10. Check for blockPerCu parameter
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[BLOCKPERCU] Searching for blockPerCu / block_per_cu...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    cl = content.lower()
                    if 'blockpercu' in cl or 'block_per_cu' in cl or 'blockspercu' in cl:
                        print(f"\n  FOUND in {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            ll = line.lower()
                            if 'blockpercu' in ll or 'block_per_cu' in ll or 'blockspercu' in ll:
                                print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 11. Check for cktile / ck_tile path
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[CKTILE] Searching for cktile paths...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'cktile' in content.lower() or 'ck_tile' in content.lower():
                        print(f"\n  {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            ll = line.lower()
                            if 'cktile' in ll or 'ck_tile' in ll:
                                print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 12. Check for d=2048 specific kernel handling
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[D2048] Searching for d=2048 / intermediate=2048...", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py'):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if '2048' in content and ('inter' in content.lower() or 'expert' in content.lower()
                                              or 'dim' in content.lower()):
                        has_relevant = any('2048' in line and (
                            'inter' in line.lower() or 'dim' in line.lower() or 'expert' in line.lower()
                            or 'kernel' in line.lower() or 'tile' in line.lower()
                        ) for line in content.split('\n'))
                        if has_relevant:
                            print(f"\n  {fpath}:", file=sys.stderr)
                            for i, line in enumerate(content.split('\n')):
                                if '2048' in line and len(line.strip()) < 200:
                                    print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 13. Check fused_moe module attributes and APIs
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[FM-API] aiter.fused_moe module inspection:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    import inspect
    all_fm_attrs = sorted(dir(fm))
    print(f"  All attrs ({len(all_fm_attrs)}): {all_fm_attrs}", file=sys.stderr)
    for attr in all_fm_attrs:
        if attr.startswith('__'):
            continue
        obj = getattr(fm, attr)
        if callable(obj):
            try:
                sig = inspect.signature(obj)
                print(f"  {attr}{sig}", file=sys.stderr)
            except:
                print(f"  {attr} (no sig)", file=sys.stderr)
        elif not attr.startswith('_'):
            print(f"  {attr} = {repr(obj)[:200]}", file=sys.stderr)

    # 14. Check torch.ops.aiter for MoE ops
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[OPS] torch.ops.aiter MoE-related:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        all_ops = sorted(dir(torch.ops.aiter))
        moe_ops = [a for a in all_ops if 'moe' in a.lower() or 'fused' in a.lower()
                   or 'sort' in a.lower() or 'expert' in a.lower()]
        print(f"  MoE ops: {moe_ops}", file=sys.stderr)
        print(f"  ALL ops ({len(all_ops)}): {all_ops}", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 15. Check for FlyDSL kernel files
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[FLYDSL] FlyDSL search:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/"):
        for fname in files:
            if 'flydsl' in fname.lower():
                fpath = os.path.join(root, fname)
                sz = os.path.getsize(fpath)
                print(f"  {sz:>8} {fpath}", file=sys.stderr)
    # Also search in hsa directory
    for root, dirs, files in os.walk("/home/runner/aiter/hsa/"):
        for fname in files:
            if 'flydsl' in fname.lower() or 'fly_dsl' in fname.lower():
                fpath = os.path.join(root, fname)
                print(f"  {fpath}", file=sys.stderr)

    # 16. Check get_2stage_cfgs signature and source
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[2STAGE-CFG] get_2stage_cfgs details:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        fn = fm.get_2stage_cfgs
        # Check if wrapped
        wrapped = getattr(fn, '__wrapped__', None)
        if wrapped:
            sig = inspect.signature(wrapped)
            print(f"  __wrapped__ sig: {sig}", file=sys.stderr)
            try:
                src = inspect.getsource(wrapped)
                print(f"  __wrapped__ source ({len(src)} bytes):", file=sys.stderr)
                print(src, file=sys.stderr)
            except:
                print(f"  (could not get source of __wrapped__)", file=sys.stderr)
        sig = inspect.signature(fn)
        print(f"  Current sig: {sig}", file=sys.stderr)
        try:
            src = inspect.getsource(fn)
            print(f"  Source ({len(src)} bytes):", file=sys.stderr)
            print(src, file=sys.stderr)
        except:
            print(f"  (could not get source)", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 17. MOEMetadata class definition
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[MOEMETA] MOEMetadata class:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    try:
        meta_cls = fm.MOEMetadata
        print(f"  Fields: {meta_cls.__annotations__ if hasattr(meta_cls, '__annotations__') else 'N/A'}", file=sys.stderr)
        try:
            src = inspect.getsource(meta_cls)
            print(f"  Source:", file=sys.stderr)
            print(src, file=sys.stderr)
        except:
            print(f"  (could not get source)", file=sys.stderr)
    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)

    # 18. Check CK stage1/stage2 function signatures
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[CK-STAGE] CK stage function signatures:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for fn_name in ['ck_moe_stage1', 'ck_moe_stage1_fwd', 'ck_moe_stage2_fwd',
                    '_ck_moe_stage1_wrapper', '_ck_moe_stage2_wrapper',
                    '_flydsl_stage2_wrapper', 'fused_moe_2stages',
                    '_moe_sorting_impl', 'moe_sorting_opus_fwd']:
        fn = getattr(fm, fn_name, None)
        if fn is None:
            fn = getattr(aiter, fn_name, None)
        if fn is not None:
            try:
                sig = inspect.signature(fn)
                print(f"  {fn_name}{sig}", file=sys.stderr)
            except:
                print(f"  {fn_name} (no sig)", file=sys.stderr)
            try:
                src = inspect.getsource(fn)
                print(f"  Source of {fn_name} ({len(src)} bytes):", file=sys.stderr)
                print(src, file=sys.stderr)
            except:
                pass
        else:
            print(f"  {fn_name}: NOT FOUND", file=sys.stderr)

    # 19. Check for any env vars affecting MoE
    print(f"\n{'='*80}", file=sys.stderr)
    print(f"[ENV] MoE env vars in source:", file=sys.stderr)
    print(f"{'='*80}", file=sys.stderr)
    for root, dirs, files in os.walk("/home/runner/aiter/aiter/"):
        for fname in files:
            if fname.endswith('.py') and ('moe' in fname.lower() or 'fused' in fname.lower()):
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath) as f:
                        content = f.read()
                    if 'os.environ' in content or 'getenv' in content or 'AITER' in content:
                        print(f"\n  {fpath}:", file=sys.stderr)
                        for i, line in enumerate(content.split('\n')):
                            if 'os.environ' in line or 'getenv' in line or 'AITER' in line:
                                print(f"    L{i+1}: {line.strip()}", file=sys.stderr)
                except:
                    pass

    # 20. Dump the quant.py for MoE quantization
    _dump_file("/home/runner/aiter/aiter/ops/triton/quant.py", "quant.py (shared with GEMM)")


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Keep proven optimizations
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)

    try:
        fm._USE_OPUS_MOE_SORTING = True
    except:
        pass

    # Wrap get_2stage_cfgs for kernel injection (E<=64, d<2048)
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def patched_get_2stage(*args, **kwargs):
        result = orig_get_2stage(*args, **kwargs)

        # Log first few calls
        if _call_count <= 10:
            try:
                print(f"\n[2STAGE] args={args[:6]} block_m={result.block_m} "
                      f"run_1stage={result.run_1stage}", file=sys.stderr)
                if not result.run_1stage:
                    s1kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    s2kw = result.stage2.keywords if hasattr(result.stage2, 'keywords') else {}
                    print(f"  s1_kernel={s1kw.get('kernelName', 'DEFAULT')}", file=sys.stderr)
                    print(f"  s2_kernel={s2kw.get('kernelName', 'DEFAULT')}", file=sys.stderr)
                    for k, v in s1kw.items():
                        if k != 'kernelName':
                            print(f"  s1.{k}={v}", file=sys.stderr)
                    for k, v in s2kw.items():
                        if k != 'kernelName':
                            print(f"  s2.{k}={v}", file=sys.stderr)
            except:
                pass

        # Inject CK kernels for E<=64 d<2048
        STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
        STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
        STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

        try:
            if len(args) >= 5:
                token, model_dim, inter_dim, expert, topk = args[:5]
                q_type = args[8] if len(args) > 8 else None
                activation = args[10] if len(args) > 10 else None
                dtype = args[5] if len(args) > 5 else None

                if (expert <= 64 and q_type == QuantType.per_1x32
                        and not result.run_1stage and inter_dim < 2048):
                    kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                    if not kw.get('kernelName', ''):
                        est_m = token * topk // expert
                        kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                        return fm.MOEMetadata(
                            functools.partial(fm.ck_moe_stage1,
                                kernelName=kn1, activation=activation,
                                quant_type=q_type, dtype=dtype,
                                splitk=0, use_non_temporal_load=False),
                            functools.partial(aiter.ck_moe_stage2_fwd,
                                kernelName=STAGE2_32, activation=activation,
                                quant_type=q_type, use_non_temporal_load=False),
                            32, 0, False)
        except:
            pass

        return result

    fm.get_2stage_cfgs = patched_get_2stage
    fm.cfg_2stages = None


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()
    _do_dump()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    if _call_count <= 10:
        bs = hidden_states.shape[0]
        E = config.get("n_expert", 0)
        d = config.get("d_expert", 0)
        dh = config.get("d_hidden", 0)
        dhp = config.get("d_hidden_pad", 0)
        dep = config.get("d_expert_pad", 0)
        print(f"\n[CALL {_call_count}] bs={bs} E={E} d={d} dh={dh} dhp={dhp} dep={dep}",
              file=sys.stderr)

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
Deep probe for d=2048 E=33 bs=512 kernel selection.
Goal: understand EXACTLY what the heuristic selects and what alternatives exist.

Probes:
1. Full get_2stage_cfgs call for (token=512, model_dim=7168, inter_dim=2048, expert=33, topk=9)
2. All available .co kernel names with their tile sizes
3. All kernel variants: Nswizzle0 vs Nswizzle1, MulRoutedWeight0 vs 1, v1 vs v3 vs v4
4. doweight_stage1=True path
5. non_temporal_load variations for d=2048
6. Per-phase timing breakdown for d=2048
7. Intercept and log the EXACT kernel name the heuristic returns
8. Try every FP4 stage1 + stage2 combination for d=2048 shape

NOTE: This is a PROBE submission. It logs everything to stderr/stdout, then runs
      the default kernel for correctness.
"""
import torch
import functools
import inspect
import os
import sys
import time
import glob as globmod
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False
_call_count = 0

P = lambda *a, **k: print(*a, **k, file=sys.stderr, flush=True)


def _probe_all():
    global _probed
    if _probed:
        return
    _probed = True

    P("\n" + "=" * 80)
    P("=== D2048 DEEP PROBE START ===")
    P("=" * 80)

    # ──────────────────────────────────────────────────────────────────
    # SECTION 1: get_2stage_cfgs source code (unwrapped original)
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[1] get_2stage_cfgs FULL SOURCE (original, unwrapped)")
    P("=" * 80)
    try:
        orig_fn = fm.get_2stage_cfgs
        if hasattr(orig_fn, '__wrapped__'):
            orig_fn = orig_fn.__wrapped__
        src = inspect.getsource(orig_fn)
        for i, line in enumerate(src.split('\n')):
            P(f"  {i:4d}: {line}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 2: MOEMetadata class
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[2] MOEMetadata class definition")
    P("=" * 80)
    try:
        src = inspect.getsource(fm.MOEMetadata)
        for i, line in enumerate(src.split('\n')):
            P(f"  {i:4d}: {line}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 3: ALL .co files in fmoe_2stages directory
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[3] ALL .co kernel files in fmoe_2stages/")
    P("=" * 80)
    fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    all_stage1 = []
    all_stage2 = []
    all_other = []

    if os.path.isdir(fmoe_dir):
        for f in sorted(os.listdir(fmoe_dir)):
            if f.endswith('.co'):
                name = f[:-3]
                sz = os.path.getsize(os.path.join(fmoe_dir, f))
                if 'gemm1' in name:
                    all_stage1.append((name, sz))
                elif 'gemm2' in name:
                    all_stage2.append((name, sz))
                else:
                    all_other.append((name, sz))

        P(f"\n  Stage1 kernels ({len(all_stage1)}):")
        for name, sz in all_stage1:
            P(f"    [{sz:>7} bytes] {name}")

        P(f"\n  Stage2 kernels ({len(all_stage2)}):")
        for name, sz in all_stage2:
            P(f"    [{sz:>7} bytes] {name}")

        if all_other:
            P(f"\n  Other kernels ({len(all_other)}):")
            for name, sz in all_other:
                P(f"    [{sz:>7} bytes] {name}")
    else:
        P(f"  DIR NOT FOUND: {fmoe_dir}")
        # Try to find it
        for cand in globmod.glob("/home/runner/aiter/hsa/**/fmoe*", recursive=True):
            P(f"  Found: {cand}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 4: Parse kernel names — extract all variant dimensions
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[4] KERNEL VARIANT ANALYSIS (all FP4 kernels)")
    P("=" * 80)

    fp4_s1 = [(n, s) for n, s in all_stage1 if 'FP4' in n]
    fp4_s2 = [(n, s) for n, s in all_stage2 if 'FP4' in n]

    # Parse kernel name components
    def parse_kernel(name):
        """Extract key parameters from kernel name."""
        info = {}
        # Tile size: first NxNxNxN pattern after gemm1_ or gemm2_
        parts = name.split('_')
        for i, p in enumerate(parts):
            if p and p[0].isdigit() and 'x' in p:
                dims = p.split('x')
                if len(dims) == 4:
                    info['tile'] = p
                    info['tile_M'] = int(dims[0])
                    info['tile_N'] = int(dims[1])
                    info['tile_K0'] = int(dims[2])
                    info['tile_K1'] = int(dims[3])
                    break
        # Wave mapping: NxN pattern right after tile
        for i, p in enumerate(parts):
            if p and p[0].isdigit() and 'x' in p:
                dims = p.split('x')
                if len(dims) == 2 and all(d.isdigit() for d in dims):
                    info['wave_map'] = p
                    break
        # Version: v1, v3, v4, etc.
        for p in parts:
            if p in ('v1', 'v3', 'v4'):
                info['version'] = p
        # Nswizzle
        for p in parts:
            if p.startswith('Nswizzle'):
                info['nswizzle'] = p
        # MulRoutedWeight
        for p in parts:
            if p.startswith('MulRoutedWeight'):
                info['mul_routed_weight'] = p
        # Quant
        for p in parts:
            if p.startswith('Quant'):
                info['quant'] = p
        # Scale type
        for p in parts:
            if 'MulABScale' in p:
                info['scale_type'] = p
        # activation (silu, etc.)
        if 'silu' in name.lower():
            info['activation'] = 'silu'
        return info

    P("\n  FP4 STAGE1 KERNEL VARIANTS:")
    unique_tiles_s1 = set()
    unique_versions_s1 = set()
    unique_nswizzle_s1 = set()
    unique_mulrw_s1 = set()
    unique_scaletype_s1 = set()

    for name, sz in fp4_s1:
        info = parse_kernel(name)
        unique_tiles_s1.add(info.get('tile', '?'))
        unique_versions_s1.add(info.get('version', '?'))
        unique_nswizzle_s1.add(info.get('nswizzle', '?'))
        unique_mulrw_s1.add(info.get('mul_routed_weight', '?'))
        unique_scaletype_s1.add(info.get('scale_type', '?'))
        P(f"    tile={info.get('tile','?'):>20s} "
          f"ver={info.get('version','?'):>3s} "
          f"nsw={info.get('nswizzle','?'):>11s} "
          f"mrw={info.get('mul_routed_weight','?'):>18s} "
          f"scale={info.get('scale_type','?'):>30s}")

    P(f"\n  Unique tiles:      {sorted(unique_tiles_s1)}")
    P(f"  Unique versions:   {sorted(unique_versions_s1)}")
    P(f"  Unique Nswizzle:   {sorted(unique_nswizzle_s1)}")
    P(f"  Unique MulRoutedW: {sorted(unique_mulrw_s1)}")
    P(f"  Unique ScaleType:  {sorted(unique_scaletype_s1)}")

    P("\n  FP4 STAGE2 KERNEL VARIANTS:")
    unique_tiles_s2 = set()
    unique_versions_s2 = set()
    unique_nswizzle_s2 = set()
    unique_mulrw_s2 = set()
    unique_scaletype_s2 = set()

    for name, sz in fp4_s2:
        info = parse_kernel(name)
        unique_tiles_s2.add(info.get('tile', '?'))
        unique_versions_s2.add(info.get('version', '?'))
        unique_nswizzle_s2.add(info.get('nswizzle', '?'))
        unique_mulrw_s2.add(info.get('mul_routed_weight', '?'))
        unique_scaletype_s2.add(info.get('scale_type', '?'))
        P(f"    tile={info.get('tile','?'):>20s} "
          f"ver={info.get('version','?'):>3s} "
          f"nsw={info.get('nswizzle','?'):>11s} "
          f"mrw={info.get('mul_routed_weight','?'):>18s} "
          f"scale={info.get('scale_type','?'):>30s}")

    P(f"\n  Unique tiles:      {sorted(unique_tiles_s2)}")
    P(f"  Unique versions:   {sorted(unique_versions_s2)}")
    P(f"  Unique Nswizzle:   {sorted(unique_nswizzle_s2)}")
    P(f"  Unique MulRoutedW: {sorted(unique_mulrw_s2)}")
    P(f"  Unique ScaleType:  {sorted(unique_scaletype_s2)}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 5: ck_moe_stage1 and ck_moe_stage2_fwd full signatures + source
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[5] CK STAGE FUNCTION SIGNATURES AND SOURCE")
    P("=" * 80)

    for fn_name, fn_obj in [
        ('fm.ck_moe_stage1', getattr(fm, 'ck_moe_stage1', None)),
        ('aiter.ck_moe_stage1_fwd', getattr(aiter, 'ck_moe_stage1_fwd', None)),
        ('aiter.ck_moe_stage2_fwd', getattr(aiter, 'ck_moe_stage2_fwd', None)),
        ('fm._ck_moe_stage1_wrapper', getattr(fm, '_ck_moe_stage1_wrapper', None)),
        ('fm._ck_moe_stage2_wrapper', getattr(fm, '_ck_moe_stage2_wrapper', None)),
        ('fm._flydsl_stage2_wrapper', getattr(fm, '_flydsl_stage2_wrapper', None)),
    ]:
        if fn_obj is not None:
            P(f"\n  {fn_name}:")
            try:
                sig = inspect.signature(fn_obj)
                P(f"    Signature: {sig}")
            except Exception as e:
                P(f"    Signature error: {e}")
            try:
                src = inspect.getsource(fn_obj)
                P(f"    Source ({len(src)} bytes):")
                for i, line in enumerate(src.split('\n')[:60]):
                    P(f"      {i:4d}: {line}")
                if len(src.split('\n')) > 60:
                    P(f"      ... ({len(src.split(chr(10)))} total lines)")
            except Exception as e:
                P(f"    Source error: {e}")
        else:
            P(f"\n  {fn_name}: NOT FOUND")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 6: use_nt and get_block_size_M source
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[6] use_nt and get_block_size_M source")
    P("=" * 80)

    for fn_name, fn_obj in [
        ('fm.use_nt', fm.use_nt),
        ('fm.get_block_size_M', fm.get_block_size_M),
        ('fm.get_padded_M', getattr(fm, 'get_padded_M', None)),
        ('fm.get_inter_dim', getattr(fm, 'get_inter_dim', None)),
    ]:
        if fn_obj is not None:
            P(f"\n  {fn_name}:")
            try:
                sig = inspect.signature(fn_obj)
                P(f"    Signature: {sig}")
                src = inspect.getsource(fn_obj)
                for line in src.split('\n')[:30]:
                    P(f"      {line}")
            except Exception as e:
                P(f"    Error: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 7: CSV matching for d=2048 shapes
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[7] CSV CONFIG MATCHING FOR d=2048")
    P("=" * 80)

    csv_paths = [
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
    ]

    for csv_path in csv_paths:
        P(f"\n  {csv_path}:")
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            header = lines[0].strip()
            P(f"    Header: {header}")
            cols = header.split(',')
            P(f"    Columns ({len(cols)}): {cols}")

            # Find ALL E=33 entries
            e33_lines = []
            for line in lines[1:]:
                if ',33,' in line:
                    e33_lines.append(line.strip())

            P(f"\n    E=33 entries ({len(e33_lines)}):")
            for line in e33_lines:
                P(f"      {line}")

            # Also find entries with inter_dim=2048 (or N related to 2048)
            d2048_lines = []
            for line in lines[1:]:
                fields = line.strip().split(',')
                for field in fields:
                    if '2048' in field or '4096' in field:
                        d2048_lines.append(line.strip())
                        break

            P(f"\n    Entries containing 2048/4096 ({len(d2048_lines)}):")
            for line in d2048_lines[:30]:
                P(f"      {line}")
            if len(d2048_lines) > 30:
                P(f"      ... ({len(d2048_lines)} total)")

            # Show first 10 lines as reference
            P(f"\n    First 10 data lines:")
            for line in lines[1:11]:
                P(f"      {line.strip()}")
        except Exception as e:
            P(f"    Error: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 8: fused_moe_2stages full source
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[8] fused_moe_2stages FULL SOURCE")
    P("=" * 80)
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        for i, line in enumerate(src.split('\n')):
            P(f"  {i:4d}: {line}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 9: fused_moe (top-level wrapper) source
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[9] fused_moe (top-level) source")
    P("=" * 80)
    try:
        src = inspect.getsource(fused_moe)
        for i, line in enumerate(src.split('\n')):
            P(f"  {i:4d}: {line}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 10: Check for FlyDSL + 1stage dirs
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[10] HSA DIRECTORY LISTING (all MoE-related)")
    P("=" * 80)
    hsa_dir = "/home/runner/aiter/hsa/gfx950/"
    if os.path.isdir(hsa_dir):
        for subdir in sorted(os.listdir(hsa_dir)):
            if 'moe' in subdir.lower() or 'fmoe' in subdir.lower() or 'flydsl' in subdir.lower():
                full = os.path.join(hsa_dir, subdir)
                if os.path.isdir(full):
                    files = sorted(os.listdir(full))
                    P(f"\n  {subdir}/ ({len(files)} files):")
                    for f in files[:20]:
                        sz = os.path.getsize(os.path.join(full, f))
                        P(f"    [{sz:>7}] {f}")
                    if len(files) > 20:
                        P(f"    ... ({len(files)} total files)")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 11: Search for "default" kernel logic in source
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[11] SEARCH: 'default' kernel fallback logic in fused_moe.py")
    P("=" * 80)
    try:
        fmoe_path = "/home/runner/aiter/aiter/fused_moe.py"
        with open(fmoe_path) as f:
            src_lines = f.readlines()
        P(f"  Total lines: {len(src_lines)}")
        # Find all lines with "default", "fallback", "not found", "empty"
        for i, line in enumerate(src_lines):
            ll = line.lower()
            if any(kw in ll for kw in ['default', 'fallback', 'not found',
                                        'empty', 'kernelname', 'kernel_name',
                                        '""', "''", 'no config', 'no entry']):
                P(f"  L{i+1}: {line.rstrip()}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 12: doweight_stage1 path analysis
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[12] doweight_stage1 ANALYSIS")
    P("=" * 80)
    try:
        src_all = open("/home/runner/aiter/aiter/fused_moe.py").read()
        for i, line in enumerate(src_all.split('\n')):
            if 'doweight' in line.lower():
                P(f"  L{i+1}: {line.rstrip()}")
    except Exception as e:
        P(f"  ERROR: {e}")

    # ──────────────────────────────────────────────────────────────────
    # SECTION 13: check if splitk is relevant for d=2048
    # ──────────────────────────────────────────────────────────────────
    P("\n" + "=" * 80)
    P("[13] splitk / ksplit analysis")
    P("=" * 80)
    try:
        for i, line in enumerate(src_all.split('\n')):
            if 'splitk' in line.lower() or 'ksplit' in line.lower() or 'split_k' in line.lower():
                P(f"  L{i+1}: {line.rstrip()}")
    except Exception as e:
        P(f"  ERROR: {e}")

    P("\n" + "=" * 80)
    P("=== D2048 DEEP PROBE COMPLETE ===")
    P("=" * 80 + "\n")


def _probe_runtime(data):
    """Called on EACH invocation to log the heuristic's selection at runtime."""
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]
    hidden_pad = d_hidden_pad - d_hidden
    intermediate_pad = d_expert_pad - d_expert

    # Get the EXACT params fused_moe would compute
    try:
        _, model_dim, inter_dim = fm.get_inter_dim(
            gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    except Exception as e:
        model_dim = d_hidden_pad
        inter_dim = d_expert_pad
        P(f"  get_inter_dim error: {e}")

    padded_M = fm.get_padded_M(M)
    est_m = M * topk // E

    P(f"\n{'─'*60}")
    P(f"[RUNTIME] M={M} E={E} topk={topk} d_hidden={d_hidden} d_expert={d_expert}")
    P(f"  d_hidden_pad={d_hidden_pad} d_expert_pad={d_expert_pad}")
    P(f"  hidden_pad={hidden_pad} intermediate_pad={intermediate_pad}")
    P(f"  model_dim={model_dim} inter_dim={inter_dim}")
    P(f"  padded_M={padded_M} est_m={est_m}")

    # What block_m does the heuristic select?
    try:
        block_m = fm.get_block_size_M(padded_M, topk, E, inter_dim)
        P(f"  get_block_size_M({padded_M}, {topk}, {E}, {inter_dim}) = {block_m}")
    except Exception as e:
        P(f"  get_block_size_M error: {e}")
        block_m = 64

    # What does use_nt return?
    try:
        nt = fm.use_nt(padded_M, topk, E)
        P(f"  use_nt({padded_M}, {topk}, {E}) = {nt}")
    except Exception as e:
        P(f"  use_nt error: {e}")

    # What does get_2stage_cfgs return?
    try:
        isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
        q_dtype_a = torch.float4_e2m1fn_x2
        q_dtype_w = gate_up_weight_shuffled.dtype

        P(f"  isG1U1={isG1U1}, q_dtype_a={q_dtype_a}, q_dtype_w={q_dtype_w}")

        metadata = fm.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
            isG1U1, ActivationType.Silu, False,
            hidden_pad, intermediate_pad, True)

        P(f"  RESULT: run_1stage={metadata.run_1stage}")
        P(f"  RESULT: block_m={metadata.block_m}")
        P(f"  RESULT: splitk={getattr(metadata, 'splitk', 'N/A')}")
        P(f"  RESULT: use_nt={getattr(metadata, 'use_nt', 'N/A')}")

        # Extract stage1 kernel info
        if not metadata.run_1stage:
            s1 = metadata.stage1
            if hasattr(s1, 'keywords'):
                P(f"  STAGE1 keywords:")
                for k, v in s1.keywords.items():
                    val_str = str(v)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    P(f"    {k} = {val_str}")
            if hasattr(s1, 'func'):
                P(f"  STAGE1 func: {s1.func}")
            if hasattr(s1, 'args'):
                P(f"  STAGE1 args: {s1.args}")

            s2 = metadata.stage2
            if hasattr(s2, 'keywords'):
                P(f"  STAGE2 keywords:")
                for k, v in s2.keywords.items():
                    val_str = str(v)
                    if len(val_str) > 100:
                        val_str = val_str[:100] + "..."
                    P(f"    {k} = {val_str}")
            if hasattr(s2, 'func'):
                P(f"  STAGE2 func: {s2.func}")
            if hasattr(s2, 'args'):
                P(f"  STAGE2 args: {s2.args}")

        # Also try doweight_stage1=True
        P(f"\n  --- Trying doweight_stage1=True ---")
        metadata_dw = fm.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
            isG1U1, ActivationType.Silu, True,
            hidden_pad, intermediate_pad, True)
        P(f"  doweight=True: run_1stage={metadata_dw.run_1stage}")
        P(f"  doweight=True: block_m={metadata_dw.block_m}")
        if not metadata_dw.run_1stage:
            if hasattr(metadata_dw.stage1, 'keywords'):
                kn = metadata_dw.stage1.keywords.get('kernelName', '')
                P(f"  doweight=True: stage1 kernel = {kn}")
            if hasattr(metadata_dw.stage2, 'keywords'):
                kn2 = metadata_dw.stage2.keywords.get('kernelName', '')
                P(f"  doweight=True: stage2 kernel = {kn2}")

        # Also try different padded_M values to see how heuristic changes
        P(f"\n  --- padded_M sweep ---")
        for test_padM in [16, 32, 64, 128, 256, 512, 1024, 2048]:
            try:
                meta_t = fm.get_2stage_cfgs(
                    test_padM, model_dim, inter_dim, E, topk,
                    torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
                    isG1U1, ActivationType.Silu, False,
                    hidden_pad, intermediate_pad, True)
                kn1 = ""
                kn2 = ""
                if not meta_t.run_1stage:
                    if hasattr(meta_t.stage1, 'keywords'):
                        kn1 = meta_t.stage1.keywords.get('kernelName', '')
                    if hasattr(meta_t.stage2, 'keywords'):
                        kn2 = meta_t.stage2.keywords.get('kernelName', '')
                P(f"  padded_M={test_padM:>5d}: run_1stage={meta_t.run_1stage} "
                  f"block_m={meta_t.block_m} "
                  f"s1={'[SET]' if kn1 else '[EMPTY]'} "
                  f"s2={'[SET]' if kn2 else '[EMPTY]'}")
                if kn1:
                    # Show just the tile part
                    tile_part = kn1.split('_')[4] if len(kn1.split('_')) > 4 else kn1
                    P(f"           s1_tile={tile_part}")
            except Exception as e:
                P(f"  padded_M={test_padM}: error={e}")

    except Exception as e:
        import traceback
        P(f"  get_2stage_cfgs error: {e}")
        traceback.print_exc(file=sys.stderr)

    # ──────────────────────────────────────────────────────────────────
    # Per-phase timing (only for d=2048 case)
    # ──────────────────────────────────────────────────────────────────
    if d_expert >= 2048:
        P(f"\n  --- d=2048 PER-PHASE TIMING (3 trials) ---")
        for trial in range(3):
            torch.cuda.synchronize()

            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)

            e0.record()
            out = fused_moe(
                hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                topk_weights, topk_ids,
                expert_mask=None, activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32, doweight_stage1=False,
                w1_scale=gate_up_weight_scale_shuffled,
                w2_scale=down_weight_scale_shuffled,
                a1_scale=None, a2_scale=None,
                hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
            )
            e1.record()
            torch.cuda.synchronize()

            t_us = e0.elapsed_time(e1) * 1000
            P(f"  Trial {trial}: {t_us:.1f} us")

    P(f"{'─'*60}")


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _call_count += 1

    # On first call, dump ALL static info
    if _call_count == 1:
        _probe_all()

    # Log runtime info for first 10 calls (each test case)
    if _call_count <= 10:
        _probe_runtime(data)

    # Run the actual kernel (unpatched, default behavior for correctness)
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

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

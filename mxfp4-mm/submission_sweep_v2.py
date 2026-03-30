#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
GEMM Config Sweep Runner v2 — Automated Triton Config Search
=============================================================
Two modes:
1. SUBMISSION MODE: Submit directly to popcorn with best known configs
2. SWEEP MODE (--sweep): Generate multiple submission files to test configs

K<=1024 → gemm_a16wfp4 (proven 6-7μs, don't touch)
K>1024  → gemm_afp4wfp4 (14-16μs bottleneck, sweep target)

Usage:
  popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm gemm-config-sweep-v2.py
  python gemm-config-sweep-v2.py --sweep --output-dir ./sweep_subs/
  python gemm-config-sweep-v2.py --sweep --shape 16,2112,7168 --mode coordesc
"""

import torch
import os
import sys
import json
from task import input_t, output_t

BENCHMARK_SHAPES = [
    (4, 2880, 512), (16, 2112, 7168), (32, 4096, 512),
    (32, 2880, 512), (64, 7168, 2048), (256, 3072, 1536),
]

# ============================================================================
# AFP4WFP4 CANDIDATE CONFIGS — Curated for each K>1024 shape
# ============================================================================
# MI355X: 304 CUs, 64KB LDS, MFMA 32x32x64 FP4
AFP4WFP4_CANDIDATES = {
    # M=16, N=2112, K=7168: Small M, big K → need small BM, large BK
    (16, 2112, 7168): [
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 2, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
         "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
         "waves_per_eu": 1, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 4, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 4, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 1, "num_warps": 2, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
    ],
    # M=64, N=7168, K=2048: Medium M, large N → leverage N parallelism
    (64, 7168, 2048): [
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 3,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 1, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 4, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 32},
    ],
    # M=256, N=3072, K=1536: Large M, good CU utilization
    (256, 3072, 1536): [
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 3,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
         "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 1, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 4, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
         "GROUP_SIZE_M": 2, "num_warps": 8, "num_stages": 2,
         "waves_per_eu": 1, "matrix_instr_nonkdim": 16},
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
         "GROUP_SIZE_M": 4, "num_warps": 4, "num_stages": 2,
         "waves_per_eu": 2, "matrix_instr_nonkdim": 32},
    ],
}

# ============================================================================
# BEST KNOWN CONFIGS — Updated as sweep finds improvements
# ============================================================================
AFP4WFP4_BEST = {
    (16, 2112, 7168): {
        "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    },
    (64, 7168, 2048): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    },
    (256, 3072, 1536): {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 256,
        "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
    },
}

# A16WFP4 CONFIGS — proven for K<=1024, don't change
A16WFP4_CONFIGS = {
    (4, 2880, 512): {
        "BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
        "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": "",
    },
    (32, 4096, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
        "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": "",
    },
    (32, 2880, 512): {
        "BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 512,
        "GROUP_SIZE_M": 4, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024,
        "num_warps": 4, "num_stages": 2,
        "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": "",
    },
}

# ============================================================================
# GLOBALS
# ============================================================================
_warmed = False
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None


def _unshuffle_e8m0(scale_sh):
    """Reverse e8m0_shuffle: permute within (32, 8) blocks."""
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


# ============================================================================
# CONFIG FILE OVERRIDE — Write custom configs to aiter's JSON config dir
# ============================================================================
def _write_afp4wfp4_config(N, K, config):
    """Write custom AFP4WFP4 config to aiter's config directory."""
    config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    os.makedirs(config_dir, exist_ok=True)
    
    fname = f"gfx950-GEMM-AFP4WFP4-N={N}-K={K}.json"
    triton_cfg = {
        "kwargs": {
            "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
            "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
            "GROUP_SIZE_M": config["GROUP_SIZE_M"],
        },
        "num_warps": config["num_warps"],
        "num_stages": config["num_stages"],
    }
    if "waves_per_eu" in config:
        triton_cfg["waves_per_eu"] = config["waves_per_eu"]
    if "matrix_instr_nonkdim" in config:
        triton_cfg["matrix_instr_nonkdim"] = config["matrix_instr_nonkdim"]
    
    with open(os.path.join(config_dir, fname), 'w') as f:
        json.dump([triton_cfg], f, indent=2)


def _inject_configs(afp4_configs):
    """Write all AFP4WFP4 configs and clear any caches."""
    for (M, N, K), cfg in afp4_configs.items():
        if K > 1024:
            _write_afp4wfp4_config(N, K, cfg)
    # Clear cached config lookups
    try:
        from aiter.ops.triton.gemm.basic import gemm_afp4wfp4 as mod
        if hasattr(mod, '_get_config') and hasattr(mod._get_config, 'cache_clear'):
            mod._get_config.cache_clear()
    except Exception:
        pass


# ============================================================================
# PRE-WARMING — Compile all kernels before benchmark timing starts
# ============================================================================
def _prewarm(B_q, B_shuffle, B_scale_sh, afp4_configs=None):
    global _warmed
    if _warmed:
        return
    _warmed = True
    
    if afp4_configs is None:
        afp4_configs = AFP4WFP4_BEST
    
    # Write config files FIRST
    _inject_configs(afp4_configs)
    
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    device = B_q.device
    
    for M, N, K in BENCHMARK_SHAPES:
        try:
            A_w = torch.randn(M, K, dtype=torch.bfloat16, device=device)
            if K <= 1024:
                bq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
                bs = torch.randint(100, 140, (N, K // 32), dtype=torch.uint8, device=device)
                cfg = A16WFP4_CONFIGS.get((M, N, K))
                gemm_a16wfp4(A_w, bq, bs, dtype=torch.bfloat16, config=cfg)
            else:
                A_q, A_s = dynamic_mxfp4_quant(A_w)
                bq = torch.randint(0, 255, (N, K // 2), dtype=torch.uint8, device=device)
                bs = torch.randint(100, 140, (N, K // 32), dtype=torch.uint8, device=device)
                # Config was already written to file; afp4wfp4 reads from file
                gemm_afp4wfp4(A_q.view(torch.uint8), bq, A_s, bs, dtype=torch.bfloat16)
        except Exception:
            pass
    
    torch.cuda.synchronize()


# ============================================================================
# SUBMISSION ENTRY POINT
# ============================================================================
def custom_kernel(data: input_t) -> output_t:
    """GEMM: C = A @ B^T, B pre-quantized to MXFP4."""
    global _bscale_ref, _bscale_raw, _bq_u8
    
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    
    # Cache B-side (doesn't change within test case)
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
    
    if not _warmed:
        _prewarm(B_q, B_shuffle, B_scale_sh)
    
    if K <= 1024:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        cfg = A16WFP4_CONFIGS.get((M, N, K))
        return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)
    else:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.utility.fp4_utils import dynamic_mxfp4_quant
        A_q, A_scale = dynamic_mxfp4_quant(A)
        # Config was written to file during prewarm; kernel reads from file
        return gemm_afp4wfp4(A_q.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)


# ============================================================================
# SWEEP MODE — Generate submission files for automated config testing
# ============================================================================
def _cfg_label(cfg):
    """Short label for a config dict."""
    abbr = {"BLOCK_SIZE_M": "bm", "BLOCK_SIZE_N": "bn", "BLOCK_SIZE_K": "bk",
            "GROUP_SIZE_M": "gm", "num_warps": "w", "num_stages": "s",
            "waves_per_eu": "wpe", "matrix_instr_nonkdim": "mi"}
    return "_".join(f"{v}{cfg[k]}" for k, v in abbr.items() if k in cfg)


def _generate_submission(shape, config, base_configs, output_dir, idx):
    """Generate a standalone .py submission testing one AFP4WFP4 config."""
    M, N, K = shape
    label = _cfg_label(config)
    fname = f"sweep_M{M}_N{N}_K{K}_{label}_{idx:03d}.py"
    filepath = os.path.join(output_dir, fname)
    
    # Build test config map: this shape gets candidate, others keep best
    test_cfgs = dict(base_configs)
    test_cfgs[(M, N, K)] = config
    
    # Serialize configs for embedding in generated code
    afp4_json = json.dumps({f"{m},{n},{k}": v for (m, n, k), v in test_cfgs.items()}, indent=2)
    a16_json = json.dumps({f"{m},{n},{k}": v for (m, n, k), v in A16WFP4_CONFIGS.items()}, indent=2)
    
    code = f"""#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
# AUTO-SWEEP: M={M}, N={N}, K={K} | {label}
# Config: {json.dumps(config)}
import torch, os, json
from task import input_t, output_t

SHAPES = {BENCHMARK_SHAPES!r}
AFP4_CFGS = {afp4_json}
A16_CFGS = {a16_json}
_warmed = False
_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def _write_cfg(N, K, cfg):
    d = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
    os.makedirs(d, exist_ok=True)
    tc = {{"kwargs": {{k: cfg[k] for k in ["BLOCK_SIZE_M","BLOCK_SIZE_N","BLOCK_SIZE_K","GROUP_SIZE_M"]}},
          "num_warps": cfg["num_warps"], "num_stages": cfg["num_stages"]}}
    if "waves_per_eu" in cfg: tc["waves_per_eu"] = cfg["waves_per_eu"]
    if "matrix_instr_nonkdim" in cfg: tc["matrix_instr_nonkdim"] = cfg["matrix_instr_nonkdim"]
    with open(os.path.join(d, f"gfx950-GEMM-AFP4WFP4-N={{N}}-K={{K}}.json"), "w") as f:
        json.dump([tc], f)

def _prewarm(B_q, B_shuffle, B_scale_sh):
    global _warmed
    if _warmed: return
    _warmed = True
    for key, cfg in AFP4_CFGS.items():
        m, n, k = [int(x) for x in key.split(",")]
        if k > 1024: _write_cfg(n, k, cfg)
    try:
        from aiter.ops.triton.gemm.basic import gemm_afp4wfp4 as mod
        if hasattr(mod, "_get_config") and hasattr(mod._get_config, "cache_clear"):
            mod._get_config.cache_clear()
    except: pass
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant
    dev = B_q.device
    for M, N, K in SHAPES:
        try:
            A_w = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
            bq = torch.randint(0, 255, (N, K//2), dtype=torch.uint8, device=dev)
            bs = torch.randint(100, 140, (N, K//32), dtype=torch.uint8, device=dev)
            if K <= 1024:
                key = f"{{M}},{{N}},{{K}}"
                cfg = A16_CFGS.get(key)
                gemm_a16wfp4(A_w, bq, bs, dtype=torch.bfloat16, config=cfg)
            else:
                A_q, A_s = dynamic_mxfp4_quant(A_w)
                gemm_afp4wfp4(A_q.view(torch.uint8), bq, A_s, bs, dtype=torch.bfloat16)
        except: pass
    torch.cuda.synchronize()

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
    if not _warmed:
        _prewarm(B_q, B_shuffle, B_scale_sh)
    if K <= 1024:
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
        key = f"{{M}},{{N}},{{K}}"
        cfg = A16_CFGS.get(key)
        return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)
    else:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.utility.fp4_utils import dynamic_mxfp4_quant
        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
"""
    
    with open(filepath, 'w') as f:
        f.write(code)
    return filepath


def _coordesc_neighbors(base_config):
    """Generate coordinate descent neighbors for a config."""
    param_ranges = {
        "BLOCK_SIZE_M": [4, 8, 16, 32, 64, 128],
        "BLOCK_SIZE_N": [32, 64, 128, 256],
        "BLOCK_SIZE_K": [64, 128, 256, 512],
        "GROUP_SIZE_M": [1, 2, 4, 8],
        "num_warps": [2, 4, 8],
        "num_stages": [1, 2, 3],
        "waves_per_eu": [1, 2, 3, 4],
        "matrix_instr_nonkdim": [16, 32],
    }
    configs = []
    for param, values in param_ranges.items():
        cur = base_config.get(param)
        if cur is None:
            continue
        try:
            idx = values.index(cur)
        except ValueError:
            continue
        for delta in [-1, 1]:
            ni = idx + delta
            if 0 <= ni < len(values):
                new_cfg = dict(base_config)
                new_cfg[param] = values[ni]
                if new_cfg != base_config:
                    configs.append(new_cfg)
    return configs


def _grid_configs(shape):
    """Generate a focused grid of configs for a shape."""
    M, N, K = shape
    if M <= 16:
        bm_range = [4, 8, 16]
    elif M <= 64:
        bm_range = [16, 32, 64]
    else:
        bm_range = [32, 64, 128]
    
    if N <= 2112:
        bn_range = [32, 64, 128]
    elif N <= 3072:
        bn_range = [64, 128, 256]
    else:
        bn_range = [128, 256]
    
    bk_range = [128, 256, 512]
    fixed_sets = [
        {"GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
        {"GROUP_SIZE_M": 2, "num_warps": 4, "num_stages": 3, "waves_per_eu": 2, "matrix_instr_nonkdim": 16},
    ]
    
    configs = []
    for bm in bm_range:
        for bn in bn_range:
            for bk in bk_range:
                lds = (bm * bk + bn * bk) // 2 + (bm + bn) * (bk // 32)
                if lds > 65536:
                    continue
                n_tiles = ((M + bm - 1) // bm) * ((N + bn - 1) // bn)
                if n_tiles < 4:
                    continue
                for f in fixed_sets:
                    configs.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, **f})
    return configs


def sweep_main():
    """CLI for generating sweep submission files."""
    import argparse
    
    p = argparse.ArgumentParser(description="GEMM AFP4WFP4 Config Sweep Generator")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--shape", type=str, help="M,N,K")
    p.add_argument("--output-dir", type=str, default="./sweep_submissions")
    p.add_argument("--mode", choices=["candidates", "grid", "coordesc", "all"], default="candidates")
    p.add_argument("--base-config", type=str, help="Base config JSON for coordesc")
    p.add_argument("--max-configs", type=int, default=50)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    
    if not args.sweep:
        print("Usage: python gemm-config-sweep-v2.py --sweep [options]")
        print("Or submit directly to popcorn.")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.shape:
        parts = [int(x) for x in args.shape.split(",")]
        targets = [(parts[0], parts[1], parts[2])]
    else:
        targets = [(M, N, K) for M, N, K in BENCHMARK_SHAPES if K > 1024]
    
    total = 0
    all_files = []
    
    for shape in targets:
        M, N, K = shape
        print(f"\n{'='*60}")
        print(f"Shape: M={M}, N={N}, K={K}")
        print(f"{'='*60}")
        
        configs = []
        if args.mode in ("candidates", "all"):
            c = AFP4WFP4_CANDIDATES.get(shape, [])
            configs.extend(c)
            print(f"  Curated: {len(c)}")
        
        if args.mode in ("grid", "all"):
            g = _grid_configs(shape)
            configs.extend(g)
            print(f"  Grid: {len(g)}")
        
        if args.mode in ("coordesc", "all"):
            base = json.loads(args.base_config) if args.base_config else AFP4WFP4_BEST.get(shape, {})
            n = _coordesc_neighbors(base)
            configs.extend(n)
            print(f"  Coordesc: {len(n)}")
        
        # Deduplicate
        seen = set()
        unique = []
        for cfg in configs:
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                unique.append(cfg)
        unique = unique[:args.max_configs]
        print(f"  Unique: {len(unique)}")
        
        if args.dry_run:
            for i, cfg in enumerate(unique):
                print(f"  [{i+1}] {_cfg_label(cfg)}")
            continue
        
        for i, cfg in enumerate(unique):
            fp = _generate_submission(shape, cfg, AFP4WFP4_BEST, args.output_dir, i)
            all_files.append(fp)
            total += 1
            print(f"  [{i+1}/{len(unique)}] {os.path.basename(fp)}")
    
    if not args.dry_run and all_files:
        # Generate runner script
        runner = os.path.join(args.output_dir, "run_sweep.sh")
        with open(runner, 'w') as f:
            f.write("#!/bin/bash\n# Auto-generated sweep runner\n")
            f.write(f"# {len(all_files)} submissions\n\n")
            for fp in all_files:
                bn = os.path.basename(fp)
                f.write(f'echo "=== {bn} ==="\n')
                f.write(f'popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark {fp} --no-tui 2>&1 | tee -a sweep_log.txt\n')
                f.write('sleep 5\n\n')
            f.write('echo "Done! Parse with: python parse_results.py sweep_log.txt"\n')
        os.chmod(runner, 0o755)
        
        # Generate results parser
        parser_file = os.path.join(args.output_dir, "parse_results.py")
        with open(parser_file, 'w') as f:
            f.write("""#!/usr/bin/env python3
import re, sys
from collections import defaultdict

def parse(logfile):
    results = []
    cur_file = cur_k = cur_m = cur_n = None
    with open(logfile) as f:
        for line in f:
            m = re.match(r"=== (.+) ===", line)
            if m: cur_file = m.group(1); continue
            km = re.match(r".*k:\s*(\d+);\s*m:\s*(\d+);\s*n:\s*(\d+)", line)
            if km: cur_k, cur_m, cur_n = int(km.group(1)), int(km.group(2)), int(km.group(3)); continue
            tm = re.search(r"time\s+([\d.]+)\s+", line)
            if tm and cur_file:
                results.append({"file": cur_file, "M": cur_m, "N": cur_n, "K": cur_k, "us": float(tm.group(1))})
    
    by_file = defaultdict(list)
    for r in results: by_file[r["file"]].append(r["us"])
    
    ranked = []
    for fn, times in by_file.items():
        if len(times) >= 6:
            gm = 1.0
            for t in times: gm *= t
            ranked.append((gm ** (1.0/len(times)), fn, times))
    ranked.sort()
    
    print("\n" + "="*70)
    print(f"SWEEP RESULTS ({len(ranked)} complete submissions)")
    print("="*70)
    for i, (gm, fn, times) in enumerate(ranked[:20]):
        print(f"  #{i+1}: {gm:.2f}us geomean - {fn}")
        print(f"       times: {[f'{t:.1f}' for t in times]}")
    if ranked:
        print(f"\nBEST: {ranked[0][1]} -> {ranked[0][0]:.2f}us")

if __name__ == "__main__":
    parse(sys.argv[1] if len(sys.argv) > 1 else "sweep_log.txt")
""")
        os.chmod(parser_file, 0o755)
        
        print(f"\n{'='*60}")
        print(f"Generated {total} submissions in {args.output_dir}/")
        print(f"Runner: {runner}")
        print(f"Parser: {parser_file}")
        print(f"Est. time: ~{total * 6 / 60:.1f}h at 6min/submission")
        print(f"{'='*60}")


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep_main()
    else:
        sweep_main()

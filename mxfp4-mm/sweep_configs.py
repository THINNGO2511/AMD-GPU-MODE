#!/usr/bin/env python3
"""
Config sweeper for MXFP4 GEMM.

Generates submission files with different hardcoded configs,
submits to popcorn benchmark, parses results, tracks best.

Usage: python sweep_configs.py
"""
import subprocess
import re
import os
import json
import time
import itertools

SUBMISSION_TEMPLATE = '''#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Auto-generated config sweep: {config_name}"""
from task import input_t, output_t
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None

def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

@triton.jit
def _fused_quant_gemm(
    a_ptr, b_ptr, c_ptr, b_scales_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn,
    stride_cm, stride_cn, stride_bsn, stride_bsk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    num_k_iters = tl.cdiv(K, BLOCK_K)
    for k_iter in range(num_k_iters):
        k_start = k_iter * BLOCK_K
        a_tile = tl.load(a_ptr + offs_m[:, None] * stride_am + (k_start + tl.arange(0, BLOCK_K))[None, :] * stride_ak).to(tl.float32)
        a_fp4, a_scales = _mxfp4_quant_op(a_tile, BLOCK_K, BLOCK_M, SCALE_GROUP)
        b_fp4 = tl.load(b_ptr + (k_start // 2 + tl.arange(0, BLOCK_K // 2))[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        b_scales = tl.load(b_scales_ptr + offs_n[:, None] * stride_bsn + (k_start // SCALE_GROUP + tl.arange(0, BLOCK_K // SCALE_GROUP))[None, :] * stride_bsk)
        acc = tl.dot_scaled(a_fp4, a_scales, "e2m1", b_fp4, b_scales, "e2m1", acc)
    c = acc.to(tl.bfloat16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N).to(tl.int64)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, c, mask=c_mask)

# Hardcoded configs per (M, K) — this is what we're sweeping
FUSED_CONFIGS = {fused_configs}

def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]
    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)
    if k <= 1024:
        C = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')
        cfg = FUSED_CONFIGS.get((m, k), {fused_default})
        grid = (triton.cdiv(m, cfg[0]) * triton.cdiv(n, cfg[1]),)
        _fused_quant_gemm[grid](
            A, _bq_u8, C, _bscale_raw, m, n, k,
            A.stride(0), A.stride(1), _bq_u8.stride(1), _bq_u8.stride(0),
            C.stride(0), C.stride(1), _bscale_raw.stride(0), _bscale_raw.stride(1),
            BLOCK_M=cfg[0], BLOCK_N=cfg[1], BLOCK_K=cfg[2], GROUP_SIZE_M=cfg[3],
            num_warps=cfg[4], num_stages=cfg[5],
        )
        return C
    else:
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4, _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
'''

# Configs to sweep: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M, num_warps, num_stages)
FUSED_CONFIG_OPTIONS = [
    (32, 64,  128, 4, 2, 1),
    (32, 64,  128, 4, 4, 2),
    (32, 128, 128, 4, 4, 2),
    (32, 128, 128, 4, 4, 0),
    (32, 128, 128, 4, 8, 2),
    (32, 128, 128, 8, 4, 2),
    (32, 128, 128, 1, 4, 2),
    (32, 256, 128, 4, 4, 2),
    (32, 256, 128, 4, 8, 2),
    (64, 64,  128, 4, 4, 2),
    (64, 128, 128, 4, 4, 2),
    (64, 128, 128, 4, 8, 2),
    (64, 256, 128, 4, 4, 2),
    (64, 128, 128, 4, 4, 0),
    (32, 64,  128, 4, 4, 0),
    (32, 128, 128, 2, 4, 2),
]

POPCORN = os.path.expanduser("~/.local/bin/popcorn-cli")
SUBMIT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(SUBMIT_DIR, "sweep_results.json")


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def generate_submission(config_idx, config):
    """Generate a submission file with a specific hardcoded config."""
    bm, bn, bk, gsm, nw, ns = config
    config_name = f"BM{bm}_BN{bn}_BK{bk}_GSM{gsm}_W{nw}_S{ns}"

    # Build the fused configs dict for all K=512 problem sizes
    fused_configs = "{\n"
    for m in [4, 32, 256]:  # M values that use K=512
        fused_configs += f"        ({m}, 512): ({bm}, {bn}, {bk}, {gsm}, {nw}, {ns}),\n"
    fused_configs += "    }"

    fused_default = f"({bm}, {bn}, {bk}, {gsm}, {nw}, {ns})"

    content = SUBMISSION_TEMPLATE.format(
        config_name=config_name,
        fused_configs=fused_configs,
        fused_default=fused_default,
    )

    filepath = os.path.join(SUBMIT_DIR, f"sweep_{config_name}.py")
    with open(filepath, 'w') as f:
        f.write(content)

    return filepath, config_name


def submit_and_parse(filepath):
    """Submit to benchmark and parse results."""
    cmd = [POPCORN, "submit", "--gpu", "MI355X", "--leaderboard", "amd-mxfp4-mm",
           "--mode", "benchmark", filepath, "--no-tui"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return None

    # Parse benchmark results
    times = {}
    pattern = r'k: (\d+); m: (\d+); n: (\d+).*?\n\s*⏱\s*([\d.]+)\s*±'
    for match in re.finditer(pattern, output):
        k, m, n, t = int(match.group(1)), int(match.group(2)), int(match.group(3)), float(match.group(4))
        times[(m, n, k)] = t

    return times


def compute_geomean(times):
    if not times:
        return float('inf')
    product = 1.0
    for t in times.values():
        product *= t
    return product ** (1.0 / len(times))


def main():
    results = load_results()
    best_geomean = float('inf')
    best_config = None

    # Check existing best
    for name, data in results.items():
        gm = data.get('geomean', float('inf'))
        if gm < best_geomean:
            best_geomean = gm
            best_config = name

    print(f"Current best: {best_config} at {best_geomean:.2f}μs geomean")
    print(f"Configs to sweep: {len(FUSED_CONFIG_OPTIONS)}")
    print()

    for i, config in enumerate(FUSED_CONFIG_OPTIONS):
        config_name = f"BM{config[0]}_BN{config[1]}_BK{config[2]}_GSM{config[3]}_W{config[4]}_S{config[5]}"

        if config_name in results:
            print(f"[{i+1}/{len(FUSED_CONFIG_OPTIONS)}] {config_name}: already tested (geomean={results[config_name].get('geomean', '?')}μs)")
            continue

        print(f"[{i+1}/{len(FUSED_CONFIG_OPTIONS)}] Testing {config_name}...")
        filepath, name = generate_submission(i, config)

        times = submit_and_parse(filepath)
        if times:
            gm = compute_geomean(times)
            results[name] = {'config': list(config), 'times': {str(k): v for k, v in times.items()}, 'geomean': gm}
            save_results(results)

            marker = " *** NEW BEST ***" if gm < best_geomean else ""
            print(f"  Geomean: {gm:.2f}μs{marker}")
            for (m, n, k), t in sorted(times.items()):
                print(f"    M={m}, N={n}, K={k}: {t:.2f}μs")

            if gm < best_geomean:
                best_geomean = gm
                best_config = name
        else:
            print(f"  FAILED to parse results")
            results[name] = {'config': list(config), 'error': True}
            save_results(results)

        # Cleanup generated file
        os.remove(filepath)

    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best_config} at {best_geomean:.2f}μs geomean")
    if best_config in results:
        cfg = results[best_config]
        print(f"Config: {cfg.get('config', '?')}")
        if 'times' in cfg:
            for k, v in sorted(cfg['times'].items()):
                print(f"  {k}: {v}μs")


if __name__ == "__main__":
    main()

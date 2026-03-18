#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
Deep probe: read aiter quant source, try different kernel configs via CSV,
and time individual operations precisely.
"""
from task import input_t, output_t
import torch
import aiter
from aiter import dtypes
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
import csv
import os

_injected = False

def _inject_and_probe():
    global _injected
    if _injected:
        return
    _injected = True

    # Read the full tuned CSV to find entries with splitK > 0
    csv_path = "/home/runner/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv"
    splitk_entries = []
    all_m_values = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_m_values.add(int(row['M']))
            if int(row['splitK']) > 0:
                splitk_entries.append(row)

    print(f"All M values in tuned CSV: {sorted(all_m_values)}")
    print(f"Entries with splitK > 0: {len(splitk_entries)}")
    for e in splitk_entries[:5]:
        print(f"  M={e['M']}, N={e['N']}, K={e['K']}, splitK={e['splitK']}, kernel={e['kernelName']}")

    # Read gemm_op_a4w4.py source to understand config lookup
    src_path = "/home/runner/aiter/aiter/ops/gemm_op_a4w4.py"
    if os.path.exists(src_path):
        with open(src_path) as f:
            src = f.read()
        # Print key functions
        lines = src.splitlines()
        print(f"\ngemm_op_a4w4.py has {len(lines)} lines")
        # Find get_GEMM_config and gemm_a4w4 functions
        for i, line in enumerate(lines):
            if 'def get_GEMM_config' in line or 'def gemm_a4w4' in line or 'def gemm_a4w4_blockscale' in line:
                # Print function and next 30 lines
                print(f"\n--- Line {i+1}: {line.strip()} ---")
                for j in range(i, min(i+35, len(lines))):
                    print(f"  {j+1}: {lines[j]}")

    # Read the quant source
    quant_path = "/home/runner/aiter/aiter/ops/triton/quant.py"
    if os.path.exists(quant_path):
        with open(quant_path) as f:
            quant_src = f.read()
        lines = quant_src.splitlines()
        print(f"\nquant.py has {len(lines)} lines")
        for i, line in enumerate(lines):
            if 'def dynamic_mxfp4_quant' in line:
                print(f"\n--- Line {i+1}: {line.strip()} ---")
                for j in range(i, min(i+25, len(lines))):
                    print(f"  {j+1}: {lines[j]}")

    # Read e8m0_shuffle source
    fp4_path = "/home/runner/aiter/aiter/utility/fp4_utils.py"
    if os.path.exists(fp4_path):
        with open(fp4_path) as f:
            fp4_src = f.read()
        lines = fp4_src.splitlines()
        print(f"\nfp4_utils.py has {len(lines)} lines")
        for i, line in enumerate(lines):
            if 'def e8m0_shuffle' in line:
                print(f"\n--- Line {i+1}: {line.strip()} ---")
                for j in range(i, min(i+30, len(lines))):
                    print(f"  {j+1}: {lines[j]}")

    # Inject configs with different kernels for our benchmark sizes
    # Try 32x256 for small M (more N-parallelism) and splitK for large K
    new_lines = []
    existing_keys = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_keys.add((int(row['M']), int(row['N']), int(row['K'])))

    benchmark_configs = [
        # (M, N, K, kernelId, splitK, kernelName)
        # Try 32x256 for N=2880 (2880/256=11.25, so 12 blocks vs 2880/128=22.5, 23 blocks)
        (4, 2880, 512, 23, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E"),
        (16, 2112, 7168, 21, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"),
        (32, 4096, 512, 21, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"),
        (32, 2880, 512, 23, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E"),
        (64, 7168, 2048, 29, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"),
        # Test sizes too
        (8, 2112, 7168, 21, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"),
        (16, 3072, 1536, 21, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128E"),
        (64, 3072, 1536, 29, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128E"),
        (256, 2880, 512, 23, 0, "_ZN5aiter41f4gemm_bf16_per1x32Fp4_BpreShuffle_32x256E"),
    ]

    for m, n, k, kid, sk, kname in benchmark_configs:
        if (m, n, k) not in existing_keys:
            new_lines.append(f"256,{m},{n},{k},{kid},{sk},5.0,{kname},1.0,100.0,0.0")

    if new_lines:
        with open(csv_path, 'a') as f:
            for line in new_lines:
                f.write(line + '\n')

    from aiter.ops.gemm_op_a4w4 import get_GEMM_config
    if hasattr(get_GEMM_config, 'cache_clear'):
        get_GEMM_config.cache_clear()


def custom_kernel(data: input_t) -> output_t:
    _inject_and_probe()

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape

    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    return aiter.gemm_a4w4(
        A_q, B_shuffle, A_scale_sh, B_scale_sh,
        dtype=dtypes.bf16, bpreshuffle=True,
    )

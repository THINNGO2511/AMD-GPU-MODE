#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Find where the tuned GEMM configs are stored."""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch, os, glob
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Find all CSV files in aiter configs
    config_dirs = [
        "/home/runner/aiter/aiter/configs",
        "/home/runner/aiter/aiter/configs/model_configs",
    ]
    for d in config_dirs:
        if os.path.exists(d):
            files = sorted(os.listdir(d))
            print(f"{d}: {files}")

    # Find the tuned f4gemm config
    for pattern in ["/home/runner/aiter/**/*f4gemm*", "/home/runner/aiter/**/*tuned*gemm*"]:
        matches = glob.glob(pattern, recursive=True)
        for m_path in sorted(matches)[:10]:
            print(f"Found: {m_path}")
            if m_path.endswith('.csv') and 'f4gemm' in m_path.lower():
                with open(m_path) as f:
                    lines = f.readlines()
                print(f"  Lines: {len(lines)}, Header: {lines[0].strip()}")
                # Show a few data lines
                for line in lines[1:4]:
                    print(f"  {line.strip()}")

    # Check get_GEMM_config source
    from aiter.ops.gemm_op_a4w4 import get_GEMM_config
    import inspect
    src_file = inspect.getfile(get_GEMM_config)
    print(f"get_GEMM_config defined in: {src_file}")

    # Standard baseline
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)

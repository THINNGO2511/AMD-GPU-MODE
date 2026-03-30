#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Read the a4w4_blockscale tuned GEMM config CSV."""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Read tuned and untuned configs
    for csv_name in ['a4w4_blockscale_tuned_gemm.csv', 'a4w4_blockscale_untuned_gemm.csv']:
        path = f"/home/runner/aiter/aiter/configs/{csv_name}"
        try:
            with open(path) as f:
                content = f.read()
            lines = content.strip().splitlines()
            print(f"\n=== {csv_name} ({len(lines)} lines) ===")
            for line in lines[:30]:  # Show up to 30 lines
                print(line)
            if len(lines) > 30:
                print(f"... and {len(lines)-30} more lines")
        except Exception as e:
            print(f"{csv_name}: {e}")

    # Also check the gemm_op_a4w4 module for config loading logic
    try:
        import aiter.ops.gemm_op_a4w4 as mod
        # Print the source of get_GEMM_config if possible
        print(f"\nget_GEMM_config: {type(mod.get_GEMM_config)}")
        # Check if it has cache info
        if hasattr(mod.get_GEMM_config, 'cache_info'):
            print(f"Cache info: {mod.get_GEMM_config.cache_info()}")
    except Exception as e:
        print(f"Module inspect error: {e}")

    # Standard baseline
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
    return aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)

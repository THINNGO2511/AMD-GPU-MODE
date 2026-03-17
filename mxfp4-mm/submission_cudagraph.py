#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Use CUDA graphs to reduce kernel launch overhead.

Profile shows 3 separate kernel launches: quant (~15us), shuffle (~18us), GEMM (~14us).
CUDA graphs can capture the full pipeline and replay with minimal launch overhead.
"""
from task import input_t, output_t

# Cache for CUDA graphs per (m,n,k) shape
_graph_cache = {}


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]
    key = (m, n, k)

    if key not in _graph_cache:
        # Warmup run (required before graph capture)
        A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
        A_q = A_fp4.view(dtypes.fp4x2)
        A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)
        out = aiter.gemm_a4w4(A_q, B_shuffle, A_scale_sh, B_scale_sh, dtype=dtypes.bf16, bpreshuffle=True)
        torch.cuda.synchronize()

        # Create static input/output tensors for graph
        static_A = A.clone()
        static_B_shuffle = B_shuffle.clone()
        static_B_scale = B_scale_sh.clone()

        # Capture graph
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            s_fp4, s_scale = dynamic_mxfp4_quant(static_A)
            s_q = s_fp4.view(dtypes.fp4x2)
            s_scale_sh = e8m0_shuffle(s_scale).view(dtypes.fp8_e8m0)
            static_out = aiter.gemm_a4w4(s_q, static_B_shuffle, s_scale_sh, static_B_scale, dtype=dtypes.bf16, bpreshuffle=True)

        _graph_cache[key] = (g, static_A, static_B_shuffle, static_B_scale, static_out)

    g, static_A, static_B_shuffle, static_B_scale, static_out = _graph_cache[key]

    # Copy new data into static tensors
    static_A.copy_(A)
    static_B_shuffle.copy_(B_shuffle)
    static_B_scale.copy_(B_scale_sh)

    # Replay graph
    g.replay()

    return static_out.clone()

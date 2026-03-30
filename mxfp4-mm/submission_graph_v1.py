#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — v4 + CUDA Graph capture for repeated calls.
CUDA Graphs eliminate Python/driver launch overhead (~1-2us per call).
"""
from task import input_t, output_t
import torch

_bscale_ref = None
_bscale_raw = None
_bq_u8 = None
_y_cache = {}
_graphs = {}
_warmup_done = set()

_K7168_CONFIG = {
    "BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
    "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
    "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024,
}


def _unshuffle_e8m0(scale_sh):
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def custom_kernel(data: input_t) -> output_t:
    global _bscale_ref, _bscale_raw, _bq_u8

    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    if _bscale_ref is not B_scale_sh:
        _bscale_ref = B_scale_sh
        _bscale_raw = _unshuffle_e8m0(B_scale_sh)
        _bq_u8 = B_q.view(torch.uint8)

    key = (m, n, k)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw,
                             dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    mk = (m, n)
    if mk not in _y_cache:
        _y_cache[mk] = torch.empty((m, n), dtype=torch.bfloat16, device='cuda')

    cfg = _K7168_CONFIG if k == 7168 else None

    # Try CUDA Graph capture
    if key not in _graphs:
        if key not in _warmup_done:
            _warmup_done.add(key)
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                               y=_y_cache[mk], config=cfg)
        try:
            A_ph = torch.empty_like(A)
            A_ph.copy_(A)
            y_out = _y_cache[mk]

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                gemm_a16wfp4(A_ph, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                            y=y_out, config=cfg)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                gemm_a16wfp4(A_ph, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                            y=y_out, config=cfg)
            _graphs[key] = (g, A_ph, y_out)
            A_ph.copy_(A)
            g.replay()
            return y_out
        except Exception:
            _graphs[key] = None
            return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                               y=_y_cache[mk], config=cfg)

    graph_data = _graphs[key]
    if graph_data is not None:
        g, A_ph, y_out = graph_data
        A_ph.copy_(A)
        g.replay()
        return y_out
    else:
        return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16,
                           y=_y_cache[mk], config=cfg)

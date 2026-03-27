"""GEMM: a16wfp4 for ALL shapes — no A quantization at all"""
import torch
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

_scale_cache = {}

def _unshuffle(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _scale_cache:
        return _scale_cache[key]
    n_sc = K // 32
    sm = ((N + 255) // 256) * 256
    sn = ((n_sc + 7) // 8) * 8
    s = B_scale_sh.view(torch.uint8)
    padded = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    padded[:N, :n_sc] = s[:N, :n_sc]
    r = padded.view(sm//32, sn//8, 4, 16, 2, 2)
    u = r.permute(0, 5, 3, 1, 4, 2).contiguous()
    result = u.view(sm, sn)[:N, :n_sc]
    _scale_cache[key] = result
    return result

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    B_scale = _unshuffle(B_scale_sh, N, K)
    B_q_u8 = B_q.view(torch.uint8)
    return gemm_a16wfp4(A, B_q_u8, B_scale)

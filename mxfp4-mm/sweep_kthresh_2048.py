import torch
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

_sc = {}
def _unshuffle(B_scale_sh, N, K):
    key = id(B_scale_sh)
    if key in _sc: return _sc[key]
    n_sc = K // 32
    sm = ((N+255)//256)*256; sn = ((n_sc+7)//8)*8
    s = B_scale_sh.view(torch.uint8)
    p = torch.zeros(sm, sn, dtype=torch.uint8, device=s.device)
    p[:N,:n_sc] = s[:N,:n_sc]
    r = p.view(sm//32, sn//8, 4, 16, 2, 2)
    u = r.permute(0,5,3,1,4,2).contiguous()
    result = u.view(sm, sn)[:N,:n_sc]
    _sc[key] = result
    return result

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape; N = B_q.shape[0]
    if K <= 2048:
        B_scale = _unshuffle(B_scale_sh, N, K)
        return gemm_a16wfp4(A, B_q.view(torch.uint8), B_scale)
    else:
        A_q, A_s = dynamic_mxfp4_quant(A)
        A_ss = e8m0_shuffle(A_s)
        return gemm_afp4wfp4(A_q, B_shuffle, A_ss, B_scale_sh)

import torch
from task import input_t, output_t
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

_probed = False

def custom_kernel(data: input_t) -> output_t:
    global _probed
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    
    if not _probed:
        _probed = True
        # Print the actual config for a16wfp4
        try:
            from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16wfp4 import _get_config
            cfg, _ = _get_config(M, N, K)
            print(f"a16wfp4 config for M={M},N={N},K={K}: {cfg}")
        except Exception as e:
            print(f"a16wfp4 _get_config error: {e}")
        
        # Print the config for afp4wfp4
        try:
            from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import _get_config as _get_config2
            cfg2, _ = _get_config2(M, N, K)
            print(f"afp4wfp4 config for M={M},N={N},K={K}: {cfg2}")
        except Exception as e:
            print(f"afp4wfp4 _get_config error: {e}")
    
    # Use working path
    A_q, A_scale = dynamic_mxfp4_quant(A)
    A_scale_sh = e8m0_shuffle(A_scale)
    return gemm_afp4wfp4(A_q, B_shuffle, A_scale_sh, B_scale_sh)

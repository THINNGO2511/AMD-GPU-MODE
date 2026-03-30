#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""GEMM: Inject M>1 tuned configs into CK ASM CSV, then use gemm_a4w4.
The default CSV only has M=1 entries. We add entries for our benchmark M values."""
import torch, os
from task import input_t, output_t

# Inject configs BEFORE importing aiter (so the LRU cache picks them up)
# The CSV format: cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio

# Available kernel binaries (from probe):
# f4gemm_bf16_per1x32Fp4_BpreShuffle_{M_tile}x{N_tile}.co
# M_tiles: 32,64,96,128,160,192,224,256
# N_tiles: 128,256,384,512,640,768,896,1024

# For M=4 K=512: Use 32x128 tile, splitK=1
# For M=16 K=7168: Use 32x128 tile, splitK=4 (K/4=1792)
# For M=32 K=512: Use 32x128 tile, splitK=1
# For M=64 K=2048: Use 64x128 tile, splitK=2
# For M=256 K=1536: Use 256x128 tile, splitK=1

INJECT_LINES = [
    # cu_num,M,N,K,kernelId,splitK,us,kernelName,tflops,bw,errRatio
    "256,4,2880,512,0,1,5.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128,0,0,0",
    "256,16,2112,7168,0,4,10.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128,0,0,0",
    "256,32,4096,512,0,1,5.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128,0,0,0",
    "256,32,2880,512,0,1,5.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128,0,0,0",
    "256,64,7168,2048,0,2,8.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_64x128,0,0,0",
    "256,256,3072,1536,0,1,10.0,f4gemm_bf16_per1x32Fp4_BpreShuffle_256x128,0,0,0",
]

# Append to the tuned CSV
CSV_PATH = "/home/runner/aiter/aiter/configs/a4w4_blockscale_tuned_gemm.csv"
try:
    with open(CSV_PATH, "a") as f:
        for line in INJECT_LINES:
            f.write("\n" + line)
except Exception:
    pass

# Clear the LRU cache so new configs are picked up
try:
    import aiter
    if hasattr(aiter, 'get_GEMM_config'):
        aiter.get_GEMM_config.cache_clear()
except Exception:
    pass

from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle
import aiter

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]
    
    A_q, A_s = dynamic_mxfp4_quant(A)
    A_s_sh = e8m0_shuffle(A_s)
    
    return aiter.gemm_a4w4(
        A_q.view(B_shuffle.dtype),
        B_shuffle,
        A_s_sh.view(B_scale_sh.dtype),
        B_scale_sh,
        dtype=torch.bfloat16,
        bpreshuffle=True,
    )

#!/usr/bin/env python3
"""
Hybrid GEMM submission: Triton gemm_a16wfp4 (K≤1024) + gemm_afp4wfp4 (K>1024)
Pure Triton/aiter — no HIP kernels.

Strategy per shape:
  K=512,  M=4:   gemm_a16wfp4  BM=16, BN=128, waves_per_eu=2
  K=512,  M=32:  gemm_a16wfp4  BM=32, BN=128, waves_per_eu=2
  K=512,  M=32:  gemm_a16wfp4  BM=32, BN=128, waves_per_eu=2
  K=1536, M=256: gemm_afp4wfp4 (separate quant)
  K=2048, M=64:  gemm_afp4wfp4 (separate quant)
  K=7168, M=16:  gemm_afp4wfp4 (separate quant)
"""

import torch
import triton
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
from aiter.utility.fp4_utils import dynamic_mxfp4_quant, e8m0_shuffle

# ---------------------------------------------------------------------------
# Shape definitions: (M, N, K)
# ---------------------------------------------------------------------------
ALL_SHAPES = [
    (4,   5120, 512),
    (32,  5120, 512),
    (32,  1024, 512),
    (256, 7168, 1536),
    (64,  7168, 2048),
    (16,  1024, 7168),
]

# ---------------------------------------------------------------------------
# Tuned Triton configs for gemm_a16wfp4 (K ≤ 1024 path)
# Every config MUST include "cache_modifier": ""
# ---------------------------------------------------------------------------
A16WFP4_CONFIGS = {
    # (M, N, K) -> config dict
    (4, 5120, 512): {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "cache_modifier": "",
    },
    (32, 5120, 512): {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "cache_modifier": "",
    },
    (32, 1024, 512): {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4,
        "SPLIT_K": 1,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 2,
        "cache_modifier": "",
    },
}

# ---------------------------------------------------------------------------
# Pre-warm cache: compile all kernels on first invocation
# ---------------------------------------------------------------------------
_WARMED = False

def _prewarm(B_q_dict, B_scale_dict, device):
    """
    Pre-warm all 6 shapes so the first real call doesn't eat compile time.
    B_q_dict / B_scale_dict are keyed by (N, K) since weights are shared.
    """
    global _WARMED
    if _WARMED:
        return
    for (M, N, K) in ALL_SHAPES:
        A_warm = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        bq = B_q_dict[(N, K)]
        bs = B_scale_dict[(N, K)]
        if K <= 1024:
            # a16wfp4 path — B_q as uint8, B_scale unshuffled
            bq_u8 = bq.view(torch.uint8)
            bs_unshuf = e8m0_shuffle(bs, "unshuffle")
            cfg = A16WFP4_CONFIGS[(M, N, K)]
            gemm_a16wfp4(A_warm, bq_u8, bs_unshuf, cfg)
        else:
            # afp4wfp4 path — quantize A on the fly
            A_q, A_s = dynamic_mxfp4_quant(A_warm)
            gemm_afp4wfp4(A_q, A_s, bq, bs)
    # sync to finish all compilation
    torch.cuda.synchronize(device)
    _WARMED = True


# ---------------------------------------------------------------------------
# Core GEMM dispatch
# ---------------------------------------------------------------------------
def gemm_fp4(A: torch.Tensor,
             B_q: torch.Tensor,
             B_scale: torch.Tensor) -> torch.Tensor:
    """
    A:       (M, K) bfloat16
    B_q:     (N, K//2) packed fp4 — stored as uint8 pairs
    B_scale: (N, K//32) e8m0 block scales
    Returns: (M, N) bfloat16
    """
    M, K = A.shape
    N = B_q.shape[0]

    if K <= 1024:
        # --- gemm_a16wfp4: skip A quantization entirely ---
        bq_u8 = B_q.view(torch.uint8)
        bs_unshuf = e8m0_shuffle(B_scale, "unshuffle")
        cfg = A16WFP4_CONFIGS.get((M, N, K))
        if cfg is None:
            # Fallback config for unexpected shapes in a16wfp4 range
            cfg = {
                "BLOCK_SIZE_M": 16 if M <= 16 else 32,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 4,
                "SPLIT_K": 1,
                "num_warps": 4,
                "num_stages": 2,
                "waves_per_eu": 2,
                "cache_modifier": "",
            }
        return gemm_a16wfp4(A, bq_u8, bs_unshuf, cfg)
    else:
        # --- gemm_afp4wfp4: quantize A, then both-sides-fp4 GEMM ---
        A_q, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_q, A_scale, B_q, B_scale)


# ---------------------------------------------------------------------------
# Competition entry point
# ---------------------------------------------------------------------------
class GemmFP4:
    """
    Drop-in replacement for the competition's GEMM interface.
    Handles pre-warming + hybrid dispatch.
    """
    def __init__(self):
        self._weight_cache_q = {}      # (N,K) -> B_q
        self._weight_cache_s = {}      # (N,K) -> B_scale
        self._device = None
        self._first_call = True

    def set_weights(self, N: int, K: int, B_q: torch.Tensor, B_scale: torch.Tensor):
        """Cache pre-quantized weights for a given (N, K)."""
        self._weight_cache_q[(N, K)] = B_q
        self._weight_cache_s[(N, K)] = B_scale
        self._device = B_q.device

    def __call__(self, A: torch.Tensor,
                 B_q: torch.Tensor,
                 B_scale: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        N = B_q.shape[0]

        # Pre-warm all shapes on first real call
        if self._first_call and self._weight_cache_q:
            _prewarm(self._weight_cache_q, self._weight_cache_s, self._device)
            self._first_call = False

        return gemm_fp4(A, B_q, B_scale)


# ---------------------------------------------------------------------------
# Standalone benchmark (run directly to test)
# ---------------------------------------------------------------------------
def _benchmark():
    import time

    device = "cuda"
    print("=" * 70)
    print("Hybrid Triton GEMM Benchmark: a16wfp4 (K≤1024) + afp4wfp4 (K>1024)")
    print("=" * 70)

    solver = GemmFP4()

    # Pre-generate weights for all shapes
    for (M, N, K) in ALL_SHAPES:
        B_q = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)
        # Scale shape: (N, K//32) e8m0 format
        B_scale = torch.randint(0, 256, (N, K // 32), dtype=torch.uint8, device=device)
        solver.set_weights(N, K, B_q, B_scale)

    for (M, N, K) in ALL_SHAPES:
        A = torch.randn(M, K, dtype=torch.bfloat16, device=device)
        B_q = solver._weight_cache_q[(N, K)]
        B_scale = solver._weight_cache_s[(N, K)]

        # Warm up
        for _ in range(5):
            C = solver(A, B_q, B_scale)
        torch.cuda.synchronize()

        # Benchmark
        n_iters = 200
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            C = solver(A, B_q, B_scale)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        us = (t1 - t0) / n_iters * 1e6
        path = "a16wfp4" if K <= 1024 else "afp4wfp4"
        tflops = 2 * M * N * K / (us * 1e-6) / 1e12
        print(f"  M={M:>4}, N={N:>5}, K={K:>5}  |  {path:>10}  |  {us:7.1f} μs  |  {tflops:5.2f} TFLOPS")

    print("=" * 70)


if __name__ == "__main__":
    _benchmark()

# ── Entry point for evaluator ──
_gemm = None
def custom_kernel(data):
    global _gemm
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B_q.shape[0]
    if _gemm is None:
        _gemm = GemmFP4()
    _gemm.set_weights(N, K, B_q, B_scale_sh)
    return _gemm(A, B_q, B_scale_sh)

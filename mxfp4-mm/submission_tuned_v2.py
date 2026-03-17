#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""
MXFP4 GEMM — Monkey-patch get_GEMM_config to inject tuned configs.

The config lookup returns None for most benchmark sizes, causing the
wrapper to use suboptimal default tile selection. We inject tuned configs
that select better kernel variants from the 35+ available CK binaries.

Available M-tiles: 32, 64, 96, 128, 160, 192, 224, 256
Available N-tiles: 128, 256, 384, 512, 640, 768, 896, 1024

The config for M=256,N=3072,K=1536 already exists (32x128, 6.18us)
but the wrapper sometimes doesn't find it. We ensure all sizes get configs.
"""
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    import torch
    import aiter
    from aiter import dtypes
    from aiter.ops.triton.quant import dynamic_mxfp4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle
    import aiter.ops.gemm_op_a4w4 as gemm_module

    A, B, B_q, B_shuffle, B_scale_sh = data
    A = A.contiguous()
    m, k = A.shape
    n = B.shape[0]

    # Quantize A to MXFP4
    A_fp4, A_scale_e8m0 = dynamic_mxfp4_quant(A)
    A_q = A_fp4.view(dtypes.fp4x2)
    A_scale_sh = e8m0_shuffle(A_scale_e8m0).view(dtypes.fp8_e8m0)

    # Monkey-patch get_GEMM_config to inject our tuned selections
    original_get_config = gemm_module.get_GEMM_config

    def tuned_get_config(m_val, n_val, k_val):
        # Try original first
        config = original_get_config(m_val, n_val, k_val)
        if config is not None:
            return config

        # Our tuned configs based on available kernels
        # Format: kernel name uses _ZN5aiter{len}f4gemm_..._MxNE
        # For small M, use 32x{N_tile} with larger N_tile to increase parallelism
        # For larger M, match M_tile to minimize padding waste

        # Choose M_tile (minimum tile >= M, from available: 32,64,96,128,160,192,224,256)
        if m_val <= 32:
            m_tile = 32
        elif m_val <= 64:
            m_tile = 64
        elif m_val <= 96:
            m_tile = 96
        elif m_val <= 128:
            m_tile = 128
        elif m_val <= 160:
            m_tile = 160
        elif m_val <= 192:
            m_tile = 192
        elif m_val <= 224:
            m_tile = 224
        else:
            m_tile = 256

        # Choose N_tile: larger N_tile = fewer blocks, good for small M
        # but must not exceed available tiles for this M_tile
        n_tiles_by_m = {
            32: [128, 256, 384, 512, 640, 768, 896, 1024],
            64: [128, 256, 384, 512, 640, 768, 896, 1024],
            96: [128, 256, 384, 512, 640],
            128: [128, 256, 384, 512],
            160: [128, 256, 384],
            192: [128, 256],
            224: [128, 256],
            256: [128, 256],
        }

        available_n = n_tiles_by_m.get(m_tile, [128])

        # MI355X has ~304 CUs. We want enough blocks for good occupancy.
        # total_blocks = m_blocks * n_blocks
        m_blocks = (m_val + m_tile - 1) // m_tile

        # Score each N-tile: balance padding waste vs parallelism
        best_n_tile = 128
        best_score = float('-inf')
        for nt in available_n:
            n_blocks = (n_val + nt - 1) // nt
            total_blocks = m_blocks * n_blocks
            waste_frac = (n_blocks * nt - n_val) / (n_blocks * nt)

            # Want: high parallelism (many blocks), low waste
            # Penalize heavily if total_blocks < 32 (GPU underutilized)
            if total_blocks < 8:
                parallelism_score = -100
            elif total_blocks < 32:
                parallelism_score = total_blocks * 2
            else:
                parallelism_score = 64 + total_blocks * 0.1

            score = parallelism_score - waste_frac * 50
            if score > best_score:
                best_score = score
                best_n_tile = nt

        tile_str = f"{m_tile}x{best_n_tile}"
        name_len = len(f"f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_str}")
        kernel_name = f"_ZN5aiter{name_len}f4gemm_bf16_per1x32Fp4_BpreShuffle_{tile_str}E"

        return {
            'kernelId': 0,
            'splitK': 0,
            'us': 0,
            'kernelName': kernel_name,
            'tflops': 0,
            'bw': 0,
            'errRatio': 0.0,
        }

    # Apply monkey-patch
    gemm_module.get_GEMM_config = tuned_get_config

    out = aiter.gemm_a4w4(
        A_q,
        B_shuffle,
        A_scale_sh,
        B_scale_sh,
        dtype=dtypes.bf16,
        bpreshuffle=True,
    )

    # Restore original
    gemm_module.get_GEMM_config = original_get_config

    return out

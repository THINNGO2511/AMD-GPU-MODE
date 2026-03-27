#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Pure PyTorch implementation of our HIP kernel logic to verify data processing."""
import torch
from task import input_t, output_t

FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
           0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def _unshuffle_e8m0(s):
    s = s.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    bu = B_q.view(torch.uint8)  # [N, K/2]
    bs_raw = _unshuffle_e8m0(B_scale_sh)[:N, :].contiguous()  # [N, K/32] uint8

    # Build FP4 LUT tensor
    lut = torch.tensor(FP4_LUT, dtype=torch.float32, device='cuda')

    # Dequant B: [N, K]
    nkb = K // 32
    B_dq = torch.zeros(N, K, dtype=torch.float32, device='cuda')
    for kb in range(nkb):
        # Get scales: [N]
        scale_bytes = bs_raw[:, kb].long()  # [N] uint8 as int
        scale = (scale_bytes << 23).view(torch.int32).view(torch.float32)  # e8m0 to float

        for j in range(16):
            byte_vals = bu[:, kb * 16 + j].long()  # [N]
            lo = byte_vals & 0xF
            hi = byte_vals >> 4
            B_dq[:, kb*32+2*j] = lut[lo] * scale
            B_dq[:, kb*32+2*j+1] = lut[hi] * scale

    # Quant-dequant A using same e8m0 logic
    A_f = A.float()  # [M, K]
    A_blocks = A_f.view(M, nkb, 32)  # [M, nkb, 32]
    A_amax = A_blocks.abs().amax(dim=2)  # [M, nkb]

    # Compute e8m0 scale for A
    amax_bits = A_amax.view(torch.int32)
    be = (amax_bits >> 23) & 0xFF
    mantissa = amax_bits & 0x7FFFFF
    se = be - torch.where(mantissa > 0x400000,
                          torch.ones_like(be), 2 * torch.ones_like(be))
    se = se.clamp(1, 254)
    se[A_amax == 0] = 0
    a_scale = (se << 23).view(torch.int32).view(torch.float32)  # [M, nkb]

    # Quantize A to fp4 then dequant
    A_qd = torch.zeros_like(A_f)
    for kb in range(nkb):
        sc = a_scale[:, kb:kb+1]  # [M, 1]
        inv_sc = torch.where(sc > 0, 1.0 / sc, torch.zeros_like(sc))
        av = A_blocks[:, kb, :]  # [M, 32]
        av_scaled = av * inv_sc  # [M, 32]
        ax = av_scaled.abs()

        # Round to nearest fp4
        idx = torch.zeros_like(ax, dtype=torch.long)
        idx[ax >= 0.25] = 1
        idx[ax >= 0.75] = 2
        idx[ax >= 1.25] = 3
        idx[ax >= 1.75] = 4
        idx[ax >= 2.5] = 5
        idx[ax >= 3.5] = 6
        idx[ax >= 5.0] = 7
        # Add sign
        idx[av_scaled < 0] += 8

        A_qd[:, kb*32:(kb+1)*32] = lut[idx] * sc

    # Matmul
    C = A_qd @ B_dq.T  # [M, N]
    return C.to(torch.bfloat16)

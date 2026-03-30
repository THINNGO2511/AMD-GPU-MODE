#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe input formats to understand B_q layout."""
import torch
from task import input_t, output_t
from aiter.ops.triton.quant import dynamic_mxfp4_quant

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    M, K = A.shape
    N = B.shape[0]

    print(f"=== SHAPES ===")
    print(f"A: {A.shape} {A.dtype}")
    print(f"B: {B.shape} {B.dtype}")
    print(f"B_q: {B_q.shape} {B_q.dtype} nbytes={B_q.nbytes}")
    print(f"B_shuffle: {B_shuffle.shape} {B_shuffle.dtype} nbytes={B_shuffle.nbytes}")
    print(f"B_scale_sh: {B_scale_sh.shape} {B_scale_sh.dtype} nbytes={B_scale_sh.nbytes}")

    # Check B_q viewed as uint8
    try:
        bu = B_q.view(torch.uint8)
        print(f"B_q.view(uint8): {bu.shape}")
    except Exception as e:
        print(f"B_q.view(uint8) failed: {e}")
        bu = B_q.reshape(-1).view(torch.uint8)
        print(f"B_q flat uint8: {bu.shape}")

    # Check B_scale_sh viewed as uint8
    try:
        su = B_scale_sh.view(torch.uint8)
        print(f"B_scale_sh.view(uint8): {su.shape}")
    except Exception as e:
        print(f"B_scale_sh.view(uint8) failed: {e}")

    # Try dynamic_mxfp4_quant on A to see format
    try:
        A_q, A_scale = dynamic_mxfp4_quant(A)
        print(f"A_q: {A_q.shape} {A_q.dtype}")
        print(f"A_scale: {A_scale.shape} {A_scale.dtype}")
    except Exception as e:
        print(f"dynamic_mxfp4_quant failed: {e}")

    # Print first few bytes of B_q and corresponding B values
    print(f"\n=== B_q vs B comparison (first 4 rows, first 64 elements) ===")
    bu = B_q.view(torch.uint8)
    for n in range(min(2, N)):
        print(f"\nRow n={n}:")
        b_vals = B[n, :64].float().cpu().tolist()
        bq_bytes = bu[n, :32].cpu().tolist()
        print(f"  B[{n},0:16] = {[f'{v:.2f}' for v in b_vals[:16]]}")
        print(f"  B_q bytes[0:8] = {[hex(b) for b in bq_bytes[:8]]}")

        # Try dequanting with different scale assumptions
        # If scale_sh[n, 0] is the scale for elements 0:31
        sc_byte = B_scale_sh.view(torch.uint8)[n, 0].item()
        sc_val = 0.0
        if sc_byte > 0:
            import struct
            sc_val = struct.unpack('f', struct.pack('I', sc_byte << 23))[0]
        print(f"  B_scale_sh[{n},0] byte={hex(sc_byte)} val={sc_val}")

        # Dequant first 8 values with low/high nibble
        lut = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
        for j in range(4):
            byte = bq_bytes[j]
            lo = byte & 0xF
            hi = byte >> 4
            lo_sign = -1 if (lo & 8) else 1
            hi_sign = -1 if (hi & 8) else 1
            lo_val = lo_sign * lut[lo & 7] * sc_val
            hi_val = hi_sign * lut[hi & 7] * sc_val
            print(f"  byte[{j}]=0x{byte:02x}: lo_nib=0x{lo:x}→{lo_val:.3f}  hi_nib=0x{hi:x}→{hi_val:.3f}  | B[{n},{2*j}]={b_vals[2*j]:.3f}  B[{n},{2*j+1}]={b_vals[2*j+1]:.3f}")

    # Also check the unshuffle
    print(f"\n=== Unshuffle test ===")
    def _unshuffle_e8m0(s):
        s = s.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        return s.view(sm, sn)

    try:
        bs_raw = _unshuffle_e8m0(B_scale_sh)
        print(f"Unshuffled scale shape: {bs_raw.shape}")
        raw_byte = bs_raw.view(torch.uint8)[0, 0].item()
        print(f"Unshuffled scale[0,0] byte={hex(raw_byte)}")
        if raw_byte > 0:
            import struct
            raw_val = struct.unpack('f', struct.pack('I', raw_byte << 23))[0]
            print(f"Unshuffled scale[0,0] val={raw_val}")
    except Exception as e:
        print(f"Unshuffle failed: {e}")

    # Use reference to compute correct output
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bu8 = B_q.view(torch.uint8)
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    C = gemm_a16wfp4(A, bu8, bs_raw, dtype=torch.bfloat16)
    return C

#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Probe: which scale mapping is correct for B_q? Shuffled or unshuffled?"""
import torch, struct
from task import input_t, output_t

FP4 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
       0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

def e8m0_f(e):
    if e == 0: return 0.0
    return struct.unpack('f', struct.pack('I', int(e) << 23))[0]

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
    bs_sh = B_scale_sh.view(torch.uint8)  # [N_padded, K/32]
    bs_unsh = _unshuffle_e8m0(B_scale_sh).view(torch.uint8)  # [N_padded, K/32]
    nkb = K // 32

    print(f"=== Scale comparison for N={N}, K={K}, nkb={nkb} ===")
    print(f"B_scale_sh shape: {bs_sh.shape}")
    print(f"B_scale_unsh shape: {bs_unsh.shape}")

    # Compare dequant error for first 4 rows, all K-blocks
    for n in range(min(2, N)):
        err_sh_total = 0.0
        err_unsh_total = 0.0
        cnt = 0
        for kb in range(min(nkb, 6)):
            sc_sh = e8m0_f(bs_sh[n, kb].item())
            sc_unsh = e8m0_f(bs_unsh[n, kb].item()) if n < bs_unsh.shape[0] else 0

            # Compute expected scale from B values
            b_block = B[n, kb*32:(kb+1)*32].float().cpu()
            amax = b_block.abs().max().item()

            # Dequant first 4 values using shuffled scale
            dq_sh = []
            dq_unsh = []
            for j in range(4):
                byte = bu[n, kb*16+j].item()
                lo, hi = byte & 0xF, byte >> 4
                dq_sh.append(FP4[lo] * sc_sh)
                dq_sh.append(FP4[hi] * sc_sh)
                dq_unsh.append(FP4[lo] * sc_unsh)
                dq_unsh.append(FP4[hi] * sc_unsh)

            b_actual = [b_block[j].item() for j in range(8)]
            e_sh = sum(abs(dq_sh[j] - b_actual[j]) for j in range(8)) / 8
            e_unsh = sum(abs(dq_unsh[j] - b_actual[j]) for j in range(8)) / 8

            if kb < 3:
                print(f"  n={n} kb={kb}: sc_sh={sc_sh:.4f} sc_unsh={sc_unsh:.4f} amax={amax:.4f}")
                print(f"    B[{n},{kb*32}:{kb*32+8}] = {[f'{v:.3f}' for v in b_actual]}")
                print(f"    dq_sh  = {[f'{v:.3f}' for v in dq_sh[:8]]}")
                print(f"    dq_unsh = {[f'{v:.3f}' for v in dq_unsh[:8]]}")
                print(f"    err_sh={e_sh:.4f}  err_unsh={e_unsh:.4f}")

            err_sh_total += e_sh
            err_unsh_total += e_unsh
            cnt += 1

        print(f"  n={n}: avg_err_sh={err_sh_total/cnt:.4f}  avg_err_unsh={err_unsh_total/cnt:.4f}")

    # Also try: compute scale from B values directly
    print(f"\n=== Recomputed scales comparison ===")
    for n in range(min(2, N)):
        for kb in range(min(3, nkb)):
            b_block = B[n, kb*32:(kb+1)*32].float().cpu()
            amax = b_block.abs().max().item()
            # Compute e8m0 scale
            if amax > 0:
                bits = struct.unpack('I', struct.pack('f', amax))[0]
                be = (bits >> 23) & 0xFF
                m = bits & 0x7FFFFF
                se = be - (1 if m > 0x400000 else 2)
                se = max(1, min(254, se))
                computed_scale = e8m0_f(se)
            else:
                computed_scale = 0
            sc_sh = e8m0_f(bs_sh[n, kb].item())
            sc_unsh = e8m0_f(bs_unsh[n, kb].item()) if n < bs_unsh.shape[0] else 0
            match_sh = "MATCH" if abs(sc_sh - computed_scale) < 1e-10 else "MISMATCH"
            match_unsh = "MATCH" if abs(sc_unsh - computed_scale) < 1e-10 else "MISMATCH"
            print(f"  n={n} kb={kb}: computed={computed_scale:.6f}  sh={sc_sh:.6f}({match_sh})  unsh={sc_unsh:.6f}({match_unsh})")

    # Use reference
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    bs_raw = _unshuffle_e8m0(B_scale_sh)
    C = gemm_a16wfp4(A, bu, bs_raw, dtype=torch.bfloat16)
    return C

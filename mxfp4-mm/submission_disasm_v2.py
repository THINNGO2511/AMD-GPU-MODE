#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Disasm v2: find correct llvm-objdump path on runner"""
from task import input_t, output_t
import torch, subprocess, os, glob

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data

    # Find llvm-objdump
    paths = glob.glob("/opt/rocm/*/bin/llvm-objdump") + glob.glob("/opt/rocm/lib/llvm/bin/*objdump*")
    paths += glob.glob("/opt/rocm/bin/*objdump*")
    print(f"objdump candidates: {paths}")

    co = "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co"
    for p in paths:
        try:
            r = subprocess.run([p, "-d", co], capture_output=True, text=True, timeout=5)
            lines = r.stdout.split('\n')
            # Find v_mfma
            for i, l in enumerate(lines):
                if 'mfma' in l.lower():
                    start = max(0, i-10)
                    end = min(len(lines), i+3)
                    print(f"\n=== MFMA found with {p} at line {i} ===")
                    for j in range(start, end):
                        print(f"  {lines[j]}")
                    break
            else:
                print(f"{p}: {len(lines)} lines, no mfma found")
                if len(lines) > 5:
                    for l in lines[:5]: print(f"  {l}")
            break
        except Exception as e:
            print(f"{p}: {e}")

    # Also try /opt/rocm/lib/llvm/bin/llvm-objdump directly
    for candidate in ["/opt/rocm/lib/llvm/bin/llvm-objdump", "/opt/rocm/llvm/bin/llvm-objdump"]:
        if os.path.exists(candidate):
            print(f"\nFound: {candidate}")
            try:
                r = subprocess.run([candidate, "-d", "--triple=amdgcn-amd-amdhsa", co],
                                   capture_output=True, text=True, timeout=10)
                lines = r.stdout.split('\n')
                for i, l in enumerate(lines):
                    if 'mfma' in l.lower() or 'scale' in l.lower():
                        start = max(0, i-15)
                        end = min(len(lines), i+3)
                        print(f"\n=== MFMA context (line {i}) ===")
                        for j in range(start, end):
                            print(f"  {lines[j]}")
                        break
            except Exception as e:
                print(f"  Error: {e}")

    # Fallback
    m, k = A.shape
    n = B.shape[0]
    _bscale_raw = B_scale_sh.view(torch.uint8)
    sm, sn = _bscale_raw.shape
    _bscale_raw = _bscale_raw.view(sm//32, sn//8, 4, 16, 2, 2).permute(0,5,3,1,4,2).contiguous().view(sm,sn)
    _bq_u8 = B_q.view(torch.uint8)
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = {"BLOCK_SIZE_M":8,"BLOCK_SIZE_N":64,"BLOCK_SIZE_K":512,"GROUP_SIZE_M":1,"num_warps":4,
           "num_stages":2,"waves_per_eu":2,"matrix_instr_nonkdim":16,"cache_modifier":None,
           "NUM_KSPLIT":8,"SPLITK_BLOCK_SIZE":1024} if k==7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)

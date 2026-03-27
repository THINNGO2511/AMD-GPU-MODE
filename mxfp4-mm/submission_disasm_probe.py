#!POPCORN leaderboard amd-mxfp4-mm
#!POPCORN gpu MI355X
"""Disassemble CK ASM kernel to see MFMA data loading pattern."""
from task import input_t, output_t
import torch

def custom_kernel(data: input_t) -> output_t:
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    import subprocess, os

    co_file = "/home/runner/aiter/hsa/gfx950/f4gemm/f4gemm_bf16_per1x32Fp4_BpreShuffle_32x128.co"
    try:
        # Try llvm-objdump
        result = subprocess.run(
            ["llvm-objdump", "-d", "--triple=amdgcn-amd-amdhsa", co_file],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.split('\n')
        # Find MFMA instruction and surrounding loads
        mfma_lines = []
        for i, line in enumerate(lines):
            if 'mfma' in line.lower() or 'v_mfma' in line.lower():
                # Print context: 20 lines before and after
                start = max(0, i-20)
                end = min(len(lines), i+5)
                for j in range(start, end):
                    mfma_lines.append(f"L{j}: {lines[j]}")
                break
        if mfma_lines:
            print("=== MFMA instruction context (first occurrence) ===")
            for l in mfma_lines:
                print(l)
        else:
            print(f"No MFMA found. Total lines: {len(lines)}")
            # Print first 50 lines
            for line in lines[:50]:
                print(line)
    except Exception as e:
        print(f"llvm-objdump failed: {e}")

    # Try rocm-objdump
    try:
        result2 = subprocess.run(
            ["/opt/rocm/bin/amdgpu-objdump", "-d", co_file],
            capture_output=True, text=True, timeout=10
        )
        if result2.stdout:
            lines2 = result2.stdout.split('\n')
            print(f"\n=== amdgpu-objdump: {len(lines2)} lines ===")
            for line in lines2[:30]:
                print(line)
    except Exception as e:
        print(f"amdgpu-objdump: {e}")

    # Check available disassembly tools
    for tool in ["llvm-objdump", "amdgpu-objdump", "rocm-objdump", "objdump"]:
        try:
            r = subprocess.run(["which", tool], capture_output=True, text=True)
            if r.returncode == 0:
                print(f"Found: {r.stdout.strip()}")
        except:
            pass

    # Fallback: use proven Triton path
    _bscale_raw = B_scale_sh.view(torch.uint8)
    sm, sn = _bscale_raw.shape
    _bscale_raw = _bscale_raw.view(sm // 32, sn // 8, 4, 16, 2, 2)
    _bscale_raw = _bscale_raw.permute(0, 5, 3, 1, 4, 2).contiguous().view(sm, sn)
    _bq_u8 = B_q.view(torch.uint8)

    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        A_fp4, A_scale = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(A_fp4.view(torch.uint8), _bq_u8, A_scale, _bscale_raw, dtype=torch.bfloat16)

    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    cfg = {"BLOCK_SIZE_M": 8, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
           "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
           "waves_per_eu": 2, "matrix_instr_nonkdim": 16, "cache_modifier": None,
           "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024} if k == 7168 else None
    return gemm_a16wfp4(A, _bq_u8, _bscale_raw, dtype=torch.bfloat16, config=cfg)

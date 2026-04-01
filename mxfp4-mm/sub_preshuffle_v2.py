import os
os.environ["HIP_FORCE_DEV_KERNARG"] = "1"
os.environ["OPTIMIZE_EPILOGUE"] = "1"

from task import input_t, output_t
import torch

_y_cache = {}
_warmed = False
_has_preshuffle = None

def custom_kernel(data: input_t) -> output_t:
    global _warmed, _has_preshuffle
    A, B, B_q, B_shuffle, B_scale_sh = data
    m, k = A.shape
    n = B.shape[0]

    # Try preshuffle path — takes B_shuffle + B_scale_sh directly (no unshuffle!)
    if _has_preshuffle is None:
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
            _has_preshuffle = True
            print(f"[PRESHUF] gemm_a16wfp4_preshuffle FOUND")
        except ImportError:
            try:
                from aiter.ops.triton.gemm.basic import gemm_a16wfp4_preshuffle as _mod
                _has_preshuffle = True
                print(f"[PRESHUF] found as separate module")
            except ImportError:
                _has_preshuffle = False
                print(f"[PRESHUF] NOT FOUND, falling back to standard path")

        # Also check what functions exist in the module
        try:
            import aiter.ops.triton.gemm.basic.gemm_a16wfp4 as _gmod
            funcs = [x for x in dir(_gmod) if not x.startswith('_')]
            print(f"[PRESHUF] gemm_a16wfp4 module exports: {funcs}")
        except Exception as e:
            print(f"[PRESHUF] module inspect error: {e}")

        # Check for preshuffle in other locations
        try:
            from aiter import gemm_a16wfp4_preshuffle as _fn
            _has_preshuffle = True
            print(f"[PRESHUF] found in aiter root")
        except ImportError:
            pass

        # List all gemm files
        try:
            gemm_dir = "/home/runner/aiter/aiter/ops/triton/gemm/basic/"
            files = sorted(os.listdir(gemm_dir))
            print(f"[PRESHUF] gemm/basic/ files: {files}")
        except Exception as e:
            print(f"[PRESHUF] listdir error: {e}")

    if _has_preshuffle:
        try:
            from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4_preshuffle
            key = (m, n)
            if key not in _y_cache:
                _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
            out = _y_cache[key]
            # Pass B_shuffle (pre-shuffled) and B_scale_sh (shuffled scales) directly
            b_sh_u8 = B_shuffle.view(torch.uint8)
            gemm_a16wfp4_preshuffle(A, b_sh_u8, B_scale_sh.view(torch.uint8),
                                     dtype=torch.bfloat16, y=out)
            return out
        except Exception as e:
            print(f"[PRESHUF] call failed: {e}, falling back")
            _has_preshuffle = False

    # Standard fallback
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    def _unshuffle_e8m0(scale_sh):
        s = scale_sh.view(torch.uint8)
        sm, sn = s.shape
        s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
        s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
        return s.view(sm, sn)

    bscale_raw = _unshuffle_e8m0(B_scale_sh)
    bq_u8 = B_q.view(torch.uint8)

    key = (m, n)
    if key not in _y_cache:
        _y_cache[key] = torch.empty(m, n, dtype=torch.bfloat16, device=A.device)
    out = _y_cache[key]

    cfg = None
    if k == 7168:
        cfg = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
               "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
               "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
               "cache_modifier": ".cg", "NUM_KSPLIT": 8, "SPLITK_BLOCK_SIZE": 1024}
    if k == 1536:
        from aiter.ops.triton.quant import dynamic_mxfp4_quant
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
        af, asc = dynamic_mxfp4_quant(A)
        return gemm_afp4wfp4(af.view(torch.uint8), bq_u8, asc, bscale_raw, dtype=torch.bfloat16)

    gemm_a16wfp4(A, bq_u8, bscale_raw, dtype=torch.bfloat16, y=out, config=cfg)
    return out

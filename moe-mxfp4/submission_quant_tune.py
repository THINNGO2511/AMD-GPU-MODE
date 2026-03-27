#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Tune fused_dynamic_mxfp4_quant_moe_sort kernel.
Profiling showed quant_sort = 28% of kernel time (called 2x per inference).
The kernel uses hardcoded BLOCK_SIZE_Mx=128. For small token counts (16-512),
smaller BLOCK_SIZE_Mx could give more parallelism or reduce wasted work.

Also: probe _fused_dynamic_mxfp4_quant_moe_sort_kernel to understand the Triton kernel.
Submit with: --mode test first!
"""
import sys
import os
import torch
import triton
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === Probe and patch fused_dynamic_mxfp4_quant_moe_sort ===
    # Find the Triton kernel
    orig_fused_quant = fm.fused_dynamic_mxfp4_quant_moe_sort

    # Try to find the kernel reference
    kernel_ref = None
    try:
        # The kernel is probably in the same module or imported
        import inspect
        src_file = inspect.getfile(orig_fused_quant)
        print(f"[QTUNE] fused_dynamic_mxfp4_quant_moe_sort from: {src_file}", file=sys.stderr)

        # Read source to find kernel import
        with open(src_file) as f:
            content = f.read()
        # Find _fused_dynamic_mxfp4_quant_moe_sort_kernel
        for line in content.split('\n'):
            if '_fused_dynamic_mxfp4_quant_moe_sort_kernel' in line and ('import' in line or 'def' in line or '=' in line):
                print(f"[QTUNE] kernel ref: {line.strip()}", file=sys.stderr)

        # Try importing the module
        mod = inspect.getmodule(orig_fused_quant)
        if hasattr(mod, '_fused_dynamic_mxfp4_quant_moe_sort_kernel'):
            kernel_ref = getattr(mod, '_fused_dynamic_mxfp4_quant_moe_sort_kernel')
            print(f"[QTUNE] Found kernel: {type(kernel_ref)}", file=sys.stderr)
    except Exception as e:
        print(f"[QTUNE] Kernel probe failed: {e}", file=sys.stderr)

    # Monkey-patch with tuned BLOCK_SIZE_Mx
    if kernel_ref is not None:
        def tuned_fused_quant_sort(x, sorted_ids, num_valid_ids, token_num, topk,
                                    block_size=32, scaling_mode="even"):
            M, N = x.shape
            MXFP4_QUANT_BLOCK_SIZE = 32

            x_fp4 = torch.empty((M, N // 2), dtype=torch.uint8, device=x.device)
            scaleN = triton.cdiv(N, MXFP4_QUANT_BLOCK_SIZE)

            # Tune BLOCK_SIZE_Mx based on M
            if M <= 32:
                BLOCK_SIZE_Mx = 32
            elif M <= 128:
                BLOCK_SIZE_Mx = 64
            else:
                BLOCK_SIZE_Mx = 128

            BLOCK_SIZE_M, BLOCK_SIZE_N = 32, 8
            BLOCK_SIZE_M_u32, BLOCK_SIZE_N_u32 = 16, 4

            M_i, N_i = M, scaleN
            M_o, N_o = sorted_ids.shape[0], N_i

            blockscale_e8m0_sorted = torch.empty(
                (triton.cdiv(M_o, BLOCK_SIZE_M),
                 triton.cdiv(N_o, BLOCK_SIZE_N),
                 BLOCK_SIZE_N_u32, BLOCK_SIZE_M_u32, 4),
                dtype=torch.uint8, device=x.device,
            )

            num_pid = (triton.cdiv(M, BLOCK_SIZE_Mx) * scaleN +
                       triton.cdiv(M_o, BLOCK_SIZE_M) * triton.cdiv(N_i, BLOCK_SIZE_N))

            kernel_ref[(num_pid,)](
                x, x_fp4, sorted_ids, num_valid_ids, blockscale_e8m0_sorted,
                M, N, scaleN,
                *x.stride(), *x_fp4.stride(), *blockscale_e8m0_sorted.stride(),
                token_num=token_num, M_i=M_i, N_i=N_i,
                MXFP4_QUANT_BLOCK_SIZE=MXFP4_QUANT_BLOCK_SIZE,
                BLOCK_SIZE_Mx=BLOCK_SIZE_Mx,
                BLOCK_SIZE_M=BLOCK_SIZE_M // 2,
                BLOCK_SIZE_N=BLOCK_SIZE_N // 2,
                TOPK=topk,
            )

            return (
                x_fp4.view(dtypes.fp4x2),
                blockscale_e8m0_sorted.view(dtypes.fp8_e8m0).view(-1, N_o),
            )

        fm.fused_dynamic_mxfp4_quant_moe_sort = tuned_fused_quant_sort
        print(f"[QTUNE] Patched fused_dynamic_mxfp4_quant_moe_sort with tuned BLOCK_SIZE_Mx", file=sys.stderr)
    else:
        print(f"[QTUNE] Could not find kernel, using original function", file=sys.stderr)

    # === Standard best_kernels patches ===
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None
    print(f"[QTUNE] All patches applied", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    return fused_moe(
        hidden_states,
        gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

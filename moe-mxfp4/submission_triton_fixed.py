#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Fixed Triton MoE stage1 with correct gather pattern.
Key findings from diagnostic:
- a1 (quantized activation) stays in ORIGINAL order [M, K//2] — NOT sorted
- a1_scale IS sorted [sorted_len, scale_K] — pre-sorted by fused_dynamic_mxfp4_quant_moe_sort
- sorted_expert_ids from moe_sorting contains garbage beyond valid range
- Must gather A via sorted_token_ids // topk during GEMM
- Must pad A to handle out-of-bounds from padded sorted_token_ids
"""
import torch
import triton
import triton.language as tl
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm
from aiter.fused_moe import moe_sorting, get_inter_dim, fused_dynamic_mxfp4_quant_moe_sort
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

_tested = False
_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched: return
    _patched = True
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
    try: fm._USE_OPUS_MOE_SORTING = True
    except: pass
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled=True):
        r = orig(token, model_dim, inter_dim, expert, topk, dtype, q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, doweight_stage1, hidden_pad, intermediate_pad, is_shuffled)
        if expert <= 64 and q_type == QuantType.per_1x32 and not r.run_1stage and inter_dim < 2048:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    return fm.MOEMetadata(functools.partial(fm.ck_moe_stage1, kernelName=kn1, activation=activation, quant_type=q_type, dtype=dtype, splitk=0, use_non_temporal_load=False), functools.partial(aiter.ck_moe_stage2_fwd, kernelName=STAGE2_32, activation=activation, quant_type=q_type, use_non_temporal_load=False), 32, 0, False)
            except: pass
        return r
    fm.get_2stage_cfgs = new
    fm.cfg_2stages = None


@triton.jit
def _triton_moe_gemm(
    # A: original activation [M, K//2] uint8 (MXFP4 quantized, NOT sorted)
    a_ptr, stride_am, stride_ak,
    # A scale: sorted [sorted_len_padded, scale_K] uint8
    a_scale_ptr, stride_asm, stride_ask,
    # W: expert weights [E, N, K//2] uint8
    w_ptr, stride_we, stride_wn, stride_wk,
    # W scale: [E, N, scale_K] uint8
    ws_ptr, stride_wse, stride_wsn, stride_wsk,
    # Output [M*topk, N] bf16
    out_ptr, stride_om, stride_on,
    # Sorted token IDs
    sorted_ids_ptr,
    # Sorted expert IDs (one per M-block from moe_sorting)
    expert_ids_ptr,
    # Dims
    N, K, num_valid_tokens, top_k,
    num_blocks,  # total M-blocks in sorted order
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if pid_m >= num_blocks:
        return

    # Load sorted token IDs for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_ids_ptr + offs_m)
    token_mask = token_ids < num_valid_tokens

    # Original token index (for A gather)
    orig_token = token_ids // top_k

    # Expert for this block — read from expert_ids
    # expert_ids has one entry per block from moe_sorting
    expert_id = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_k_packed = tl.arange(0, BLOCK_K // 2)

        # Gather A from original order [M, K//2] using orig_token
        a = tl.load(a_ptr + orig_token[:, None] * stride_am + (k_start // 2 + offs_k_packed)[None, :] * stride_ak,
                     mask=token_mask[:, None], other=0)

        # Load pre-sorted A scales [sorted_len, scale_K]
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)
        a_scales = tl.load(
            a_scale_ptr + offs_m[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_ask,
            mask=token_mask[:, None], other=0)

        # Load W for this expert [BLOCK_K//2, BLOCK_N]
        w = tl.load(w_ptr + expert_id * stride_we + (k_start // 2 + offs_k_packed)[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Load W scales [BLOCK_N, BLOCK_K//SCALE_GROUP]
        w_scales = tl.load(ws_ptr + expert_id * stride_wse + offs_n[:, None] * stride_wsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_wsk)

        acc = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", acc)

    # Store output indexed by token_id (scatter)
    result = acc.to(tl.bfloat16)
    out_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(out_ptr + token_ids[:, None] * stride_om + offs_n[None, :] * stride_on,
             result, mask=out_mask)


def custom_kernel(data: input_t) -> output_t:
    global _tested
    _patch()
    (hidden_states, gate_up_weight, down_weight, gate_up_weight_scale, down_weight_scale, gate_up_weight_shuffled, down_weight_shuffled, gate_up_weight_scale_shuffled, down_weight_scale_shuffled, topk_weights, topk_ids, config) = data
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _tested:
        _tested = True
        M = hidden_states.shape[0]
        E = gate_up_weight.shape[0]
        topk = topk_ids.shape[1]
        d_hidden_pad = config["d_hidden_pad"]
        d_expert_pad = config["d_expert_pad"]
        _, model_dim, inter_dim = get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)

        try:
            BLOCK_M = 32
            sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
                topk_ids, topk_weights, E, model_dim, torch.bfloat16, BLOCK_M, None, None, 0)

            # Quantize (stays in original order)
            a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
                hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
                token_num=M, topk=1, block_size=BLOCK_M)

            a1_u8 = a1.view(torch.uint8)  # [M, K//2]
            a1_scale_u8 = a1_scale.view(torch.uint8)  # [sorted_len_padded, scale_K]

            # Pad A to handle gather for large sorted_token_ids
            max_orig_row = sorted_ids.max().item() // topk + 1
            if a1_u8.shape[0] < max_orig_row:
                pad = max_orig_row - a1_u8.shape[0]
                a1_u8 = torch.cat([a1_u8, torch.zeros(pad, a1_u8.shape[1], dtype=torch.uint8, device='cuda')])

            N = 2 * d_expert_pad
            K = d_hidden_pad
            w1 = gate_up_weight.view(torch.uint8)
            w1_scale_3d = gate_up_weight_scale.view(torch.uint8).view(E, N, K // 32)

            sorted_len = sorted_ids.shape[0]
            num_blocks = sorted_len // BLOCK_M
            num_valid = M * topk

            # Compute expert ID per block from sorted_ids
            # Take the expert of the FIRST token in each block
            block_starts = sorted_ids[::BLOCK_M][:num_blocks]
            # expert_id = topk_ids[orig_token, k_idx] but we need to map back
            # Actually use sorted_expert_ids[:num_blocks] — let's check if it has valid data
            expert_ids_per_block = sorted_expert_ids[:num_blocks]
            print(f"[FIX] expert_ids[:5]: {expert_ids_per_block[:5].tolist()}")
            print(f"[FIX] expert_ids range: [{expert_ids_per_block.min()}, {expert_ids_per_block.max()}]")
            print(f"[FIX] a1_u8: {a1_u8.shape}, a1_scale: {a1_scale_u8.shape}")
            print(f"[FIX] num_blocks={num_blocks}, sorted_len={sorted_len}, num_valid={num_valid}")

            # Output buffer
            out = torch.zeros((num_valid, N), dtype=torch.bfloat16, device='cuda')

            BLOCK_N = 128
            BLOCK_K = 128
            grid = (num_blocks * triton.cdiv(N, BLOCK_N),)
            print(f"[FIX] Grid: {grid}")

            # Only launch if expert_ids look valid
            if expert_ids_per_block.min() >= 0 and expert_ids_per_block.max() < E:
                _triton_moe_gemm[grid](
                    a1_u8, a1_u8.stride(0), a1_u8.stride(1),
                    a1_scale_u8, a1_scale_u8.stride(0), a1_scale_u8.stride(1),
                    w1, w1.stride(0), w1.stride(1), w1.stride(2),
                    w1_scale_3d, w1_scale_3d.stride(0), w1_scale_3d.stride(1), w1_scale_3d.stride(2),
                    out, out.stride(0), out.stride(1),
                    sorted_ids, expert_ids_per_block,
                    N, K, num_valid, topk, num_blocks,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                )
                torch.cuda.synchronize()
                print(f"[FIX] SUCCESS! out range: [{out.min():.4f}, {out.max():.4f}]")
            else:
                print(f"[FIX] SKIPPED — invalid expert_ids")
        except Exception as e:
            import traceback
            print(f"[FIX] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

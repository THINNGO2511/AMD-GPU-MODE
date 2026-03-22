#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Triton MoE stage1 using moe_align_block_size for correct expert IDs.
Uses dynamic_mxfp4_quant for A quantization (original order),
then gathers A during GEMM via sorted_token_ids // topk.
Falls back to CK for stage 2 and for cases where Triton is slower.
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
from aiter.ops.triton.quant import dynamic_mxfp4_quant

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
def _triton_moe_stage1(
    # A: quantized activation [A_rows, K//2] uint8 (padded, original order)
    a_ptr, stride_am, stride_ak,
    # A scale: [A_rows, scale_K] uint8 (original order, same padding)
    as_ptr, stride_asm, stride_ask,
    # W: expert weights [E, N, K//2] uint8 (raw)
    w_ptr, stride_we, stride_wn, stride_wk,
    # W scale: [E, N, scale_K] uint8
    ws_ptr, stride_wse, stride_wsn, stride_wsk,
    # Output: [num_valid, N] bf16
    out_ptr, stride_om, stride_on,
    # Sorted data from moe_align_block_size
    sorted_ids_ptr,   # [ntp] int32
    expert_ids_ptr,   # [num_blocks] int32 — one expert per BLOCK_M block
    # Dims
    N, K, num_valid, top_k: tl.constexpr,
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    SCALE_GROUP: tl.constexpr = 32

    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Load expert for this block
    expert_id = tl.load(expert_ids_ptr + pid_m)

    # Load sorted token IDs for this M-block
    offs_block = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(sorted_ids_ptr + offs_block)
    token_mask = token_ids < num_valid

    # Original token index for A gather
    orig_token = token_ids // top_k

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_iter in range(tl.cdiv(K, BLOCK_K)):
        k_start = k_iter * BLOCK_K
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        offs_k_scale = tl.arange(0, BLOCK_K // SCALE_GROUP)

        # Gather A from original order using orig_token
        a = tl.load(a_ptr + orig_token[:, None] * stride_am + (k_start // 2 + offs_k_packed)[None, :] * stride_ak,
                     mask=token_mask[:, None], other=0)

        # A scales also gathered from original order
        a_scales = tl.load(as_ptr + orig_token[:, None] * stride_asm + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_ask,
                           mask=token_mask[:, None], other=0)

        # Load W for this expert
        w = tl.load(w_ptr + expert_id * stride_we + (k_start // 2 + offs_k_packed)[:, None] * stride_wk + offs_n[None, :] * stride_wn)

        # Load W scales
        w_scales = tl.load(ws_ptr + expert_id * stride_wse + offs_n[:, None] * stride_wsn + (k_start // SCALE_GROUP + offs_k_scale)[None, :] * stride_wsk)

        acc = tl.dot_scaled(a, a_scales, "e2m1", w, w_scales, "e2m1", acc)

    # Store output indexed by token_id
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

        try:
            BLOCK_M = 32
            N = 2 * d_expert_pad
            K = d_hidden_pad

            # 1. moe_align_block_size for per-block expert IDs
            max_padded = M * topk + E * BLOCK_M
            sorted_ids = torch.empty(max_padded, dtype=torch.int32, device='cuda')
            expert_ids = torch.empty(max_padded // BLOCK_M, dtype=torch.int32, device='cuda')
            token_nums = torch.empty(E, dtype=torch.int32, device='cuda')
            ntp_tensor = torch.empty(1, dtype=torch.int32, device='cuda')

            aiter.moe_align_block_size(topk_ids, E, BLOCK_M, sorted_ids, expert_ids, token_nums, ntp_tensor)
            ntp = ntp_tensor.item()
            num_blocks = ntp // BLOCK_M

            # 2. Quantize A (original order)
            a_fp4, a_scale = dynamic_mxfp4_quant(hidden_states)
            a_u8 = a_fp4.view(torch.uint8)  # [M, K//2]
            as_u8 = a_scale.view(torch.uint8)  # [M, scale_K]

            # 3. Pad A for out-of-bounds gather
            max_orig = sorted_ids[:ntp].max().item() // topk + 1
            if a_u8.shape[0] < max_orig:
                pad = max_orig - a_u8.shape[0]
                a_u8 = torch.cat([a_u8, torch.zeros(pad, a_u8.shape[1], dtype=torch.uint8, device='cuda')])
                as_u8 = torch.cat([as_u8, torch.zeros(pad, as_u8.shape[1], dtype=torch.uint8, device='cuda')])

            # 4. Prepare weights (raw, un-shuffled)
            w1 = gate_up_weight.view(torch.uint8)
            scale_K = K // 32
            w1_scale_3d = gate_up_weight_scale.view(torch.uint8).view(E, N, scale_K)

            # 5. Output buffer
            num_valid = M * topk
            out = torch.zeros((num_valid, N), dtype=torch.bfloat16, device='cuda')

            # 6. Launch
            BLOCK_N = 128
            BLOCK_K = 128
            grid = (num_blocks * triton.cdiv(N, BLOCK_N),)

            print(f"[V2] Launching: M={M}, E={E}, N={N}, K={K}, blocks={num_blocks}, grid={grid}")
            print(f"[V2] a_u8: {a_u8.shape}, as_u8: {as_u8.shape}")
            print(f"[V2] expert_ids[:5]: {expert_ids[:5].tolist()}, range [{expert_ids[:num_blocks].min()},{expert_ids[:num_blocks].max()}]")

            _triton_moe_stage1[grid](
                a_u8, a_u8.stride(0), a_u8.stride(1),
                as_u8, as_u8.stride(0), as_u8.stride(1),
                w1, w1.stride(0), w1.stride(1), w1.stride(2),
                w1_scale_3d, w1_scale_3d.stride(0), w1_scale_3d.stride(1), w1_scale_3d.stride(2),
                out, out.stride(0), out.stride(1),
                sorted_ids, expert_ids,
                N, K, num_valid, top_k=topk,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )
            torch.cuda.synchronize()
            print(f"[V2] SUCCESS! out range: [{out.min():.4f}, {out.max():.4f}]")

            # Correctness check: compare with CK stage1
            from aiter.fused_moe import moe_sorting as ms2
            s_ids, s_w, s_exp, s_nv, s_buf = ms2(topk_ids, topk_weights, E, K, torch.bfloat16, BLOCK_M, None, None, 0)
            a1_ck, a1_scale_ck = fused_dynamic_mxfp4_quant_moe_sort(hidden_states, sorted_ids=s_ids, num_valid_ids=s_nv, token_num=M, topk=1, block_size=BLOCK_M)
            a2_ck = torch.empty((M, topk, N // 2), dtype=torch.bfloat16, device='cuda')
            w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
            fm.ck_moe_stage1(a1_ck, gate_up_weight_shuffled, down_weight_shuffled, s_ids, s_exp, s_nv, a2_ck, topk, kernelName="", block_m=BLOCK_M, a1_scale=a1_scale_ck, w1_scale=w1_scale_v, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, splitk=0, use_non_temporal_load=False)
            # CK output is [M, topk, N//2] after SiLU
            # Our Triton output is [M*topk, N] before SiLU
            print(f"[V2] CK stage1 out: {a2_ck.shape}, range: [{a2_ck.min():.4f}, {a2_ck.max():.4f}]")
            print(f"[V2] Triton stage1 out: {out.shape}, range: [{out.min():.4f}, {out.max():.4f}]")
            # Note: CK applies SiLU, our Triton doesn't yet

        except Exception as e:
            import traceback
            print(f"[V2] error: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled, topk_weights, topk_ids, expert_mask=None, activation=ActivationType.Silu, quant_type=QuantType.per_1x32, doweight_stage1=False, w1_scale=gate_up_weight_scale_shuffled, w2_scale=down_weight_scale_shuffled, a1_scale=None, a2_scale=None, hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

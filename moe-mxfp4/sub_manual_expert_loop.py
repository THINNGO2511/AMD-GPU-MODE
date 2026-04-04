#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Manual expert loop PROBE using gemm_a16wfp4 per expert.

PURPOSE: Measure actual timing to validate/invalidate the manual loop approach.
Expected outcome: ~2000us total (7x worse than fused_moe 337us) due to:
1. 33 sequential kernel launches with small M (~140) = low GPU occupancy
2. Gather/scatter overhead per expert
3. Only 2 Triton JIT compilations (N,K pairs are fixed per stage)

APPROACH:
For the d=2048 bottleneck (E=33 bs=512, currently ~337us with fused_moe):
1. Sort tokens by expert (Python-side, using topk_ids)
2. For each expert: gather tokens, call gemm_a16wfp4 (stage1), SiLU, gemm_a16wfp4 (stage2)
3. Weighted scatter-add back to output

Uses RAW (unshuffled) weights and scales since gemm_a16wfp4 needs unshuffled E8M0.
For other shapes (d<2048), uses the proven CK fused_moe pipeline.
"""
import torch
import torch.nn.functional as F
import functools
import sys
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_initialized = False
_call_count = 0

# CK kernel names for d<2048
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _unshuffle_e8m0(scale_sh):
    """Unshuffle E8M0 scales for gemm_a16wfp4."""
    s = scale_sh.view(torch.uint8)
    sm, sn = s.shape
    s = s.view(sm // 32, sn // 8, 4, 16, 2, 2)
    s = s.permute(0, 5, 3, 1, 4, 2).contiguous()
    return s.view(sm, sn)


def _init_ck_patches():
    """Standard CK pipeline patches for d<2048 shapes."""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # use_nt=False globally
    fm.use_nt = lambda t, k, e: False

    # block_m tuning
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50:
                return 32
            elif inter_dim >= 2048 and est_m >= 100:
                return 128
            else:
                return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # CK kernel injection for E<=64 d<2048
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
                            kernelName=STAGE2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


def _manual_expert_moe(hidden_states, gate_up_weight, down_weight,
                       gate_up_weight_scale, down_weight_scale,
                       topk_weights, topk_ids, config):
    """
    Manual expert loop MoE using gemm_a16wfp4 per expert.

    For E=33 bs=512 d=2048:
      - 512 tokens, topk=9 => 4608 (token,expert) assignments
      - ~140 tokens per expert
      - Stage1: [~140, 7168] x [4096, 7168]^T = [~140, 4096]
      - Stage2: [~140, 2048] x [7168, 2048]^T = [~140, 7168]
    """
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    E = gate_up_weight.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]

    # Prepare raw weights as uint8 for gemm_a16wfp4
    # gate_up_weight: [E, 2*d_expert_pad, d_hidden_pad//2] fp4x2 (raw)
    # down_weight: [E, d_hidden_pad, d_expert_pad//2] fp4x2 (raw)
    # gate_up_weight_scale: [E, 2*d_expert_pad, d_hidden_pad//32] e8m0 (raw)
    # down_weight_scale: [E, d_hidden_pad, d_expert_pad//32] e8m0 (raw)

    # gemm_a16wfp4 needs: A[M,K] bf16, B[N,K//2] uint8, B_scale[N,K//32] uint8 (unshuffled)
    # gate_up: N=2*d_expert_pad, K=d_hidden_pad
    # down: N=d_hidden_pad, K=d_expert_pad

    device = hidden_states.device

    # Output accumulator
    output = torch.zeros(M, d_hidden_pad, dtype=torch.bfloat16, device=device)

    # Build per-expert token lists
    # topk_ids: [M, topk] => for each (token_i, slot_j), expert = topk_ids[i,j]
    # We need to gather tokens for each expert
    flat_ids = topk_ids.view(-1)  # [M*topk]
    flat_weights = topk_weights.view(-1)  # [M*topk]
    flat_token_idx = torch.arange(M, device=device).unsqueeze(1).expand(M, topk).reshape(-1)  # which original token

    for eid in range(E):
        # Find which (token, slot) pairs are assigned to this expert
        mask = (flat_ids == eid)
        if not mask.any():
            continue

        token_indices = flat_token_idx[mask]  # original token indices
        weights = flat_weights[mask]          # routing weights
        num_tokens = token_indices.shape[0]

        # Gather input tokens for this expert: [num_tokens, d_hidden]
        # Pad hidden_states to d_hidden_pad if needed
        x = hidden_states[token_indices]  # [num_tokens, d_hidden]
        if d_hidden_pad > d_hidden:
            x = F.pad(x, (0, d_hidden_pad - d_hidden))  # [num_tokens, d_hidden_pad]

        # Stage 1: gate_up projection
        # W1[eid]: [2*d_expert_pad, d_hidden_pad//2] fp4x2 => viewed as uint8
        w1_e = gate_up_weight[eid].view(torch.uint8)  # [2*d_expert_pad, d_hidden_pad//2]
        s1_e = gate_up_weight_scale[eid].view(torch.uint8)  # [2*d_expert_pad, d_hidden_pad//32]

        # gemm_a16wfp4(A, B_q_uint8, B_scale_unshuffled, dtype, y, config)
        # A: [num_tokens, d_hidden_pad] bf16
        # B: [2*d_expert_pad, d_hidden_pad//2] uint8
        # B_scale: [2*d_expert_pad, d_hidden_pad//32] uint8
        # Output: [num_tokens, 2*d_expert_pad] bf16
        gate_up_out = gemm_a16wfp4(x, w1_e, s1_e, dtype=torch.bfloat16)
        # gate_up_out: [num_tokens, 2*d_expert_pad]

        # SwiGLU: gate = SiLU(gate_up_out[:, :d_expert_pad]) * gate_up_out[:, d_expert_pad:]
        gate = gate_up_out[:, :d_expert_pad]
        up = gate_up_out[:, d_expert_pad:2*d_expert_pad]
        intermediate = F.silu(gate) * up  # [num_tokens, d_expert_pad]

        # Trim to actual d_expert if padded (for stage2 input)
        if d_expert_pad > d_expert:
            intermediate = intermediate[:, :d_expert]
            if d_expert_pad > d_expert:
                intermediate = F.pad(intermediate, (0, d_expert_pad - d_expert))

        # Stage 2: down projection
        # W2[eid]: [d_hidden_pad, d_expert_pad//2] fp4x2 => viewed as uint8
        w2_e = down_weight[eid].view(torch.uint8)  # [d_hidden_pad, d_expert_pad//2]
        s2_e = down_weight_scale[eid].view(torch.uint8)  # [d_hidden_pad, d_expert_pad//32]

        # A: [num_tokens, d_expert_pad] bf16
        # B: [d_hidden_pad, d_expert_pad//2] uint8
        # B_scale: [d_hidden_pad, d_expert_pad//32] uint8
        # Output: [num_tokens, d_hidden_pad] bf16
        expert_out = gemm_a16wfp4(intermediate, w2_e, s2_e, dtype=torch.bfloat16)
        # expert_out: [num_tokens, d_hidden_pad]

        # Weighted accumulation: output[token_i] += weight * expert_out
        output.index_add_(0, token_indices, expert_out * weights.unsqueeze(1))

    return output[:, :d_hidden]


def _manual_expert_moe_probe(hidden_states, gate_up_weight, down_weight,
                              gate_up_weight_scale, down_weight_scale,
                              topk_weights, topk_ids, config):
    """
    Probe version: measure timing of individual steps.
    """
    from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4
    import time

    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]
    E = gate_up_weight.shape[0]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]

    device = hidden_states.device

    print(f"\n[MANUAL] M={M}, E={E}, topk={topk}, d_hidden={d_hidden}, d_expert={d_expert}", file=sys.stderr)
    print(f"[MANUAL] d_hidden_pad={d_hidden_pad}, d_expert_pad={d_expert_pad}", file=sys.stderr)
    print(f"[MANUAL] W1 shape: {gate_up_weight.shape} {gate_up_weight.dtype}", file=sys.stderr)
    print(f"[MANUAL] W2 shape: {down_weight.shape} {down_weight.dtype}", file=sys.stderr)
    print(f"[MANUAL] W1_scale shape: {gate_up_weight_scale.shape} {gate_up_weight_scale.dtype}", file=sys.stderr)
    print(f"[MANUAL] W2_scale shape: {down_weight_scale.shape} {down_weight_scale.dtype}", file=sys.stderr)

    # Count tokens per expert
    flat_ids = topk_ids.view(-1)
    for eid in range(min(E, 5)):
        cnt = (flat_ids == eid).sum().item()
        print(f"[MANUAL] Expert {eid}: {cnt} tokens", file=sys.stderr)
    active_experts = 0
    max_tokens = 0
    min_tokens = 999999
    for eid in range(E):
        cnt = (flat_ids == eid).sum().item()
        if cnt > 0:
            active_experts += 1
            max_tokens = max(max_tokens, cnt)
            min_tokens = min(min_tokens, cnt)
    print(f"[MANUAL] Active experts: {active_experts}/{E}, tokens/expert: {min_tokens}-{max_tokens}", file=sys.stderr)

    # Time a single gemm_a16wfp4 call to estimate total
    w1_0 = gate_up_weight[0].view(torch.uint8)
    s1_0 = gate_up_weight_scale[0].view(torch.uint8)
    x_test = hidden_states[:min(max_tokens, M)]
    if d_hidden_pad > d_hidden:
        x_test = F.pad(x_test, (0, d_hidden_pad - d_hidden))

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    _ = gemm_a16wfp4(x_test, w1_0, s1_0, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    gemm1_us = (t1 - t0) / 1000
    print(f"[MANUAL] Single stage1 gemm_a16wfp4 ({x_test.shape[0]}x{d_hidden_pad} x {w1_0.shape[0]}x{w1_0.shape[1]}): {gemm1_us:.1f}us", file=sys.stderr)

    w2_0 = down_weight[0].view(torch.uint8)
    s2_0 = down_weight_scale[0].view(torch.uint8)
    x_test2 = torch.randn(min(max_tokens, M), d_expert_pad, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    _ = gemm_a16wfp4(x_test2, w2_0, s2_0, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    gemm2_us = (t1 - t0) / 1000
    print(f"[MANUAL] Single stage2 gemm_a16wfp4 ({x_test2.shape[0]}x{d_expert_pad} x {w2_0.shape[0]}x{w2_0.shape[1]}): {gemm2_us:.1f}us", file=sys.stderr)

    est_total = active_experts * (gemm1_us + gemm2_us)
    print(f"[MANUAL] Estimated total: {active_experts} experts x ({gemm1_us:.1f} + {gemm2_us:.1f}) = {est_total:.1f}us", file=sys.stderr)
    print(f"[MANUAL] vs fused_moe baseline: ~337us", file=sys.stderr)

    # Now do the actual computation with timing
    torch.cuda.synchronize()
    t_start = time.perf_counter_ns()

    result = _manual_expert_moe(hidden_states, gate_up_weight, down_weight,
                                 gate_up_weight_scale, down_weight_scale,
                                 topk_weights, topk_ids, config)

    torch.cuda.synchronize()
    t_end = time.perf_counter_ns()
    total_us = (t_end - t_start) / 1000
    print(f"[MANUAL] Actual total: {total_us:.1f}us", file=sys.stderr)
    sys.stderr.flush()

    return result


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _init_ck_patches()
    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    d_expert = config["d_expert"]
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # Only use manual expert loop for d=2048 (the bottleneck with no CK tuned config)
    if d_expert >= 2048:
        try:
            if _call_count <= 2:
                # First calls: probe with timing info
                return _manual_expert_moe_probe(
                    hidden_states, gate_up_weight, down_weight,
                    gate_up_weight_scale, down_weight_scale,
                    topk_weights, topk_ids, config)
            else:
                return _manual_expert_moe(
                    hidden_states, gate_up_weight, down_weight,
                    gate_up_weight_scale, down_weight_scale,
                    topk_weights, topk_ids, config)
        except Exception as e:
            if _call_count <= 3:
                import traceback
                print(f"[MANUAL] FAILED: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.stderr.flush()

    # Standard CK path for d<2048
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

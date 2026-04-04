#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
Minimal d=2048 probe. Answers 5 questions via stderr logs, then runs default fused_moe.
"""
import torch
import sys
import os
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

P = lambda *a, **k: print(*a, **k, file=sys.stderr, flush=True)
_done = False


def custom_kernel(data: input_t) -> output_t:
    global _done
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _done:
        _done = True
        _run_probe(data, hidden_pad, intermediate_pad)

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


def _run_probe(data, hidden_pad, intermediate_pad):
    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    topk = topk_ids.shape[1]
    d_hidden = config["d_hidden"]
    d_expert = config["d_expert"]
    d_hidden_pad = config["d_hidden_pad"]
    d_expert_pad = config["d_expert_pad"]

    P("\n" + "=" * 70)
    P("D2048 SIMPLE PROBE")
    P("=" * 70)

    # --- Q5: E8M0 scale shapes ---
    P(f"\n[Q5] SCALE SHAPES (d_hidden={d_hidden} d_expert={d_expert})")
    P(f"  gate_up_weight_scale:          {gate_up_weight_scale.shape} {gate_up_weight_scale.dtype}")
    P(f"  down_weight_scale:             {down_weight_scale.shape} {down_weight_scale.dtype}")
    P(f"  gate_up_weight_scale_shuffled: {gate_up_weight_scale_shuffled.shape} {gate_up_weight_scale_shuffled.dtype}")
    P(f"  down_weight_scale_shuffled:    {down_weight_scale_shuffled.shape} {down_weight_scale_shuffled.dtype}")
    P(f"  gate_up_weight_shuffled:       {gate_up_weight_shuffled.shape} {gate_up_weight_shuffled.dtype}")
    P(f"  down_weight_shuffled:          {down_weight_shuffled.shape} {down_weight_shuffled.dtype}")

    # Compute heuristic inputs
    try:
        _, model_dim, inter_dim = fm.get_inter_dim(
            gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    except Exception as e:
        model_dim, inter_dim = d_hidden_pad, d_expert_pad
        P(f"  get_inter_dim fallback: {e}")

    padded_M = fm.get_padded_M(M)
    est_m = M * topk // E
    isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
    q_dtype_a = torch.float4_e2m1fn_x2
    q_dtype_w = gate_up_weight_shuffled.dtype

    P(f"\n  M={M} E={E} topk={topk} padded_M={padded_M} est_m={est_m}")
    P(f"  model_dim={model_dim} inter_dim={inter_dim} isG1U1={isG1U1}")
    P(f"  q_dtype_a={q_dtype_a} q_dtype_w={q_dtype_w}")
    P(f"  hidden_pad={hidden_pad} intermediate_pad={intermediate_pad}")

    # --- Q2: block_m ---
    P(f"\n[Q2] BLOCK_M SELECTION")
    try:
        bm = fm.get_block_size_M(padded_M, topk, E, inter_dim)
        P(f"  get_block_size_M({padded_M},{topk},{E},{inter_dim}) = {bm}")
    except Exception as e:
        P(f"  ERROR: {e}")
    # Also test est_m=140 directly (bs=512, E=33, topk=9 => est_m~139)
    for test_M in [16, 32, 64, 128, 256, 512, 1024]:
        try:
            bm2 = fm.get_block_size_M(test_M, topk, E, inter_dim)
            em = test_M * topk // E
            P(f"  padded_M={test_M:>5d} est_m={em:>5d} => block_m={bm2}")
        except Exception as e:
            P(f"  padded_M={test_M}: {e}")

    # --- Q1 + Q4: get_2stage_cfgs kernel names ---
    P(f"\n[Q1] KERNEL SELECTION (doweight_stage1=False)")
    _query_cfgs(padded_M, model_dim, inter_dim, E, topk,
                q_dtype_a, q_dtype_w, isG1U1, hidden_pad, intermediate_pad,
                doweight=False)

    P(f"\n[Q4] KERNEL SELECTION (doweight_stage1=True)")
    _query_cfgs(padded_M, model_dim, inter_dim, E, topk,
                q_dtype_a, q_dtype_w, isG1U1, hidden_pad, intermediate_pad,
                doweight=True)

    # padded_M sweep for kernel transitions
    P(f"\n[Q1b] KERNEL vs padded_M sweep")
    for test_M in [16, 32, 64, 128, 256, 512, 1024, 2048]:
        _query_cfgs(test_M, model_dim, inter_dim, E, topk,
                    q_dtype_a, q_dtype_w, isG1U1, hidden_pad, intermediate_pad,
                    doweight=False, compact=True)

    # --- Q3: All kernel names in the compiled .so ---
    P(f"\n[Q3] COMPILED MODULE KERNEL NAMES")
    _list_module_kernels()

    # Also list .co files with FP4 for d=2048
    P(f"\n[Q3b] FP4 .co FILES in fmoe_2stages/")
    fmoe_dir = "/home/runner/aiter/hsa/gfx950/fmoe_2stages/"
    if os.path.isdir(fmoe_dir):
        fp4_files = sorted(f for f in os.listdir(fmoe_dir) if 'FP4' in f and f.endswith('.co'))
        P(f"  Total FP4 .co files: {len(fp4_files)}")
        for f in fp4_files:
            P(f"  {f}")
    else:
        P(f"  DIR NOT FOUND: {fmoe_dir}")

    P("\n" + "=" * 70)
    P("PROBE DONE")
    P("=" * 70)


def _query_cfgs(padded_M, model_dim, inter_dim, E, topk,
                q_dtype_a, q_dtype_w, isG1U1, hidden_pad, intermediate_pad,
                doweight=False, compact=False):
    try:
        meta = fm.get_2stage_cfgs(
            padded_M, model_dim, inter_dim, E, topk,
            torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
            isG1U1, ActivationType.Silu, doweight,
            hidden_pad, intermediate_pad, True)

        kn1 = kn2 = ""
        if not meta.run_1stage:
            if hasattr(meta.stage1, 'keywords'):
                kn1 = meta.stage1.keywords.get('kernelName', '')
            if hasattr(meta.stage2, 'keywords'):
                kn2 = meta.stage2.keywords.get('kernelName', '')

        if compact:
            em = padded_M * topk // E
            P(f"  pM={padded_M:>5d} em={em:>5d} 1stg={meta.run_1stage} bm={meta.block_m}"
              f" s1={kn1[-60:] if kn1 else 'EMPTY'}")
        else:
            P(f"  run_1stage={meta.run_1stage} block_m={meta.block_m}")
            P(f"  splitk={getattr(meta, 'splitk', 'N/A')} use_nt={getattr(meta, 'use_nt', 'N/A')}")
            P(f"  STAGE1 kernel: {kn1 if kn1 else 'EMPTY'}")
            P(f"  STAGE2 kernel: {kn2 if kn2 else 'EMPTY'}")
            # Dump all stage1/stage2 keywords
            if not meta.run_1stage:
                for label, stage in [("STAGE1", meta.stage1), ("STAGE2", meta.stage2)]:
                    if hasattr(stage, 'keywords'):
                        for k, v in stage.keywords.items():
                            if k != 'kernelName':
                                P(f"    {label} {k}={v}")
                    if hasattr(stage, 'func'):
                        P(f"    {label} func={stage.func}")
    except Exception as e:
        if compact:
            P(f"  pM={padded_M}: ERROR {e}")
        else:
            import traceback
            P(f"  ERROR: {e}")
            traceback.print_exc(file=sys.stderr)


def _list_module_kernels():
    """List all functions/kernels available in aiter's compiled MoE modules."""
    # Check torch.ops.aiter for MoE-related ops
    try:
        ops = [x for x in dir(torch.ops.aiter) if 'moe' in x.lower() or 'fmoe' in x.lower()]
        P(f"  torch.ops.aiter MoE ops ({len(ops)}): {ops}")
    except Exception as e:
        P(f"  torch.ops.aiter: {e}")

    # Check aiter module for ck_moe/fmoe functions
    ck_fns = [x for x in dir(aiter) if 'moe' in x.lower() or 'fmoe' in x.lower()]
    P(f"  aiter MoE functions ({len(ck_fns)}): {ck_fns}")

    # Check fm module
    fm_fns = [x for x in dir(fm) if not x.startswith('__')]
    P(f"  fm public attrs ({len(fm_fns)}): {fm_fns}")

    # Inspect ck_moe_stage1 and ck_moe_stage2_fwd signatures
    for name in ['ck_moe_stage1', 'ck_moe_stage2_fwd']:
        fn = getattr(fm, name, None) or getattr(aiter, name, None)
        if fn:
            try:
                sig = inspect.signature(fn)
                P(f"  {name} sig: {sig}")
            except Exception as e:
                P(f"  {name} sig error: {e}")

    # MOEMetadata fields
    try:
        sig = inspect.signature(fm.MOEMetadata)
        P(f"  MOEMetadata sig: {sig}")
    except Exception as e:
        P(f"  MOEMetadata: {e}")

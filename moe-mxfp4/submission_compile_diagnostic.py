#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE -- torch.compile DIAGNOSTIC: Prove graph breaks occur.

This submission:
1. Runs torch.compile with TORCH_LOGS="graph_breaks" to capture graph break info
2. Uses torch._dynamo.explain() to get a detailed breakdown
3. Measures compile overhead vs eager baseline
4. Prints all findings, then falls back to proven optimized path

Purpose: Generate evidence for whether torch.compile can ever help this pipeline.
"""
import torch
import torch._inductor.config as inductor_config
import torch._dynamo
import functools
import sys
import time
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

# ROCm-safe config
inductor_config.triton.cudagraphs = False
inductor_config.triton.cudagraph_trees = False
inductor_config.max_autotune = False
inductor_config.memory_planning = False

_patched = False
_diagnosed = False
_call_count = 0

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    fm.use_nt = lambda token, topk, expert: False
    if hasattr(aiter, 'moe_sorting_opus_fwd'):
        fm._USE_OPUS_MOE_SORTING = True
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            if est_m < 50: return 32
            elif inter_dim >= 2048 and est_m >= 100: return 128
            else: return 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def new_g2s(token, model_dim, inter_dim, expert, topk,
                dtype, qa, qw, qt, g1, act, dw, hp, ip, sh=True):
        r = orig(token, model_dim, inter_dim, expert, topk,
                 dtype, qa, qw, qt, g1, act, dw, hp, ip, sh)
        if expert <= 64 and qt == QuantType.per_1x32 and not r.run_1stage and inter_dim < 2048:
            try:
                kw = r.stage1.keywords if hasattr(r.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est = token * topk // expert
                    kn = S1_256 if est >= 100 else S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn, activation=act,
                            quant_type=qt, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_V1, activation=act,
                            quant_type=qt, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
                pass
        return r
    fm.get_2stage_cfgs = new_g2s
    fm.cfg_2stages = None


def _call_moe(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s):
    return fused_moe(
        hidden_states, w1, w2, tw, ti,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=w1s, w2_scale=w2s,
        a1_scale=None, a2_scale=None,
        hidden_pad=hp, intermediate_pad=ip,
    )


def _diagnose(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s):
    """Run torch.compile diagnostics on first call only."""
    global _diagnosed
    if _diagnosed:
        return
    _diagnosed = True

    print("=" * 60, flush=True)
    print("=== torch.compile DIAGNOSTIC for MoE ===", flush=True)
    print("=" * 60, flush=True)

    # 1. Check if torch.compile is available
    print(f"\ntorch version: {torch.__version__}", flush=True)
    print(f"ROCm HIP: {torch.version.hip}", flush=True)
    print(f"torch._dynamo available: {hasattr(torch, '_dynamo')}", flush=True)

    # 2. Use explain() to find graph breaks
    print("\n--- torch._dynamo.explain() ---", flush=True)
    try:
        explanation = torch._dynamo.explain(
            _call_moe,
            hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s,
        )
        print(f"Graph count: {explanation.graph_count}", flush=True)
        print(f"Graph break count: {explanation.graph_break_count}", flush=True)
        print(f"Op count: {explanation.op_count}", flush=True)

        if hasattr(explanation, 'break_reasons'):
            print(f"\nBreak reasons ({len(explanation.break_reasons)}):", flush=True)
            for i, reason in enumerate(explanation.break_reasons[:10]):
                print(f"  {i}: {reason}", flush=True)

        if hasattr(explanation, 'graphs'):
            print(f"\nGraphs ({len(explanation.graphs)}):", flush=True)
            for i, g in enumerate(explanation.graphs[:5]):
                gstr = str(g)[:300]
                print(f"  Graph {i}: {gstr}", flush=True)

        # Print full explanation string
        expl_str = str(explanation)
        print(f"\nFull explanation ({len(expl_str)} chars):", flush=True)
        for line in expl_str.split('\n')[:30]:
            print(f"  {line}", flush=True)

    except Exception as e:
        print(f"explain() FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 3. Time eager vs compiled
    print("\n--- Timing comparison ---", flush=True)
    try:
        # Eager baseline
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            _call_moe(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s)
            torch.cuda.synchronize()
        t_eager = (time.perf_counter() - t0) / 5

        # Compiled (default mode, ROCm-safe)
        compiled_fn = torch.compile(_call_moe, mode="default", fullgraph=False, dynamic=False)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        compiled_fn(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s)
        torch.cuda.synchronize()
        t_first_compile = time.perf_counter() - t0

        # Compiled steady-state
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            compiled_fn(hidden_states, w1, w2, tw, ti, hp, ip, w1s, w2s)
            torch.cuda.synchronize()
        t_compiled = (time.perf_counter() - t0) / 5

        print(f"Eager avg:           {t_eager*1e6:.1f}us", flush=True)
        print(f"Compiled first call: {t_first_compile*1e6:.1f}us", flush=True)
        print(f"Compiled steady:     {t_compiled*1e6:.1f}us", flush=True)
        print(f"Speedup:             {t_eager/t_compiled:.3f}x", flush=True)
    except Exception as e:
        print(f"Timing FAILED: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 4. Check registered custom ops
    print("\n--- Custom ops in torch.ops.aiter ---", flush=True)
    try:
        aiter_ops = [n for n in dir(torch.ops.aiter) if not n.startswith('_')]
        moe_ops = [n for n in aiter_ops if 'moe' in n.lower() or 'fused' in n.lower()]
        print(f"Total aiter ops: {len(aiter_ops)}", flush=True)
        print(f"MoE-related ops: {moe_ops}", flush=True)

        # Check if any have fake implementations
        for op_name in moe_ops[:5]:
            try:
                op = getattr(torch.ops.aiter, op_name)
                has_fake = hasattr(op, 'default') and hasattr(op.default, '_abstract_fn')
                print(f"  {op_name}: has_fake={has_fake}", flush=True)
            except Exception as e2:
                print(f"  {op_name}: error={e2}", flush=True)
    except Exception as e:
        print(f"Custom ops check FAILED: {e}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("=== END DIAGNOSTIC ===", flush=True)
    print("=" * 60, flush=True)
    sys.stdout.flush()


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()

    (
        hidden_states, _, _, _, _,
        w1_sh, w2_sh, w1_scale_sh, w2_scale_sh,
        topk_weights, topk_ids, config,
    ) = data

    hp = config["d_hidden_pad"] - config["d_hidden"]
    ip = config["d_expert_pad"] - config["d_expert"]

    _call_count += 1
    if _call_count == 1:
        _diagnose(hidden_states, w1_sh, w2_sh,
                  topk_weights, topk_ids, hp, ip,
                  w1_scale_sh, w2_scale_sh)

    # Always use eager path for actual benchmark (compile won't help)
    return _call_moe(
        hidden_states, w1_sh, w2_sh,
        topk_weights, topk_ids, hp, ip,
        w1_scale_sh, w2_scale_sh,
    )

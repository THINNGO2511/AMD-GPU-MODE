"""
MoE — Try CK kernel injection for ALL cases including d=2048.
Also probe: available kernel names, Triton moe_mxfp4_silu calling convention,
and proper profiling with warmup.
"""
import torch
import functools
import time
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0
_timings = {}

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe():
    """Probe available kernels and Triton MoE calling convention."""
    print("\n=== KERNEL PROBE ===", flush=True)

    # 1. Read fused_mxfp4_quant.py to understand BLOCK_SIZE_M constraint
    try:
        path = '/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py'
        content = open(path).read()
        lines = content.split('\n')
        # Find the assertion and BLOCK_SIZE_M definition
        for i, line in enumerate(lines):
            if 'BLOCK_SIZE_M' in line and ('=' in line or 'assert' in line):
                print(f"  quant BLOCK_SIZE_M L{i+1}: {line.strip()}", flush=True)
        # Find fused_dynamic_mxfp4_quant_moe_sort function
        for i, line in enumerate(lines):
            if 'def fused_dynamic_mxfp4_quant_moe_sort' in line:
                print(f"\n--- fused_dynamic_mxfp4_quant_moe_sort (L{i+1}) ---", flush=True)
                for j in range(i, min(i+50, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break
    except Exception as e:
        print(f"  quant error: {e}", flush=True)

    # 2. Read the Triton MoE MXFP4 kernel (moe_op_mxfp4.py) — launch function
    try:
        path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4.py'
        content = open(path).read()
        lines = content.split('\n')
        # Print lines 75-end to see the kernel launch
        print(f"\n--- moe_op_mxfp4.py launch (L75+) ---", flush=True)
        for i in range(74, min(len(lines), 180)):
            print(f"  L{i+1}: {lines[i]}", flush=True)
    except Exception as e:
        print(f"  moe_op_mxfp4 error: {e}", flush=True)

    # 3. Read moe_op_mxfp4_silu_fused.py launch function
    try:
        path = '/home/runner/aiter/aiter/ops/triton/moe/moe_op_mxfp4_silu_fused.py'
        content = open(path).read()
        lines = content.split('\n')
        print(f"\n--- moe_op_mxfp4_silu_fused.py launch (L75+) ---", flush=True)
        for i in range(74, min(len(lines), 200)):
            print(f"  L{i+1}: {lines[i]}", flush=True)
    except Exception as e:
        print(f"  moe_op_mxfp4_silu error: {e}", flush=True)

    # 4. Read the Triton kernel source (_triton_kernels/moe/moe_op_mxfp4_silu_fused.py)
    try:
        import os
        kernel_dir = '/home/runner/aiter/aiter/ops/triton/_triton_kernels/moe/'
        if os.path.isdir(kernel_dir):
            files = os.listdir(kernel_dir)
            print(f"\n--- _triton_kernels/moe/ files ---", flush=True)
            for f in sorted(files):
                print(f"  {f}", flush=True)
    except Exception as e:
        print(f"  kernel dir error: {e}", flush=True)

    # 5. Check what the quant function does — read source for num_stages
    try:
        path = '/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py'
        content = open(path).read()
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'num_stages' in line or 'num_warps' in line:
                print(f"  quant config L{i+1}: {line.strip()}", flush=True)
    except:
        pass

    # 6. Read fused_moe_2stages function signature and first 30 lines
    try:
        src = open('/home/runner/aiter/aiter/fused_moe.py').read()
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'def fused_moe_2stages' in line:
                print(f"\n--- fused_moe_2stages (L{i+1}) ---", flush=True)
                for j in range(i, min(i+30, len(lines))):
                    print(f"  L{j+1}: {lines[j]}", flush=True)
                break
    except:
        pass

    print("\n=== END PROBE ===\n", flush=True)


def _make_timer(name, orig_fn):
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = orig_fn(*args, **kwargs)
        end.record()
        if name not in _timings:
            _timings[name] = []
        _timings[name].append((start, end))
        return result
    return wrapper


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe()

    # Patch use_nt: disable for E<=64
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    # Patch block_m: 32 min (block_m=16 fails quant assertion)
    orig_bsm = fm.get_block_size_M
    def new_bsm(token, topk, expert, inter_dim):
        if expert <= 64:
            est_m = token * topk // expert
            return 32 if est_m < 50 else 64
        return orig_bsm(token, topk, expert, inter_dim)
    fm.get_block_size_M = new_bsm

    # Patch get_2stage_cfgs: inject for ALL E<=64 cases (including d=2048)
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
                and not result.run_1stage):
            try:
                kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
                if not kw.get('kernelName', ''):
                    est_m = token * topk // expert
                    kn1 = STAGE1_256 if est_m >= 100 else STAGE1_64
                    print(f"[INJECT] E={expert} d={inter_dim} token={token} est_m={est_m} -> {kn1.split('_')[3]}", flush=True)
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=STAGE2_32, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception as e:
                print(f"[INJECT ERR] {e}", flush=True)
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None

    # Instrument for profiling
    fm.moe_sorting = _make_timer("sort", fm.moe_sorting)
    fm.ck_moe_stage1 = _make_timer("stage1", fm.ck_moe_stage1)
    orig_s2 = aiter.ck_moe_stage2_fwd
    aiter.ck_moe_stage2_fwd = _make_timer("stage2", orig_s2)
    if hasattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort'):
        fm.fused_dynamic_mxfp4_quant_moe_sort = _make_timer("quant", fm.fused_dynamic_mxfp4_quant_moe_sort)


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    M = hidden_states.shape[0]
    E = gate_up_weight_shuffled.shape[0]
    d = config.get("d_expert", "?")

    _timings.clear()

    result = fused_moe(
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

    _call_count += 1
    if _call_count <= 21:
        torch.cuda.synchronize()
        resolved = {}
        for name, events in _timings.items():
            resolved[name] = sum(s.elapsed_time(e) for s, e in events)
        total = sum(resolved.values())
        parts = []
        for name in ["sort", "quant", "stage1", "stage2"]:
            if name in resolved:
                t = resolved[name]
                pct = 100*t/total if total > 0 else 0
                parts.append(f"{name}={t:.1f}us({pct:.0f}%)" if t < 1 else f"{name}={t*1000:.0f}us({pct:.0f}%)")
        print(f"[PROF] #{_call_count} M={M} E={E} d={d}: {' | '.join(parts)} | total={total*1000:.0f}us", flush=True)

    return result

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe CU count + CSV config matching.
Hypothesis: runner CU count != 256 → CSV configs don't match → heuristic fallback.
"""
import sys
import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # === PROBE: CU count ===
    try:
        cu_num = fm.get_cu_num()
        print(f"[CU] fm.get_cu_num() = {cu_num}", file=sys.stderr)
    except Exception as e:
        print(f"[CU] fm.get_cu_num() error: {e}", file=sys.stderr)

    try:
        cu_num2 = aiter.get_cu_num() if hasattr(aiter, 'get_cu_num') else "N/A"
        print(f"[CU] aiter.get_cu_num() = {cu_num2}", file=sys.stderr)
    except Exception as e:
        print(f"[CU] aiter.get_cu_num() error: {e}", file=sys.stderr)

    # hipGetDeviceProperties CU count
    try:
        props = torch.cuda.get_device_properties(0)
        print(f"[CU] torch CU count: {props.multi_processor_count}", file=sys.stderr)
        print(f"[CU] device name: {props.name}", file=sys.stderr)
    except Exception as e:
        print(f"[CU] torch props error: {e}", file=sys.stderr)

    # === PROBE: CSV config loading ===
    import inspect
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        # Find cu_num references
        print(f"\n[CU] get_2stage_cfgs cu_num refs:", file=sys.stderr)
        for i, line in enumerate(lines):
            if 'cu_num' in line.lower() or 'cu_count' in line.lower() or 'get_cu' in line.lower():
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    print(f"  L{j+1}: {lines[j]}", file=sys.stderr)
                print(f"  ---", file=sys.stderr)
        # Find config lookup logic
        print(f"\n[CU] get_2stage_cfgs config lookup:", file=sys.stderr)
        for i, line in enumerate(lines):
            if 'cfg' in line and ('keys' in line or 'lookup' in line or 'get(' in line or 'INDEX' in line):
                print(f"  L{j+1}: {lines[j]}", file=sys.stderr)
    except Exception as e:
        print(f"[CU] source error: {e}", file=sys.stderr)

    # === PROBE: What configs actually load for our benchmark sizes ===
    # Intercept get_2stage_cfgs to log what it returns
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__

    @functools.lru_cache(maxsize=2048)
    def logging_get_2stage(token, model_dim, inter_dim, expert, topk,
                           dtype, q_dtype_a, q_dtype_w, q_type,
                           use_g1u1, activation, doweight_stage1,
                           hidden_pad, intermediate_pad, is_shuffled=True):
        result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                                dtype, q_dtype_a, q_dtype_w, q_type,
                                use_g1u1, activation, doweight_stage1,
                                hidden_pad, intermediate_pad, is_shuffled)
        # Log what config was selected
        s1_kn = ""
        s2_kn = ""
        if hasattr(result.stage1, 'keywords'):
            s1_kn = result.stage1.keywords.get('kernelName', '')
        if hasattr(result.stage2, 'keywords'):
            s2_kn = result.stage2.keywords.get('kernelName', '')
        print(f"[CFG] token={token} E={expert} d={inter_dim} block_m={result.block_m} "
              f"ksplit={result.ksplit} 1stage={result.run_1stage}", file=sys.stderr)
        print(f"[CFG]   stage1: {s1_kn[:80]}", file=sys.stderr)
        print(f"[CFG]   stage2: {s2_kn[:80]}", file=sys.stderr)

        # For E<=64: inject CK kernels
        if (expert <= 64 and q_type == QuantType.per_1x32
                and not result.run_1stage and inter_dim < 2048):
            try:
                if not s1_kn:
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

    fm.get_2stage_cfgs = logging_get_2stage
    fm.cfg_2stages = None

    # Patch use_nt
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)

    orig_bsm = fm.get_block_size_M
    def new_bsm(t, k, e, d):
        if e <= 64:
            est_m = t * k // e
            return 32 if est_m < 50 else 64
        return orig_bsm(t, k, e, d)
    fm.get_block_size_M = new_bsm

    print(f"\n[CU] All patches applied", file=sys.stderr)


def custom_kernel(data: input_t) -> output_t:
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
    return fused_moe(
        hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
        topk_weights, topk_ids,
        expert_mask=None, activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32, doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

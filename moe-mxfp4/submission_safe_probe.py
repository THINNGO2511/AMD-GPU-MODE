"""
MoE — Safe probe of flydsl, cktile, get_2stage_cfgs internals.
NO file modifications. Just read and report.
"""
import torch
import functools
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe():
    print("=== SAFE PROBE ===", flush=True)

    # 1. flydsl
    try:
        avail = fm.is_flydsl_available()
        print(f"flydsl available: {avail}", flush=True)
        if avail:
            src = inspect.getsource(fm._flydsl_stage2_wrapper)
            for i, line in enumerate(src.split('\n')[:25]):
                print(f"  flydsl L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"flydsl: {e}", flush=True)

    # 2. get_2stage_cfgs source
    try:
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\nget_2stage_cfgs ({len(lines)} lines):", flush=True)
        for i, line in enumerate(lines[:100]):
            print(f"  L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"get_2stage_cfgs: {e}", flush=True)

    # 3. cktile source
    try:
        if hasattr(fm, 'cktile_moe_stage1'):
            src = inspect.getsource(fm.cktile_moe_stage1)
            print(f"\ncktile_moe_stage1:", flush=True)
            for i, line in enumerate(src.split('\n')[:20]):
                print(f"  L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"cktile: {e}", flush=True)

    # 4. asm_stage1 source
    try:
        if hasattr(fm, 'asm_stage1'):
            src = inspect.getsource(fm.asm_stage1)
            print(f"\nasm_stage1:", flush=True)
            for i, line in enumerate(src.split('\n')[:20]):
                print(f"  L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"asm_stage1: {e}", flush=True)

    # 5. torch.ops.aiter MoE-related ops
    try:
        moe_ops = [n for n in dir(torch.ops.aiter) if 'moe' in n.lower()]
        print(f"\ntorch.ops.aiter MoE ops: {moe_ops}", flush=True)
    except Exception as e:
        print(f"ops error: {e}", flush=True)

    print("=== END ===\n", flush=True)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    _probe()
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)
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
            except:
                pass
        return result
    fm.get_2stage_cfgs = new_get_2stage
    fm.cfg_2stages = None


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

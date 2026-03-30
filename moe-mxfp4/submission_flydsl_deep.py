"""
MoE — Deep probe of FlyDSL kernels and get_2stage_cfgs L100-350.
FlyDSL is available — need to understand how to use it.
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
    print("=== FLYDSL DEEP PROBE ===", flush=True)

    # 1. get_2stage_cfgs lines 100-350
    try:
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"get_2stage_cfgs L100-250:", flush=True)
        for i in range(99, min(250, len(lines))):
            print(f"  L{i+1}: {lines[i]}", flush=True)
    except Exception as e:
        print(f"get_2stage_cfgs error: {e}", flush=True)

    # 2. FlyDSL module contents
    try:
        import aiter.ops.flydsl as flydsl
        attrs = [a for a in dir(flydsl) if not a.startswith('__')]
        print(f"\naiter.ops.flydsl attrs: {attrs}", flush=True)

        if hasattr(flydsl, 'moe_kernels'):
            mk_attrs = [a for a in dir(flydsl.moe_kernels) if not a.startswith('__')]
            print(f"  moe_kernels attrs: {mk_attrs}", flush=True)

            if hasattr(flydsl.moe_kernels, 'get_flydsl_kernel_params'):
                src = inspect.getsource(flydsl.moe_kernels.get_flydsl_kernel_params)
                for i, line in enumerate(src.split('\n')[:30]):
                    print(f"  gfkp L{i+1}: {line}", flush=True)

            # List available FlyDSL kernel names
            if hasattr(flydsl.moe_kernels, 'FLYDSL_KERNELS'):
                print(f"  FLYDSL_KERNELS: {flydsl.moe_kernels.FLYDSL_KERNELS}", flush=True)
            if hasattr(flydsl.moe_kernels, 'get_available_kernels'):
                kernels = flydsl.moe_kernels.get_available_kernels()
                print(f"  available kernels: {kernels}", flush=True)

        if hasattr(flydsl, 'flydsl_moe_stage2'):
            print(f"\n  flydsl_moe_stage2 type: {type(flydsl.flydsl_moe_stage2)}", flush=True)
            try:
                sig = inspect.signature(flydsl.flydsl_moe_stage2)
                print(f"  sig: {sig}", flush=True)
            except:
                pass
    except Exception as e:
        print(f"flydsl module error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 3. Check CSV for flydsl_fallback entries
    try:
        import os
        csv_paths = [
            '/home/runner/aiter/aiter/configs/tuned_fmoe.csv',
            '/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv',
        ]
        for csv_path in csv_paths:
            if os.path.exists(csv_path):
                with open(csv_path) as f:
                    all_lines = f.readlines()
                print(f"\n{csv_path} ({len(all_lines)} lines):", flush=True)
                # Print header
                print(f"  H: {all_lines[0].strip()}", flush=True)
                # Find flydsl entries
                for line in all_lines[1:]:
                    if 'flydsl' in line.lower():
                        print(f"  FLY: {line.strip()[:200]}", flush=True)
                # Find entries for E=33
                for line in all_lines[1:]:
                    if ',33,' in line:
                        print(f"  E33: {line.strip()[:200]}", flush=True)
    except Exception as e:
        print(f"CSV error: {e}", flush=True)

    # 4. Check _flydsl_stage2_wrapper full source
    try:
        src = inspect.getsource(fm._flydsl_stage2_wrapper)
        lines = src.split('\n')
        print(f"\n_flydsl_stage2_wrapper ({len(lines)} lines):", flush=True)
        for i, line in enumerate(lines):
            print(f"  L{i+1}: {line}", flush=True)
    except Exception as e:
        print(f"flydsl wrapper error: {e}", flush=True)

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

"""
MoE — Try FlyDSL kernels for stage1 + stage2.
FlyDSL is available and may be faster than CK.
"""
import torch
import functools
import inspect
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False

STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe_flydsl_kernels():
    """Enumerate ALL available FlyDSL kernel names and their params."""
    print("=== FLYDSL KERNEL ENUMERATION ===", flush=True)
    try:
        import aiter.ops.flydsl as flydsl
        mk = flydsl.moe_kernels

        # 1. List _KERNEL_PARAMS
        print(f"_KERNEL_PARAMS has {len(mk._KERNEL_PARAMS)} entries:", flush=True)
        for name, params in sorted(mk._KERNEL_PARAMS.items()):
            print(f"  {name}: {params}", flush=True)

        # 2. flydsl_kernel_name function
        if hasattr(mk, 'flydsl_kernel_name'):
            src = inspect.getsource(mk.flydsl_kernel_name)
            print(f"\nflydsl_kernel_name:", flush=True)
            for i, line in enumerate(src.split('\n')[:15]):
                print(f"  L{i+1}: {line}", flush=True)

        # 3. _register_all_configs
        if hasattr(mk, '_register_all_configs'):
            src = inspect.getsource(mk._register_all_configs)
            print(f"\n_register_all_configs:", flush=True)
            for i, line in enumerate(src.split('\n')[:40]):
                print(f"  L{i+1}: {line}", flush=True)

        # 4. get_flydsl_stage1_kernels / get_flydsl_stage2_kernels
        for fn_name in ['get_flydsl_stage1_kernels', 'get_flydsl_stage2_kernels']:
            if hasattr(mk, fn_name):
                fn = getattr(mk, fn_name)
                try:
                    result = fn()
                    print(f"\n{fn_name}(): {result}", flush=True)
                except Exception as e:
                    print(f"{fn_name}() error: {e}", flush=True)

        # 5. Try compile_flydsl_moe_stage1/stage2
        for fn_name in ['compile_flydsl_moe_stage1', 'compile_flydsl_moe_stage2']:
            if hasattr(mk, fn_name):
                src = inspect.getsource(getattr(mk, fn_name))
                print(f"\n{fn_name}:", flush=True)
                for i, line in enumerate(src.split('\n')[:30]):
                    print(f"  L{i+1}: {line}", flush=True)

        # 6. flydsl_moe_stage1 signature
        if hasattr(flydsl, 'flydsl_moe_stage1'):
            sig = inspect.signature(flydsl.flydsl_moe_stage1)
            print(f"\nflydsl_moe_stage1 sig: {sig}", flush=True)

        # 7. Try generating kernel name for our config
        if hasattr(mk, 'flydsl_kernel_name'):
            for stage in [1, 2]:
                for tile_m in [32, 64]:
                    for tile_n in [64, 128]:
                        for tile_k in [128, 256]:
                            for a_dt in ['fp4', 'fp8']:
                                try:
                                    name = mk.flydsl_kernel_name(
                                        stage=stage, tile_m=tile_m, tile_n=tile_n,
                                        tile_k=tile_k, a_dtype=a_dt, b_dtype='fp4',
                                        out_dtype='bf16')
                                    params = mk.get_flydsl_kernel_params(name)
                                    if params:
                                        print(f"  valid: {name} -> {params}", flush=True)
                                except:
                                    pass
    except Exception as e:
        print(f"flydsl enum error: {e}", flush=True)
        import traceback
        traceback.print_exc()

    # 8. Read get_2stage_cfgs lines 250-350 (where FlyDSL stage2 is injected)
    try:
        orig = fm.get_2stage_cfgs
        if hasattr(orig, '__wrapped__'):
            orig = orig.__wrapped__
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\nget_2stage_cfgs L250-350:", flush=True)
        for i in range(249, min(len(lines), 350)):
            print(f"  L{i+1}: {lines[i]}", flush=True)
    except Exception as e:
        print(f"g2sc error: {e}", flush=True)

    print("=== END ===\n", flush=True)


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    _probe_flydsl_kernels()
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

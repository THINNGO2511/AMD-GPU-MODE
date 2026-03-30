#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
Deep probe: dump everything we need to understand optimization opportunities.
1. Per-phase CUDA-event timing for ALL benchmark cases
2. Configs selected for each case (kernel names, block_m, etc)
3. fused_moe_2stages source code
4. Available kernel names from CK
5. moe_sorting internals
6. Whether block_m=16 is feasible
7. Triton MoE kernel availability
"""
import torch
import functools
import inspect
import os
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_call_count = 0
_probed_source = False
_patched = False

# Best known kernel names
STAGE1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
STAGE2_32 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"


def _probe_source():
    """Dump critical source code sections."""
    global _probed_source
    if _probed_source:
        return
    _probed_source = True

    # 1. fused_moe_2stages source
    print("=" * 60)
    print("[SRC] fused_moe_2stages source:")
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        for i, line in enumerate(src.split('\n')[:80]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. moe_sorting source
    print("\n[SRC] moe_sorting source:")
    try:
        src = inspect.getsource(fm.moe_sorting)
        for i, line in enumerate(src.split('\n')[:40]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. get_2stage_cfgs source (original)
    print("\n[SRC] get_2stage_cfgs source:")
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        for i, line in enumerate(src.split('\n')[:80]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4. use_nt source
    print("\n[SRC] use_nt source:")
    try:
        src = inspect.getsource(fm.use_nt)
        for line in src.split('\n')[:20]:
            print(f"  {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 5. get_block_size_M source
    print("\n[SRC] get_block_size_M source:")
    try:
        src = inspect.getsource(fm.get_block_size_M)
        for line in src.split('\n')[:20]:
            print(f"  {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 6. Available CK kernel names (search in files)
    print("\n[KERNELS] Searching for available MoE CK kernels...")
    try:
        import glob
        csv_files = glob.glob("/home/runner/aiter/**/dsv3*.csv", recursive=True)
        csv_files += glob.glob("/home/runner/aiter/**/*fmoe*.csv", recursive=True)
        csv_files += glob.glob("/home/runner/aiter/**/*moe*tuned*.csv", recursive=True)
        print(f"  CSV files found: {csv_files}")
        for cf in csv_files[:3]:
            with open(cf) as f:
                lines = f.readlines()
            print(f"\n  {cf} ({len(lines)} lines):")
            for line in lines[:20]:
                print(f"    {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 7. Check for MOEMetadata class details
    print("\n[META] MOEMetadata:")
    try:
        src = inspect.getsource(fm.MOEMetadata)
        for line in src.split('\n')[:20]:
            print(f"  {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 8. fused_dynamic_mxfp4_quant_moe_sort source
    print("\n[SRC] fused_dynamic_mxfp4_quant_moe_sort:")
    try:
        src = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
        for i, line in enumerate(src.split('\n')[:40]):
            print(f"  {i}: {line}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 9. Check aiter version and available MoE-related modules
    print("\n[VER] aiter version info:")
    try:
        print(f"  __version__: {getattr(aiter, '__version__', 'N/A')}")
        print(f"  __file__: {aiter.__file__}")
    except:
        pass

    # 10. Search for Triton MoE kernels
    print("\n[TRITON] MoE Triton kernels:")
    try:
        import glob
        triton_files = glob.glob("/home/runner/aiter/**/moe_op_mxfp4*.py", recursive=True)
        triton_files += glob.glob("/home/runner/aiter/**/moe_op_e2e*.py", recursive=True)
        print(f"  Files: {triton_files}")
        for tf in triton_files[:2]:
            with open(tf) as f:
                lines = f.readlines()
            print(f"\n  {tf} ({len(lines)} lines):")
            # Print function signatures
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    print(f"    {i}: {line.rstrip()}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 11. Check for _USE_OPUS_MOE_SORTING
    print("\n[SORT] Opus sorting:")
    try:
        val = getattr(fm, '_USE_OPUS_MOE_SORTING', 'NOT_FOUND')
        print(f"  _USE_OPUS_MOE_SORTING = {val}")
    except:
        pass
    try:
        # Check if opus_sort or opus_moe_sort exists
        for name in dir(fm):
            if 'opus' in name.lower() or 'sort' in name.lower():
                print(f"  fm.{name}")
        for name in dir(aiter):
            if 'opus' in name.lower() or ('sort' in name.lower() and 'moe' in name.lower()):
                print(f"  aiter.{name}")
    except:
        pass

    # 12. Check token_num_quant_moe_sort_switch
    print("\n[THRESH] Quant/sort switch threshold:")
    try:
        for name in ['token_num_quant_moe_sort_switch', 'FUSED_QUANT_SORT_THRESHOLD']:
            val = getattr(fm, name, None)
            if val is not None:
                print(f"  fm.{name} = {val}")
    except:
        pass
    # Search in source
    try:
        src = inspect.getsource(fm.fused_moe_2stages)
        for line in src.split('\n'):
            if 'quant_moe_sort' in line.lower() or 'token_num' in line.lower() or 'switch' in line.lower():
                print(f"  [in source]: {line.strip()}")
    except:
        pass

    # 13. Check for block_m=16 support
    print("\n[BM16] Block_m=16 support check:")
    try:
        src = inspect.getsource(fm.get_block_size_M)
        if '16' in src:
            print("  block_m=16 mentioned in get_block_size_M!")
        for line in src.split('\n'):
            if '16' in line or 'block' in line.lower():
                print(f"  {line.strip()}")
    except:
        pass

    # 14. ck_moe_stage1 signature
    print("\n[CK] ck_moe_stage1 signature:")
    try:
        sig = inspect.signature(fm.ck_moe_stage1)
        print(f"  {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 15. ck_moe_stage2_fwd signature
    print("\n[CK] ck_moe_stage2_fwd signature:")
    try:
        sig = inspect.signature(aiter.ck_moe_stage2_fwd)
        print(f"  {sig}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("=" * 60)


def _profile_case(data):
    """Profile a single benchmark case with CUDA events."""
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
    _, model_dim, inter_dim = fm.get_inter_dim(
        gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]
    block_m = fm.get_block_size_M(fm.get_padded_M(M), topk, E, inter_dim)

    # Show what config was selected
    print(f"\n[CFG] M={M} E={E} d_model={model_dim} d_inter={inter_dim} topk={topk} block_m={block_m}")

    # Get the 2stage config
    try:
        q_dtype_a = torch.float4_e2m1fn_x2
        q_dtype_w = torch.float4_e2m1fn_x2
        isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
        cfg = fm.get_2stage_cfgs(
            fm.get_padded_M(M), model_dim, inter_dim, E, topk,
            torch.bfloat16, q_dtype_a, q_dtype_w, QuantType.per_1x32,
            isG1U1, ActivationType.Silu, False,
            hidden_pad, intermediate_pad, True)
        print(f"  run_1stage={cfg.run_1stage} block_m={cfg.block_m} splitk={cfg.splitk}")
        if hasattr(cfg.stage1, 'keywords'):
            kn = cfg.stage1.keywords.get('kernelName', '')
            print(f"  stage1 kernel: {kn[:80]}...")
        if hasattr(cfg, 'stage2') and hasattr(cfg.stage2, 'keywords'):
            kn2 = cfg.stage2.keywords.get('kernelName', '')
            print(f"  stage2 kernel: {kn2[:80]}...")
    except Exception as e:
        print(f"  cfg error: {e}")

    # CUDA event timing (5 iterations, take median)
    times_list = []
    for trial in range(5):
        events = [torch.cuda.Event(enable_timing=True) for _ in range(7)]
        torch.cuda.synchronize()

        events[0].record()
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = fm.moe_sorting(
            topk_ids, topk_weights, E, model_dim, torch.bfloat16, block_m, None, None, 0)
        events[1].record()

        from aiter.fused_moe import fused_dynamic_mxfp4_quant_moe_sort
        a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
            hidden_states, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=1, block_size=block_m)
        events[2].record()

        w1_scale_v = gate_up_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        a2 = torch.empty((M * topk, inter_dim), dtype=torch.bfloat16, device='cuda')
        fm.ck_moe_stage1(
            a1, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, a2, topk,
            kernelName="", block_m=block_m,
            a1_scale=a1_scale, w1_scale=w1_scale_v,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            splitk=0, use_non_temporal_load=False, dtype=torch.bfloat16)
        events[3].record()

        a2_q, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
            a2, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
            token_num=M, topk=topk, block_size=block_m)
        events[4].record()

        w2_scale_v = down_weight_scale_shuffled.view(dtypes.fp8_e8m0)
        aiter.ck_moe_stage2_fwd(
            a2_q, gate_up_weight_shuffled, down_weight_shuffled,
            sorted_ids, sorted_expert_ids, num_valid_ids, moe_buf, topk,
            kernelName="", block_m=block_m,
            a2_scale=a2_scale, w2_scale=w2_scale_v,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            sorted_weights=sorted_weights, use_non_temporal_load=False)
        events[5].record()

        events[6].record()
        torch.cuda.synchronize()

        times = {
            'sort': events[0].elapsed_time(events[1]) * 1000,
            'q1': events[1].elapsed_time(events[2]) * 1000,
            's1': events[2].elapsed_time(events[3]) * 1000,
            'q2': events[3].elapsed_time(events[4]) * 1000,
            's2': events[4].elapsed_time(events[5]) * 1000,
            'total': events[0].elapsed_time(events[5]) * 1000,
        }
        times_list.append(times)

    # Take median
    med = {}
    for k in times_list[0]:
        vals = sorted([t[k] for t in times_list])
        med[k] = vals[len(vals)//2]

    print(f"  [TIME] sort={med['sort']:.0f}us q1={med['q1']:.0f}us "
          f"s1={med['s1']:.0f}us q2={med['q2']:.0f}us "
          f"s2={med['s2']:.0f}us | total={med['total']:.0f}us")

    # Phase percentages
    for k in ['sort', 'q1', 's1', 'q2', 's2']:
        pct = med[k] / med['total'] * 100 if med['total'] > 0 else 0
        print(f"    {k}: {pct:.1f}%")


def _patch():
    global _patched
    if _patched:
        return
    _patched = True
    # Minimal patches for the actual kernel
    orig_use_nt = fm.use_nt
    fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
    orig_bsm = fm.get_block_size_M
    fm.get_block_size_M = lambda t, k, e, d: (32 if t*k//e < 50 else 64) if e <= 64 else orig_bsm(t, k, e, d)


def custom_kernel(data: input_t) -> output_t:
    global _call_count
    _patch()

    _call_count += 1

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    # On first call: dump source code
    if _call_count == 1:
        _probe_source()

    # Profile each unique case (warmup + profile)
    if _call_count <= 7:
        try:
            # Warmup
            fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                      topk_weights, topk_ids, expert_mask=None,
                      activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                      doweight_stage1=False,
                      w1_scale=gate_up_weight_scale_shuffled,
                      w2_scale=down_weight_scale_shuffled,
                      hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)
            torch.cuda.synchronize()
            _profile_case(data)
        except Exception as e:
            import traceback
            print(f"[PROF ERR] _call_count={_call_count}: {e}")
            traceback.print_exc()

    return fused_moe(hidden_states, gate_up_weight_shuffled, down_weight_shuffled,
                     topk_weights, topk_ids, expert_mask=None,
                     activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                     doweight_stage1=False,
                     w1_scale=gate_up_weight_scale_shuffled,
                     w2_scale=down_weight_scale_shuffled,
                     hidden_pad=hidden_pad, intermediate_pad=intermediate_pad)

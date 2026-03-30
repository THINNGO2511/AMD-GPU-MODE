#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE — Probe get_2stage_cfgs matching logic + try injecting tuned kernel names.
"""
import torch
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False


def custom_kernel(data: input_t) -> output_t:
    global _probed

    (
        hidden_states, gate_up_weight, down_weight,
        gate_up_weight_scale, down_weight_scale,
        gate_up_weight_shuffled, down_weight_shuffled,
        gate_up_weight_scale_shuffled, down_weight_scale_shuffled,
        topk_weights, topk_ids, config,
    ) = data

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

    if not _probed:
        _probed = True
        import inspect

        # 1. Read get_2stage_cfgs source
        try:
            src = inspect.getsource(fm.get_2stage_cfgs)
            print(f"[CFG] get_2stage_cfgs ({len(src)} chars):")
            for i, line in enumerate(src.split('\n')):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[CFG] error: {e}")

        # 2. Read get_padded_M
        try:
            src2 = inspect.getsource(fm.get_padded_M)
            print(f"\n[PAD] get_padded_M:")
            for i, line in enumerate(src2.split('\n')):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[PAD] error: {e}")

        # 3. Read moe_sorting source
        try:
            src3 = inspect.getsource(fm.moe_sorting)
            print(f"\n[SORT] moe_sorting ({len(src3)} chars):")
            for i, line in enumerate(src3.split('\n')[:50]):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[SORT] error: {e}")

        # 4. Read fused_moe_2stages source
        try:
            src4 = inspect.getsource(fm.fused_moe_2stages)
            print(f"\n[2STAGE] fused_moe_2stages ({len(src4)} chars):")
            for i, line in enumerate(src4.split('\n')):
                print(f"  {i}: {line}")
        except Exception as e:
            print(f"[2STAGE] error: {e}")

        # 5. Test get_padded_M for our benchmark sizes
        M = hidden_states.shape[0]
        try:
            padded = fm.get_padded_M(M)
            print(f"\n[TEST] get_padded_M({M}) = {padded}")
            for test_m in [16, 128, 512]:
                print(f"  get_padded_M({test_m}) = {fm.get_padded_M(test_m)}")
        except Exception as e:
            print(f"[TEST] error: {e}")

        # 6. Try calling get_2stage_cfgs directly for a benchmark case
        try:
            E = gate_up_weight_shuffled.shape[0]
            _, model_dim, inter_dim = fm.get_inter_dim(gate_up_weight_shuffled.shape, down_weight_shuffled.shape)
            topk = topk_ids.shape[1]
            isG1U1 = inter_dim != gate_up_weight_shuffled.shape[1]
            isShuffled = getattr(gate_up_weight_shuffled, "is_shuffled", False)

            print(f"\n[MATCH] Params: M={M}, model_dim={model_dim}, inter_dim={inter_dim}, E={E}, topk={topk}")
            print(f"  isG1U1={isG1U1}, isShuffled={isShuffled}")
            print(f"  q_dtype_a=fp4x2, q_dtype_w={gate_up_weight_shuffled.dtype}")

            metadata = fm.get_2stage_cfgs(
                fm.get_padded_M(M), model_dim, inter_dim, E, topk,
                torch.bfloat16,  # dtype
                torch.float4_e2m1fn_x2,  # q_dtype_a (fp4x2 for per_1x32 + Silu)
                gate_up_weight_shuffled.dtype,  # q_dtype_w
                QuantType.per_1x32,
                isG1U1,
                ActivationType.Silu,
                False,  # doweight_stage1
                hidden_pad,
                intermediate_pad,
                isShuffled,
            )
            print(f"  metadata: run_1stage={metadata.run_1stage}")
            print(f"  block_m={metadata.block_m}")
            for attr in ['kernelName1', 'kernelName2', 'ksplit', 'stage1', 'stage2']:
                val = getattr(metadata, attr, 'N/A')
                if callable(val):
                    print(f"  {attr}={val.__name__ if hasattr(val, '__name__') else val}")
                else:
                    print(f"  {attr}={val}")
        except Exception as e:
            import traceback
            print(f"[MATCH] error: {e}")
            traceback.print_exc()

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

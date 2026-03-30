#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE probe — dump fused_moe_2stages source, ck_moe_stage1/2 signatures,
moe_sorting internals, and all function args needed for direct calls.
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

    if not _probed:
        _probed = True
        import inspect

        # 1. fused_moe_2stages FULL source
        try:
            src = inspect.getsource(fm.fused_moe_2stages)
            print(f"[1] fused_moe_2stages ({len(src)} chars):\n{src}")
        except Exception as e:
            print(f"[1] failed: {e}")

        # 2. _moe_sorting_impl source
        try:
            src = inspect.getsource(fm._moe_sorting_impl)
            print(f"\n[2] _moe_sorting_impl:\n{src}")
        except Exception as e:
            print(f"[2] failed: {e}")

        # 3. ck_moe_stage1 wrapper source (not the C++ op)
        try:
            src = inspect.getsource(fm.ck_moe_stage1)
            print(f"\n[3] ck_moe_stage1:\n{src}")
        except Exception as e:
            print(f"[3] failed: {e}")

        # 4. ck_moe_stage1_fwd signature (the C++ op)
        try:
            import aiter as _aiter
            if hasattr(_aiter, 'ck_moe_stage1_fwd'):
                sig = inspect.signature(_aiter.ck_moe_stage1_fwd)
                print(f"\n[4] ck_moe_stage1_fwd{sig}")
            else:
                # Check torch.ops.aiter
                attrs = [a for a in dir(_aiter) if 'stage1' in a.lower() or 'ck_moe' in a.lower()]
                print(f"\n[4] aiter stage1/ck_moe attrs: {attrs}")
        except Exception as e:
            print(f"[4] failed: {e}")

        # 5. ck_moe_stage2_fwd signature
        try:
            import aiter as _aiter
            if hasattr(_aiter, 'ck_moe_stage2_fwd'):
                sig = inspect.signature(_aiter.ck_moe_stage2_fwd)
                print(f"\n[5] ck_moe_stage2_fwd{sig}")
            else:
                attrs = [a for a in dir(_aiter) if 'stage2' in a.lower()]
                print(f"\n[5] aiter stage2 attrs: {attrs}")
        except Exception as e:
            print(f"[5] failed: {e}")

        # 6. moe_sorting_fwd / moe_sorting_opus_fwd signature
        try:
            import aiter as _aiter
            sort_attrs = [a for a in dir(_aiter) if 'sorting' in a.lower() or 'moe_sort' in a.lower()]
            print(f"\n[6] aiter sorting attrs: {sort_attrs}")
            for attr in sort_attrs:
                try:
                    fn = getattr(_aiter, attr)
                    sig = inspect.signature(fn)
                    print(f"  {attr}{sig}")
                except:
                    print(f"  {attr}: no signature")
        except Exception as e:
            print(f"[6] failed: {e}")

        # 7. fused_dynamic_mxfp4_quant_moe_sort source
        try:
            if hasattr(fm, 'fused_dynamic_mxfp4_quant_moe_sort'):
                src = inspect.getsource(fm.fused_dynamic_mxfp4_quant_moe_sort)
                print(f"\n[7] fused_dynamic_mxfp4_quant_moe_sort:\n{src}")
        except Exception as e:
            print(f"[7] failed: {e}")

        # 8. per_1x32_f4_quant source
        try:
            if hasattr(fm, 'per_1x32_f4_quant'):
                src = inspect.getsource(fm.per_1x32_f4_quant)
                print(f"\n[8] per_1x32_f4_quant:\n{src}")
        except Exception as e:
            print(f"[8] failed: {e}")

        # 9. moe_mxfp4_sort source
        try:
            from aiter.utility import fp4_utils
            if hasattr(fp4_utils, 'moe_mxfp4_sort'):
                src = inspect.getsource(fp4_utils.moe_mxfp4_sort)
                print(f"\n[9] moe_mxfp4_sort:\n{src}")
        except Exception as e:
            print(f"[9] failed: {e}")

        # 10. MOEMetadata definition
        try:
            src = inspect.getsource(fm.MOEMetadata)
            print(f"\n[10] MOEMetadata:\n{src}")
        except Exception as e:
            try:
                print(f"\n[10] MOEMetadata fields: {fm.MOEMetadata._fields}")
            except:
                print(f"[10] failed: {e}")

        # 11. moe_op.py source (contains ck_moe_stage1_fwd definition)
        try:
            moe_op_path = "/home/runner/aiter/aiter/ops/moe_op.py"
            with open(moe_op_path) as f:
                src = f.read()
            print(f"\n[11] moe_op.py ({len(src)} chars):\n{src}")
        except Exception as e:
            print(f"[11] failed: {e}")

    hidden_pad = config["d_hidden_pad"] - config["d_hidden"]
    intermediate_pad = config["d_expert_pad"] - config["d_expert"]

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

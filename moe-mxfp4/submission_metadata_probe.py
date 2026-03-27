#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""Probe: Check MOEMetadata fields, get_2stage_cfgs full source, and test ksplit."""
import torch
import sys
import os
import inspect
from task import input_t, output_t
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False

def _probe():
    global _probed
    if _probed:
        return
    _probed = True

    # 1. MOEMetadata structure
    try:
        print(f"\n=== MOEMetadata ===", file=sys.stderr)
        if hasattr(fm, 'MOEMetadata'):
            meta_cls = fm.MOEMetadata
            print(f"type: {type(meta_cls)}", file=sys.stderr)
            if hasattr(meta_cls, '_fields'):
                print(f"namedtuple fields: {meta_cls._fields}", file=sys.stderr)
            if hasattr(meta_cls, '__slots__'):
                print(f"__slots__: {meta_cls.__slots__}", file=sys.stderr)
            try:
                src = inspect.getsource(meta_cls)
                for i, line in enumerate(src.split('\n')[:30]):
                    print(f"  {i}: {line}", file=sys.stderr)
            except:
                pass
            # Try creating one
            try:
                test_meta = meta_cls(None, None, 32, 2, False)
                print(f"test_meta: {test_meta}", file=sys.stderr)
                print(f"test_meta.ksplit: {test_meta.ksplit}", file=sys.stderr)
                print(f"test_meta fields: {[f for f in dir(test_meta) if not f.startswith('_')]}", file=sys.stderr)
            except Exception as e:
                print(f"create error: {e}", file=sys.stderr)
    except Exception as e:
        print(f"MOEMetadata error: {e}", file=sys.stderr)

    # 2. get_2stage_cfgs FULL source
    try:
        orig = fm.get_2stage_cfgs.__wrapped__ if hasattr(fm.get_2stage_cfgs, '__wrapped__') else fm.get_2stage_cfgs
        src = inspect.getsource(orig)
        lines = src.split('\n')
        print(f"\n=== get_2stage_cfgs ({len(lines)} lines) ===", file=sys.stderr)
        for i, line in enumerate(lines):
            print(f"  {i:3d}: {line}", file=sys.stderr)
    except Exception as e:
        print(f"get_2stage_cfgs error: {e}", file=sys.stderr)

    # 3. All FP4 kernel binary names
    try:
        d = "/home/runner/aiter/hsa/gfx950/fmoe_2stages"
        if os.path.isdir(d):
            files = sorted(os.listdir(d))
            fp4 = [f for f in files if 'FP4' in f]
            print(f"\n=== FP4 kernel binaries ({len(fp4)}) ===", file=sys.stderr)
            for f in fp4:
                print(f"  {f}", file=sys.stderr)
    except Exception as e:
        print(f"kernel error: {e}", file=sys.stderr)

    # 4. DSv3 CSV E=33 entries
    try:
        csv_path = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        with open(csv_path) as f:
            lines = f.readlines()
        e33 = [l for l in lines if ',33,' in l]
        print(f"\n=== CSV E=33 ({len(e33)}) ===", file=sys.stderr)
        if lines:
            print(f"  header: {lines[0].rstrip()}", file=sys.stderr)
        for l in e33[:15]:
            print(f"  {l.rstrip()}", file=sys.stderr)
    except Exception as e:
        print(f"CSV error: {e}", file=sys.stderr)

    sys.stderr.flush()


def custom_kernel(data: input_t) -> output_t:
    _probe()
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
        topk_weights, topk_ids, expert_mask=None,
        activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
        doweight_stage1=False,
        w1_scale=gate_up_weight_scale_shuffled,
        w2_scale=down_weight_scale_shuffled,
        a1_scale=None, a2_scale=None,
        hidden_pad=hidden_pad, intermediate_pad=intermediate_pad,
    )

#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Probe: Print EVERYTHING about d=2048 kernel selection.
Submit as BENCHMARK to see stdout. This tells us what to optimize.
"""
import torch
import functools
import sys
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_call_count = 0

def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    # Spy on get_2stage_cfgs
    orig = fm.get_2stage_cfgs.__wrapped__
    @functools.lru_cache(maxsize=2048)
    def spy_get_2stage(*args, **kwargs):
        result = orig(*args, **kwargs)
        # Extract kernel info
        s1kw = result.stage1.keywords if hasattr(result.stage1, 'keywords') else {}
        s2kw = result.stage2.keywords if hasattr(result.stage2, 'keywords') else {}
        s1kn = s1kw.get('kernelName', '<EMPTY>')
        s2kn = s2kw.get('kernelName', '<EMPTY>')
        print(f"[PROBE get_2stage_cfgs]", flush=True)
        print(f"  args: token={args[0]} model_dim={args[1]} inter_dim={args[2]} "
              f"expert={args[3]} topk={args[4]}", flush=True)
        print(f"  dtype={args[5]} q_dtype_a={args[6]} q_dtype_w={args[7]} "
              f"q_type={args[8]}", flush=True)
        if len(args) > 13:
            print(f"  hidden_pad={args[12]} intermediate_pad={args[13]}", flush=True)
        print(f"  stage1 kernelName: {s1kn}", flush=True)
        print(f"  stage2 kernelName: {s2kn}", flush=True)
        print(f"  block_m={result.block_m} ksplit={result.ksplit} "
              f"run_1stage={result.run_1stage}", flush=True)
        # Print ALL stage1 keywords
        print(f"  stage1 ALL kwargs: {s1kw}", flush=True)
        print(f"  stage2 ALL kwargs: {s2kw}", flush=True)
        sys.stdout.flush()
        return result
    fm.get_2stage_cfgs = spy_get_2stage
    fm.cfg_2stages = None

    # Also spy on use_nt
    orig_nt = fm.use_nt
    def spy_nt(token, topk, expert):
        result = orig_nt(token, topk, expert)
        print(f"[PROBE use_nt] token={token} topk={topk} expert={expert} -> {result}", flush=True)
        return result
    fm.use_nt = spy_nt

    # Spy on get_block_size_M
    orig_bsm = fm.get_block_size_M
    def spy_bsm(token, topk, expert, inter_dim):
        result = orig_bsm(token, topk, expert, inter_dim)
        est_m = token * topk // expert
        print(f"[PROBE block_size_M] token={token} topk={topk} expert={expert} "
              f"inter_dim={inter_dim} est_m={est_m} -> block_m={result}", flush=True)
        return result
    fm.get_block_size_M = spy_bsm

    # Print CSV info
    try:
        import pandas as pd
        csv_path = "/home/runner/aiter/aiter/configs/tuned_fmoe.csv"
        df = pd.read_csv(csv_path)
        print(f"\n[PROBE CSV] tuned_fmoe.csv: {len(df)} rows, columns: {list(df.columns)}", flush=True)
        # Show E=33 entries
        e33 = df[df['expert'] == 33] if 'expert' in df.columns else df[df.iloc[:,4] == 33]
        print(f"[PROBE CSV] E=33 rows: {len(e33)}", flush=True)
        if len(e33) > 0:
            print(e33.to_csv(index=False), flush=True)
        # Show E=257 entries
        e257 = df[df['expert'] == 257] if 'expert' in df.columns else df[df.iloc[:,4] == 257]
        print(f"[PROBE CSV] E=257 rows: {len(e257)}", flush=True)
        if len(e257) > 0:
            print(e257.head(5).to_csv(index=False), flush=True)
    except Exception as e:
        print(f"[PROBE CSV] Error: {e}", flush=True)

    # Print dsv3 CSV
    try:
        csv2 = "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv"
        df2 = pd.read_csv(csv2)
        print(f"\n[PROBE dsv3 CSV] {len(df2)} rows, columns: {list(df2.columns)}", flush=True)
        print(df2.to_csv(index=False), flush=True)
    except Exception as e:
        print(f"[PROBE dsv3 CSV] Error: {e}", flush=True)

    # Print available CK kernels for MoE
    try:
        import glob
        co_files = glob.glob("/home/runner/aiter/hsa/gfx950/fmoe_2stages/*.co")
        print(f"\n[PROBE] fmoe_2stages .co files: {len(co_files)}", flush=True)
        # Find d=2048-relevant kernels (larger tiles)
        for f in sorted(co_files):
            name = f.split("/")[-1]
            if "256x" in name or "128x128" in name:
                print(f"  {name}", flush=True)
    except Exception as e:
        print(f"[PROBE .co] Error: {e}", flush=True)

    sys.stdout.flush()


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

    if _call_count <= 8:
        print(f"\n[CALL {_call_count}] hidden={hidden_states.shape} "
              f"E={config.get('nroutedexperts',0)+config.get('nsharedexperts',0)} "
              f"d_expert={config['dexpert']} bs={config['bs']} "
              f"topk={config['nexpertspertoken']+config['nsharedexperts']}", flush=True)

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

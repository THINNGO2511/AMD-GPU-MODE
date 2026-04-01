#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE: Fixed CSV format for AITER_CONFIG_FMOE.
Previous attempt used string "QuantType.per_1x32" — likely needs numeric 3.
Also: try injecting 256x32 stage1 for ALL E=257 shapes (not just E<=64).
"""
import os
import tempfile

S1_64 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S1_256 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_v3_Nswizzle0_Quant3_MulRoutedWeight0_silu_FP4X2_FP4X2_B16"
S2_V1 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_Nswizzle0_Quant3_MulRoutedWeight1_FP4X2_FP4X2_B16"

# First probe what the existing CSV looks like to match format
csv_path = "/tmp/custom_fmoe_v2.csv"

os.environ["HIP_FORCE_DEV_KERNARG"] = "1"

import torch
import functools
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_patched = False
_probed_csv = False

def _probe_csv():
    global _probed_csv
    if _probed_csv:
        return
    _probed_csv = True

    # Read the existing merged CSV to understand column format
    try:
        import pandas as pd
        csv_paths = [
            "/home/runner/aiter/aiter/configs/tuned_fmoe.csv",
            "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        ]
        for p in csv_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                print(f"[CSV2] {p}: {len(df)} rows, columns={list(df.columns)}")
                # Show E=257 entries
                e257 = df[df['expert'] == 257] if 'expert' in df.columns else pd.DataFrame()
                print(f"[CSV2] E=257 entries: {len(e257)}")
                if len(e257) > 0:
                    print(f"[CSV2] E=257 sample:\n{e257.head(3).to_string()}")
                # Show E=33 entries
                e33 = df[df['expert'] == 33] if 'expert' in df.columns else pd.DataFrame()
                print(f"[CSV2] E=33 entries: {len(e33)}")
                if len(e33) > 0:
                    print(f"[CSV2] E=33 sample:\n{e33.head(3).to_string()}")
                # Show all column dtypes
                print(f"[CSV2] dtypes:\n{df.dtypes.to_string()}")
                # Show first 3 rows raw
                print(f"[CSV2] First 3 rows:\n{df.head(3).to_string()}")
    except Exception as e:
        print(f"[CSV2] CSV probe error: {e}")

    # Check what cu_num the runner reports
    try:
        print(f"[CSV2] fm.CU_NUM = {getattr(fm, 'CU_NUM', 'not found')}")
        print(f"[CSV2] fm.cu_num = {getattr(fm, 'cu_num', 'not found')}")
    except:
        pass


def _patch():
    global _patched
    if _patched:
        return
    _patched = True

    _probe_csv()

    fm.use_nt = lambda token, topk, expert: False

    # CK kernel injection for E<=64 d<2048
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
                    kn1 = S1_256 if est_m >= 100 else S1_64
                    return fm.MOEMetadata(
                        functools.partial(fm.ck_moe_stage1,
                            kernelName=kn1, activation=activation,
                            quant_type=q_type, dtype=dtype,
                            splitk=0, use_non_temporal_load=False),
                        functools.partial(aiter.ck_moe_stage2_fwd,
                            kernelName=S2_V1, activation=activation,
                            quant_type=q_type, use_non_temporal_load=False),
                        32, 0, False)
            except Exception:
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

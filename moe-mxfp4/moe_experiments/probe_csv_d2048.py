#!POPCORN leaderboard amd-moe-mxfp4
#!POPCORN gpu MI355X
"""
MoE Probe: Check if new aiter CSV entries for E=33 d=2048 exist on runner.
Commit fc0c54bb added entries for 32 experts, d=2048, topk=8.
Our benchmark uses E=33 (32+1 shared), topk=9 (8+1). Will the CSV match?

This probe:
1. Reads dsv3_fp4_tuned_fmoe.csv - checks for E=32/33, d=2048 entries
2. Shows CSV column headers and all matching rows
3. Shows what get_2stage_cfgs returns for our exact shapes
4. Checks aiter version/git info for recent commits
5. Checks the 13/15-field key structure
"""
import torch
from task import input_t, output_t
import aiter
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
import aiter.fused_moe as fm

_probed = False


def _probe():
    global _probed
    if _probed:
        return
    _probed = True
    import os, glob as globmod

    # 1. Find and read the MoE CSV
    csv_candidates = [
        "/home/runner/aiter/aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv",
        "/home/runner/aiter/aiter/configs/model_configs/tuned_fmoe.csv",
    ]
    csv_candidates += globmod.glob("/home/runner/aiter/aiter/configs/**/*fmoe*.csv", recursive=True)
    csv_candidates += globmod.glob("/home/runner/aiter/aiter/configs/**/*moe*.csv", recursive=True)
    csv_candidates = list(set(csv_candidates))

    for csv_path in csv_candidates:
        if not os.path.exists(csv_path):
            continue
        print("PROBE:CSV_FILE:%s" % csv_path)
        try:
            with open(csv_path) as f:
                lines = f.readlines()
            print("PROBE:CSV_TOTAL_LINES:%d" % len(lines))

            # Print header
            if lines:
                print("PROBE:CSV_HEADER:%s" % lines[0].strip())

            # Find ALL rows with expert=32 or expert=33 or d=2048
            for i, line in enumerate(lines):
                line_lower = line.lower()
                # Check for expert counts 32/33 and dimension 2048
                fields = line.strip().split(',')
                if any(f.strip() in ('32', '33') for f in fields) or '2048' in line:
                    print("PROBE:CSV_MATCH_LINE%d:%s" % (i, line.strip()[:300]))

            # Also count entries by expert count
            expert_counts = {}
            for line in lines[1:]:  # skip header
                fields = line.strip().split(',')
                if len(fields) > 5:
                    # Try to identify expert count field
                    for j, f in enumerate(fields):
                        f = f.strip()
                        if f.isdigit() and int(f) in range(1, 300):
                            expert_counts.setdefault(j, {})
                            expert_counts[j][f] = expert_counts[j].get(f, 0) + 1
            # Print field value distributions for first few fields
            for j in sorted(expert_counts.keys())[:10]:
                vals = expert_counts[j]
                if len(vals) < 20:  # likely a categorical field
                    print("PROBE:CSV_FIELD%d_DIST:%s" % (j, dict(sorted(vals.items(), key=lambda x: -x[1])[:10])))

        except Exception as e:
            print("PROBE:CSV_ERROR:%s %s" % (csv_path, e))

    # 2. Check get_2stage_cfgs for our exact benchmark shapes
    print("\n--- get_2stage_cfgs probe ---")
    try:
        # Our MoE benchmark shapes
        # E=257 d=256: bs=16/128/512
        # E=33 d=512: bs=16/128/512
        # E=33 d=2048: bs=512
        test_shapes = [
            # (token, model_dim, inter_dim, expert, topk)
            (16, 256, 256, 257, 9, "E=257 bs=16 d=256"),
            (512, 256, 256, 257, 9, "E=257 bs=512 d=256"),
            (16, 512, 512, 33, 9, "E=33 bs=16 d=512"),
            (512, 512, 512, 33, 9, "E=33 bs=512 d=512"),
            (512, 2048, 2048, 33, 9, "E=33 bs=512 d=2048"),  # THE BOTTLENECK
            # Also try E=32 topk=8 (what the commit might have)
            (512, 2048, 2048, 32, 8, "E=32 bs=512 d=2048 (commit format?)"),
            (512, 2048, 2048, 33, 8, "E=33 bs=512 d=2048 topk=8"),
            (512, 2048, 2048, 32, 9, "E=32 bs=512 d=2048 topk=9"),
        ]

        # Get the original (unwrapped) function
        get_2stage = fm.get_2stage_cfgs
        if hasattr(get_2stage, '__wrapped__'):
            get_2stage = get_2stage.__wrapped__

        import torch
        dtype = torch.bfloat16
        q_dtype_a = torch.uint8  # FP4
        q_dtype_w = torch.uint8  # FP4
        q_type = QuantType.per_1x32
        use_g1u1 = True
        activation = ActivationType.Silu
        doweight = False

        for token, mdim, idim, expert, topk, label in test_shapes:
            try:
                # Check what cu_num is used
                cu_num = aiter.get_cu_num()
                print("PROBE:CU_NUM:%d" % cu_num)

                result = get_2stage(
                    token, mdim, idim, expert, topk,
                    dtype, q_dtype_a, q_dtype_w, q_type,
                    use_g1u1, activation, doweight,
                    0, 0,  # hidden_pad, intermediate_pad
                    True   # is_shuffled
                )
                # Decode result
                s1_name = ""
                s2_name = ""
                if hasattr(result, 'stage1') and hasattr(result.stage1, 'keywords'):
                    s1_name = result.stage1.keywords.get('kernelName', 'NONE')
                if hasattr(result, 'stage2') and hasattr(result.stage2, 'keywords'):
                    s2_name = result.stage2.keywords.get('kernelName', 'NONE')

                print("PROBE:CFG_%s:stage1=%s" % (label, s1_name[:80] if s1_name else "DEFAULT"))
                print("PROBE:CFG_%s:stage2=%s" % (label, s2_name[:80] if s2_name else "DEFAULT"))
                print("PROBE:CFG_%s:block_m=%s ksplit=%s run_1stage=%s" % (
                    label,
                    getattr(result, 'block_m', '?'),
                    getattr(result, 'ksplit', '?'),
                    getattr(result, 'run_1stage', '?'),
                ))
            except Exception as e:
                print("PROBE:CFG_%s:ERROR=%s" % (label, str(e)[:200]))

    except Exception as e:
        print("PROBE:GET_2STAGE_ERROR:%s" % e)

    # 3. Check aiter git info
    try:
        import subprocess
        r = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            capture_output=True, text=True, cwd="/home/runner/aiter",
            timeout=5
        )
        if r.returncode == 0:
            for line in r.stdout.strip().split('\n'):
                print("PROBE:GIT_LOG:%s" % line[:150])
        else:
            print("PROBE:GIT_LOG_ERROR:%s" % r.stderr[:200])
    except Exception as e:
        print("PROBE:GIT_ERROR:%s" % e)

    # 4. Check get_2stage_cfgs source for key format
    try:
        import inspect
        src = inspect.getsource(fm.get_2stage_cfgs.__wrapped__)
        # Find the CSV key construction
        for line in src.split('\n'):
            if 'key' in line.lower() or 'csv' in line.lower() or 'config' in line.lower():
                print("PROBE:GET2STAGE_SRC:%s" % line.rstrip()[:200])
    except Exception as e:
        print("PROBE:GET2STAGE_SRC_ERROR:%s" % e)

    # 5. Check token_num_quant_moe_sort_switch
    try:
        import inspect
        src = inspect.getsource(fm.fused_moe_2stages)
        for line in src.split('\n'):
            if 'switch' in line.lower() or 'token_num' in line.lower():
                print("PROBE:FUSED2STAGES_SRC:%s" % line.rstrip()[:200])
    except Exception as e:
        pass

    print("PROBE:DONE")


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

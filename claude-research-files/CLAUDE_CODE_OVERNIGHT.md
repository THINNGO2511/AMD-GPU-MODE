# CLAUDE CODE AUTONOMOUS SESSION — AMD GPU MODE Hackathon
## Overnight April 3-4, 2026 | Deadline: April 6, 11:59 PM PST
## Edward is sleeping. You run everything autonomously until he wakes up.

---

## MISSION

Improve our scores on 3 GPU kernel optimization problems to make top 10 aggregate in a $1.1M hackathon. You have full autonomy. Submit, analyze, iterate, pivot. Don't wait for human input.

## CURRENT SCORES

| Problem | Score | Rank | Top 20 Cutoff | In Top 20? | Best File |
|---------|-------|------|---------------|------------|-----------|
| MLA | **35.5μs** | ~#13 | ~35μs | BORDERLINE | mixed-mla/mla_fp8q_all.py |
| GEMM | **15.7μs** | ~#160 | ~9μs | NO | mxfp4-mm/sub_ultimate_v1.py |
| MoE | **163μs** | ~#65 | ~143μs | NO | moe-mxfp4/submission_optimized_v2.py |

**Aggregate math:** Need all 3 in top 20. MLA is borderline. GEMM needs 43% improvement. MoE needs 12% improvement.

## REPO
```
/home/claude/AMD-GPU-MODE/
```

---

## YOUR OPERATING LOOP

You have 1M context. Use it. Run continuously for 8+ hours. Here is your loop:

```
while edward_is_sleeping:
    1. Submit MLA ratchet to leaderboard (every ~65 min)
    2. Run next GEMM experiment (benchmark, analyze result, adapt)
    3. Run next MoE experiment (benchmark, analyze result, adapt)
    4. Log everything to overnight_logs/
    5. Every 3 hours: review all results, reassess strategy, pivot if needed
```

### Rate Limits (per problem, independent)
- **6 benchmarks/hour** = 1 every 10 minutes
- **1 leaderboard/hour** = 1 every 65 minutes (be safe)
- All 3 problems run in parallel — you have 18 benchmark slots + 3 leaderboard slots per hour total

### Submission Commands
```bash
# Benchmark (6/hr per problem) — use for testing
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark <file> --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark <file> --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark <file> --no-tui

# Leaderboard (1/hr per problem) — use for real score
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard <file> --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard <file> --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard <file> --no-tui
```

### Timing Strategy
Interleave submissions across problems to maximize throughput:
```
Min 0:  GEMM benchmark #1 + MoE benchmark #1 + MLA leaderboard
Min 10: GEMM benchmark #2 + MoE benchmark #2
Min 20: GEMM benchmark #3 + MoE benchmark #3
Min 30: GEMM benchmark #4 + MoE benchmark #4
Min 40: GEMM benchmark #5 + MoE benchmark #5
Min 50: GEMM benchmark #6 + MoE benchmark #6
Min 65: Next hour starts — MLA leaderboard again
```

You can submit multiple problems simultaneously (they're independent queues). Don't wait for one to finish before starting another.

---

## PHASE 1: INTELLIGENCE GATHERING (First 30 minutes)

Do ALL of these immediately, in parallel:

### 1a. Discord grep (instant, no slots needed)
```bash
cd /home/claude/AMD-GPU-MODE
grep -ri "maxwell" discord-logs/ 2>/dev/null
grep -ri "deepgemm" discord-logs/ 2>/dev/null
grep -ri "107" discord-logs/ 2>/dev/null
grep -ri "moe.*custom\|moe.*kernel\|quant.*fast" discord-logs/ 2>/dev/null
```

### 1b. deepgemm probe (1 MoE or GEMM benchmark slot)
Write a submission that prints:
```python
import aiter
print([x for x in dir(aiter) if 'deep' in x.lower()])
# Try: from aiter import deepgemm_ck; help(deepgemm_ck)
# Try: import deepgemm; print(dir(deepgemm))
# Grep runner source: subprocess.run(['grep', '-rl', 'deepgemm', '/home/runner/aiter/'])
```
Still define custom_kernel so the eval harness doesn't crash.

### 1c. MoE d=2048 kernel probe (1 MoE benchmark slot)
Monkey-patch get_2stage_cfgs to print what kernel/config the library picks for the d=2048 bs=512 shape. This 337μs shape is the geomean killer.

### 1d. GEMM: start config sweep immediately (remaining GEMM benchmark slots)
Use the sweep generator at `/home/claude/AMD-GPU-MODE/scripts/gemm_sweep_gen.py` to create config files:
```bash
cd /home/claude/AMD-GPU-MODE
python3 scripts/gemm_sweep_gen.py --shape both --count 150
```
Then start submitting sweep files as GEMM benchmarks. Target the two bottleneck shapes:
- M=16, N=2112, K=7168 (currently 13.6μs benchmark)
- M=64, N=7168, K=2048 (currently 14.1μs benchmark)

### 1e. MLA ratchet (1 MLA leaderboard slot)
```bash
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui
```

---

## PHASE 2: ADAPTIVE EXECUTION (Hours 1-8)

After Phase 1 probes return results, follow this decision tree:

### IF deepgemm_ck EXISTS:
This is highest priority. Figure out its API and test it immediately.
- Try it for GEMM: does it handle MXFP4? What shapes does it support?
- Try it for MoE: could it replace the CK 2-stage pipeline?
- This could be what Maxwell Cipher uses for 107μs MoE.

### IF deepgemm_ck is DEAD:
Focus on GEMM config sweep (high volume) + MoE creative approaches.

### GEMM Strategy (ongoing)
1. Keep running config sweep. Prioritize configs near the current best but exploring new BM/BN/warps combos.
2. After each benchmark result, log it: `echo "timestamp,shape,config,score" >> overnight_logs/gemm_sweep_results.csv`
3. If any config beats 9.5μs benchmark, IMMEDIATELY submit to leaderboard.
4. If 50+ configs all score ≥10μs, the Triton space may be exhausted. Consider:
   - Reading the actual Triton kernel source to understand what parameters really matter
   - Trying gemm_afp4wfp4 for ALL shapes (not just K=1536) with fresh configs
   - Investigating the 35 CK ASM .co files at `/home/runner/aiter/hsa/gfx950/f4gemm/`

### MoE Strategy (after probes)
1. If d=2048 probe reveals different kernel configs, try injecting faster ones.
2. Try `token_num_quant_moe_sort_switch` values: 512, 256, 2048 (it's a local var in fused_moe_2stages, may need monkey-patching).
3. Check if there are CK stage1 kernel variants (v2, v3, v4) for d>=2048 tiles.
4. If nothing works after 3 hours of MoE experiments, fall back to MoE ratcheting.

### MLA Strategy (simple)
Submit to leaderboard every ~65 minutes. No code changes. Variance play.

---

## PHASE 3: CONSOLIDATION (Final 2 hours before Edward wakes)

1. Review ALL results from the night.
2. Write a summary to `overnight_logs/MORNING_REPORT.md` covering:
   - Best GEMM benchmark score found and which config
   - MoE probe findings and any improvements
   - MLA ratchet history (did we get under 35μs?)
   - Recommended next steps for when Edward is awake
3. Submit best-found configs to leaderboard for all 3 problems.

---

## KEY aiter APIs

```python
# GEMM
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4   # bf16 A, fused quant — BEST
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4  # fp4 A+B — K=1536 only
from aiter.ops.triton.quant import dynamic_mxfp4_quant
from aiter.utility.fp4_utils import e8m0_shuffle

# MLA
from aiter.mla import mla_decode_fwd
from aiter import get_mla_metadata_info_v1, get_mla_metadata_v1

# MoE
from aiter.fused_moe import fused_moe   # NOT: from aiter import fused_moe
from aiter import ActivationType, QuantType
```

## GEMM Config Injection Pattern
The config directory on the runner is WRITABLE:
```python
# In submission file, BEFORE importing aiter GEMM functions:
import os, json
config_dir = "/home/runner/aiter/aiter/ops/triton/configs/gemm"
config = {
    "1": {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 512,
          "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 2,
          "waves_per_eu": 2, "matrix_instr_nonkdim": 16,
          "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 1024},
    # ... same config for all M-tiers: "2", "4", "8", "16", "32", "64", etc.
}
# Filename format: gfx950-GEMM-A16WFP4-N={N}-K={2*K}.json
with open(f"{config_dir}/gfx950-GEMM-A16WFP4-N=2112-K=14336.json", "w") as f:
    json.dump(config, f)
```

## MoE Monkey-Patch Pattern (proven safe)
```python
import aiter.fused_moe as fm
import functools
import aiter

fm.use_nt = lambda token, topk, expert: False
fm._USE_OPUS_MOE_SORTING = True

try:
    orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
except AttributeError:
    orig_get_2stage = fm.get_2stage_cfgs

@functools.lru_cache(maxsize=2048)
def patched_get_2stage(token, model_dim, inter_dim, expert, topk,
                       dtype, q_dtype_a, q_dtype_w, q_type,
                       use_g1u1, activation, doweight_stage1,
                       hidden_pad, intermediate_pad, is_shuffled=True):
    result = orig_get_2stage(token, model_dim, inter_dim, expert, topk,
                             dtype, q_dtype_a, q_dtype_w, q_type,
                             use_g1u1, activation, doweight_stage1,
                             hidden_pad, intermediate_pad, is_shuffled)
    if expert <= 64 and inter_dim < 2048:
        if token < 50:
            s1 = "moe_ck2stages_gemm1_64x32x32x128_1x1_MulABScaleShuffled_silu_FP4X2_FP4X2_B16"
            s2 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_FP4X2_FP4X2_B16"
            bm = 32
        else:
            s1 = "moe_ck2stages_gemm1_256x32x128x128_1x4_MulABScaleShuffled_silu_FP4X2_FP4X2_B16"
            s2 = "moe_ck2stages_gemm2_64x32x32x128_1x1_MulABScaleExpertWeightShuffled_v1_FP4X2_FP4X2_B16"
            bm = 64
        return fm.MOEMetadata(
            functools.partial(fm.ck_moe_stage1, kernelName=s1, noOpus=False),
            functools.partial(aiter.ck_moe_stage2_fwd, kernelName=s2),
            bm, 0, False
        )
    return result

fm.get_2stage_cfgs = patched_get_2stage
fm.cfg_2stages = None
```

## GEMM Shapes
```
Shape 1: M=4,   N=2880, K=512   → ~6.2μs   | config: gfx950-GEMM-A16WFP4-N=2880-K=1024.json
Shape 2: M=16,  N=2112, K=7168  → ~13.6μs  ← BOTTLENECK | config: gfx950-GEMM-A16WFP4-N=2112-K=14336.json
Shape 3: M=32,  N=4096, K=512   → ~6.2μs   | config: gfx950-GEMM-A16WFP4-N=4096-K=1024.json
Shape 4: M=32,  N=2880, K=512   → ~6.7μs   | config: gfx950-GEMM-A16WFP4-N=2880-K=1024.json
Shape 5: M=64,  N=7168, K=2048  → ~14.1μs  ← BOTTLENECK | config: gfx950-GEMM-A16WFP4-N=7168-K=4096.json
Shape 6: M=256, N=3072, K=1536  → ~16.1μs  (uses afp4wfp4) | config: gfx950-GEMM-A16WFP4-N=3072-K=3072.json
```

## MoE Shapes
```
E=257, bs=16,  d=256  → ~127μs
E=257, bs=128, d=256  → ~206μs
E=257, bs=512, d=256  → ~241μs
E=33,  bs=16,  d=512  → ~87μs
E=33,  bs=128, d=512  → ~111μs
E=33,  bs=512, d=512  → ~178μs
E=33,  bs=512, d=2048 → ~337μs  ← GEOMEAN KILLER
```

---

## CONFIRMED DEAD ENDS — DO NOT RETRY

### GEMM (18 dead ends)
1. hipBLASLt FP4: 14 attempts, accumulation order mismatch, 38% relative error. DEAD.
2. gemm_a4w4 CK ASM: 3-launch overhead (quant 12μs + shuffle 1μs + GEMM 3-8μs). DEAD.
3. gemm_a16wfp4_preshuffle: Triton KeyError 'float8_e8m0fnu'. DEAD.
4. gemm_a8wfp4: eval assertion fails on B_scale shape. DEAD.
5. Custom Triton tl.dot_scaled: 3x slower. DEAD.
6. Custom HIP MFMA via load_inline: ~90s compile eats timeout. DEAD.
7. KSPLIT > 1 for ANY shape: reduce kernel adds ~19μs. Tested 2,4,8,12,14. ALL worse. DEAD.
8. num_stages=3 for K=2048: +34% regression. DEAD.
9. Per-shape JSON config with KSPLIT=2: K=7168 became 2.4x SLOWER. DEAD.
10. afp4wfp4 for all shapes: slower than a16wfp4. DEAD.
11. L2 cache pre-warming: adds overhead. DEAD.
12. XCD remap monkey-patch: accuracy broken. DEAD.
13. CUDA/HIP graphs: 2x worse. DEAD.
14. 200+ config sweep (prior sessions): no improvement beyond ~9.74μs benchmark. DEAD.
15. deepgemm: wrong API for runner. DEAD.
16. All env vars (TRITON_HIP_USE_IN_THREAD_TRANSPOSE, AMDGCN_USE_BUFFER_OPS, etc): no effect. DEAD.

### MLA (12 dead ends)
1. pg2 for kv=1024: ~4% mismatch, fails secret seed. DEAD.
2. qseqlen2 kernel: GPU memory fault. DEAD.
3. MXFP4 KV cache: dim=288 not divisible by 512. DEAD.
4. auto_splits: fails secret seed. DEAD.
5. splits=1 for small bs: underutilizes CUs. DEAD.
6. pg4, pg16: fail accuracy. DEAD.
7. fast_mode=True: 5-10% worse. DEAD.
8. a16w8 for kv=8192: 2x slower. DEAD.
9. Custom HIP flash-decoding: 561μs (14x slower). DEAD.
10. Custom Triton flash-decoding: JIT timeout. DEAD.
11. mla_tuned_v2, mla_splits8: slower. DEAD.
12. kv_granularity tuning: zero effect. DEAD.

### MoE (23 dead ends)
1. ksplit=2: triggers cktile timeout. DEAD.
2. block_m=16: assertion error. DEAD.
3. AITER_USE_OPUS_MOE_SORTING=0 env var: CRASHES. DEAD.
4. 1-stage kernel: 182μs (slower). DEAD.
5. CK injection for d=2048 with large tiles: no improvement. DEAD.
6. Direct fused_moe_2stages call: GPU memory fault. DEAD.
7. Direct ck_moe_stage1_fwd/stage2_fwd calls: GPU memory fault. DEAD.
8. Buffer reuse: GPU crash. DEAD.
9. doweight_stage1=True: wrong results. DEAD.
10. FlyDSL: zero binaries on runner. DEAD.
11. Sort caching: moe_sorting called once already. DEAD.
12. torch.compile: DEAD.
13. dispatch_policy=1: 80% slower. DEAD.
14. ksplit=2 for d=2048: 2x slower. DEAD.
15. Custom Triton MoE: JIT timeout. DEAD.
16. sepqsort (switch=0): 179μs, worse. DEAD.
17. CSV override with 128x128 tiles for E=33: GPU crash. DEAD.
18. Stage2 v3 for E=33: same speed. DEAD.
19. 256x64x128x128 stage1 for E=33: 30% worse. DEAD.
20. OPUS off: 175μs (worse than 163μs). DEAD.
21. Removing E=33 injection: 184μs (worse). DEAD.
22. use_nt=True for d=2048: 185μs (worse). DEAD.
23. All env vars exhausted. DEAD.

---

## CRITICAL RULES

1. **NEVER use the word "stream"** in any submission — grep-filtered, submission BLOCKED
2. **Rate limits are per-problem and independent.** You have 6 bench + 1 leaderboard per hour for EACH of the 3 problems.
3. **In-sweep timing is UNRELIABLE** — L2 cache is warm on runner. Only trust popcorn-cli benchmark/leaderboard results.
4. **Ephemeral pods:** every submission starts from scratch. No state persists. Triton JIT cache destroyed.
5. **pip install is BLOCKED.**
6. **Direct fused_moe internal calls = GPU memory fault.** Always go through the C++ wrapper via `fused_moe()`.
7. **AITER_USE_OPUS_MOE_SORTING=0 env var CRASHES.** Use `fm._USE_OPUS_MOE_SORTING = False` monkey-patch.
8. **load_inline ~90s compile** for complex kernels. Only ~10s for simple ones. Budget carefully.
9. **kv_granularity = max(1, 16 // page_size)** — NOT max(page_size, 16).
10. **Log everything** to `overnight_logs/`. Edward needs to see what happened.

---

## LOGGING

Create and maintain these files:
```
overnight_logs/gemm_sweep_results.csv     — timestamp, config, benchmark score
overnight_logs/moe_experiment_log.txt     — what was tried, result
overnight_logs/mla_ratchet_log.txt        — timestamp, leaderboard score
overnight_logs/probe_results.txt          — deepgemm, d2048 probe output
overnight_logs/MORNING_REPORT.md          — summary for Edward when he wakes up
```

---

## START NOW

Begin with Phase 1 immediately. Submit the MLA ratchet, fire off the deepgemm probe, the d=2048 probe, and the first GEMM sweep configs all at once. You have the context, you have the tools, you have 8 hours. Go.

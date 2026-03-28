# Pickup Prompt — NEVER STOP MODE

Continue autonomous GPU kernel optimization for AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline April 7, 2026. This is a continuation of ongoing non-stop work — NOT a new session. Pick up exactly where we left off.

**MODE: NEVER STOP. Fully autonomous. Don't ask permission. Auto-submit, auto-research, auto-iterate, auto-sweep configs. Only stop when top 10 on ALL 3 leaderboards or I tell you to stop. NO sessions — this is continuous work.**

## Read These Files First
- `memory/active_leads.md` — current priorities and what to do next
- `memory/dead_ends.md` — what NOT to try (saves you hours)
- `memory/MEMORY.md` — index of all knowledge from 14+ rounds of work
- `KNOWLEDGE.md`, `CLAUDE.md` — technical reference
- Check `/tmp/auto_submit_v5_log.txt` — auto-submit v5 loop may still be running
- Check `ps aux | grep auto_submit` for running loops
- Check `auto_research_logs/` for sweep results

## Current Standings (as of Mar 28 2026)
- **GEMM**: 16.222μs rank 151 → need 8.4μs (top 10). **2x gap.**
- **MLA**: 42.488μs rank 11 → need 41.8μs (top 10). **0.7μs gap.**
- **MoE**: 169.131μs rank ~52 → need 136μs (top 10). **20% gap.**
- popcorn-cli at ~/.local/bin/popcorn-cli, --no-tui flag
- Rate: 6 submissions/hr, 1 leaderboard/hr per problem
- Header MUST be `#!POPCORN leaderboard <name>` (NOT benchmark!)

## Auto-Submit Loop (v5)
Auto-submit v5 may still be running (`/tmp/auto_submit_v5.sh`). Submits hourly:
- GEMM: `submission_stages3_all.py` (num_stages=3 + KSPLIT=16 for K=7168)
- MLA: `submission_pg2_pingpong.py` (pg2 + Triton BLOCK_PINGPONG env var)
- MoE: `submission_envvars_moe.py` (Triton env vars)
If not running, restart: `nohup /tmp/auto_submit_v5.sh > /tmp/auto_submit_v5_log.txt 2>&1 &`

## WHAT ACTUALLY WORKS (and what doesn't)

### Competitors' approach (8μs GEMM):
- josusanmartin: 4713 submissions = automated config sweeping
- They find optimal aiter configs through brute force, NOT custom kernels
- Key params: BM, BN, BK, NUM_KSPLIT, num_stages, waves_per_eu, matrix_instr_nonkdim

### What DOESN'T work (proven across 14 sessions):
- Custom Triton kernels: work correctly but JIT overhead negates speed gains
- Triton env vars: already auto-enabled on gfx950
- Custom kernel for K=512 only: not enough to move the geomean
- MoE: ALL approaches exhausted (FlyDSL, CU_NUM, cktile, pre-quant, compile)
- MLA pg2 for kv=1024: fails on SOME secret seeds. Competitors made it reliable (pg2_fix). The fix is unknown — need to research.

## EXECUTION PLAN

### 1. GEMM: Automated Config Sweep (HIGHEST PRIORITY)
Build a benchmark submission that tests multiple configs per run:
```python
# For each shape, try configs and print timing to stdout:
# "SWEEP M{m}N{n}K{k} BM{bm}_BN{bn}_BK{bk}_KS{ks}_S{stages}: {time}us"
```
Focus on K=7168 (M=16, N=2112) — it dominates the geomean.
Config space: BM=[8,16,32], BN=[32,64,128], BK=[128,256,512], KSPLIT=[1,2,4,8,16], stages=[2,3], waves=[1,2]

### 2. MLA: NOT just a lottery — competitors made pg2 RELIABLE
- Ananda Sai A: "pg2_fix" at 33μs (279 subs). John Hahn: "pg2_hybrid" at 32μs.
- willfisher: **"pingpong.py"** at 35μs — Triton BLOCK_PINGPONG env var!
- **They found a way to make pg2 reliable.** We haven't. The "fix" is unknown.
- Try: pg2 + TRITON_HIP_USE_BLOCK_PINGPONG=1 (affects Triton reduce kernel)
- Try: different num_kv_splits combos with pg2 (sweep 4,8,12,16,32 per shape)
- Try: "direct_stage" approach (Yufeng98 at 37μs — calling stage1+reduce directly)
- **Auto-submit retries pg2_pingpong every hour.**
- **Also write autosweep_mla.py** for focused num_kv_splits sweep with pg2

### 3. MoE: Automated Config Sweep (SAME APPROACH AS GEMM)
- josusanmartin: 3861 MoE submissions → 127μs. Ananda Sai A: 276 → 110μs.
- We tried ~30 submissions and gave up. They tried THOUSANDS. That's the gap.
- The CK kernel configs ARE tunable via monkey-patching get_2stage_cfgs:
  - block_m: [16, 32, 64, 128, 256] per shape
  - stage1 kernel: [S1_64, S1_256, S1_256x64, S1_256x128] per shape
  - stage2 kernel: [S2_V1, S2_256, S2_256x64, S2_256x128] per shape
  - ksplit: [0, 1, 2, 4, 8] per shape
  - use_nt: [True, False] per shape
- That's 5×4×4×5×2 = 800 combos per shape × 7 shapes = 5600 total
- We tested ~20. Build automated sweep like GEMM.
- **Focus on d=2048 (E=33, bs=512): 333μs bottleneck. Even 10% improvement helps.**
- **IMMEDIATE TEST: Remove our block_m override entirely!** "no_blocksizeM" at 145.8μs beats us at 169μs.
- Also try: no CK kernel injection (let defaults handle everything) — our injection may be HURTING

## COMPETITOR ANALYSIS (from leaderboard filenames)

### GEMM competitors:
- Top 3-10 all use AUTOMATED CONFIG SWEEPING (v788, solution_514, v469 = hundreds of versions)
- **`splitk0`** (Bortlesboat, 8.9μs): SPLITK=0 (AUTO) beats forced split-K!
- **`patched_a16wfp4`** (olezhka_007, 10.5μs): monkey-patching aiter's kernel
- **`cfgsearch`** (oofbaroomf, 10.7μs): automated config search
- **`tunedcsv`** (ry2009, 13.3μs): custom CSV config override via env var

### MoE competitors:
- **`no_blocksizeM`** (romepen788, 145.8μs): NOT overriding block_m = BETTER than us (169μs)!
- **`selective_quant/splitk`**: per-shape selective configs, not global
- **`noopus_257bs512k1`**: disabling OPUS for specific shapes
- **`selective_ksplit4`**: ksplit=4 for specific shapes
- josusanmartin: 1084 MoE versions = massive automated sweeping
- **OUR BLOCK_M OVERRIDE MAY BE HURTING US** — try removing it!

### MLA competitors:
- **`pg2_fix`** (Ananda Sai A, 33μs): pg2 made reliable somehow
- **`pg2_hybrid`** (John Hahn, 32μs): pg2 hybrid approach
- **`pingpong.py`** (willfisher, 35μs): Triton BLOCK_PINGPONG env var
- **`direct_stage`** (Yufeng98, 37μs): calling stage1/reduce directly

## KEY TECHNICAL REFERENCE
- gemm_a16wfp4(A, w, w_scales, dtype, y=None, config=None) — pass config dict
- Config keys: BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, num_warps, num_stages, waves_per_eu, matrix_instr_nonkdim, cache_modifier, NUM_KSPLIT, SPLITK_BLOCK_SIZE
- B_scale must be UNSHUFFLED (use _unshuffle_e8m0)
- K=1536 uses gemm_afp4wfp4 (separate quant path)
- The word "stream" is GREP-FILTERED from submissions

## AUTONOMOUS SCRIPTS (run independently of Claude Code)
Start these in separate terminal tabs. They run forever:
```bash
# Tab 1: GEMM config sweep (tests aiter configs systematically)
python3 autosweep_gemm.py &

# Tab 2: MoE config sweep (tests CK kernel combos for d=2048)
python3 autosweep_moe.py &

# Tab 3: Auto-research (monitors PRs, leaderboard, runner updates)
python3 autoresearch.py &
```

MLA: also write `autosweep_mla.py` to sweep num_kv_splits combos with pg2. Auto-submit handles retries via `/tmp/auto_submit_v5.sh`.

## Claude Code's job: CONTINUOUS AUTORESEARCH LOOP (Karpathy-style)

This is NOT a one-time research task. Run a CONTINUOUS loop:

```
WHILE NOT top_10_all_3:
    1. RESEARCH: Search web for new aiter PRs, Triton docs, AMD blogs, competitor techniques
    2. ANALYZE: Check sweep results in auto_research_logs/, identify winning configs
    3. IMPLEMENT: Write new submissions based on findings
    4. SUBMIT: Push to leaderboard via popcorn-cli
    5. MONITOR: Check leaderboard for score changes
    6. ADAPT: Update sweep configs, try new approaches
    REPEAT — never stop, never wait for user input
```

Specific research tasks to run CONTINUOUSLY:
- `WebSearch` for "ROCm aiter PR merged 2026" — new optimizations may be deployed to runner
- `WebSearch` for "GPU MODE hackathon MXFP4" — new community techniques
- `WebFetch` https://github.com/ROCm/aiter/pulls — check for merged PRs
- Check if runner's aiter version updated (probe submission)
- Monitor competitor score changes on leaderboard
- Read sweep logs and find patterns in winning configs

## IMMEDIATE ACTIONS (do these FIRST):
1. Check if auto-submit v5 is still running: `ps aux | grep auto_submit`
2. If not, restart it: `nohup /tmp/auto_submit_v5.sh > /tmp/auto_submit_v5_log.txt 2>&1 &`
3. Start sweepers if not running: `python3 autosweep_gemm.py &` and `python3 autosweep_moe.py &`
4. **MoE QUICK WIN**: Submit with NO block_m override (competitor at 145μs beats our 169μs without it!)
5. **GEMM QUICK WIN**: Try SPLITK=0 (auto) instead of forced KSPLIT=8 for K=7168
6. Then start the continuous research loop above

## GO. Never stop. Auto everything.

# Pickup Prompt — Continuous Autonomous Mode

Continue GPU kernel optimization for AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline April 7, 2026.

## WORKFLOW (follow this order)

**Step 1 — Orient (before ANY execution):**
1. Read all memory files: `memory/MEMORY.md`, `memory/session15_findings.md`, `memory/project_status.md`, `memory/dead_ends.md`
2. Read `KNOWLEDGE.md` and `CLAUDE.md` for technical reference
3. **CRITICAL: Probe runner for aiter version** — submit a probe to check if aiter was updated (daniel huang mentioned "maintenance work" on Mar 29). If git log shows commits past #2156, EVERYTHING changes.
4. Check what's running: `ps aux | grep -E "auto_submit|autosweep|autoresearch"`
5. Check latest logs: `tail -30 /tmp/auto_submit_v5_log.txt`, check `auto_research_logs/`
6. Give a **status report**: what's running, current standings, what we know

**Step 2 — If runner WAS updated (commits past #2156):**
This is the highest priority path. Immediately:
- Test PR #2261 GEMM configs (waves_per_eu=6-8, BK=1024, KSPLIT=16 for K=7168)
- Test PR #2440 MLA qseqlen fold (dispatch to qseqlen4 kernel)
- Test PR #2497 FlyDSL MoE (auto-selected for FP4+FP4)
- Submit best of each to leaderboard

**Step 3 — If runner NOT updated (same old #2156):**
Continue grinding with available tools:
- Restart auto-submit loop (MLA pg2 rotation, GEMM/MoE best)
- Restart GEMM batch sweeper (autosweep_gemm_batch.py, 3 configs/batch, occupancy filter)
- Restart autoresearch (web monitoring)
- Focus GEMM sweep on K=7168 configs (biggest impact on geomean)
- MoE is stuck (d=2048 injection always crashes, no FlyDSL binaries)

**Step 4 — Go fully autonomous:**
- Don't ask permission for individual actions
- Auto-submit, auto-research, auto-iterate
- If something fails → fix it and keep going
- Only stop when top 10 aggregate or I tell you to stop

## Current Standings (Mar 29)
- **GEMM**: 16.22μs rank #154/292 → need 8.4μs for top 10. **2x gap.**
- **MLA**: 42.28μs rank #12/154 → need ~41μs for top 10. **0.8μs gap.**
- **MoE**: 169.13μs rank #60/201 → need 129μs for top 10. **24% gap.**
- **Aggregate**: NOT in top 20. Need competitive on ALL 3 for Phase 2.

## Submission Infrastructure
- popcorn-cli at `~/.local/bin/popcorn-cli`, always use `--no-tui`
- Rate: 6 benchmark/hour, 1 leaderboard/hour per problem (NOT 10 as docs say)
- Header: `#!POPCORN leaderboard <name>` (NOT benchmark!)
- Leaderboard names: `amd-mxfp4-mm`, `amd-mixed-mla`, `amd-moe-mxfp4`
- Runner timeout: 12 minutes total (JIT + benchmark must fit)

## Auto-Submit Loop
```bash
nohup /tmp/auto_submit_v5.sh > /tmp/auto_submit_v5_log.txt 2>&1 &
```
Submits hourly to leaderboard rotating:
- GEMM: `submission_prewarm.py`
- MLA: rotates `submission_pg2_pingpong.py`, `exp_optimal_splits.py`, `submission_pg8_v2.py`
- MoE: `submission_optimized_v2.py`

## Autonomous Scripts
```bash
nohup python3 -u autosweep_gemm_batch.py > auto_research_logs/gemm_batch_sweep.log 2>&1 &
nohup python3 -u autoresearch.py > auto_research_logs/autoresearch_run.log 2>&1 &
```

## Current Best Submissions
- GEMM: `mxfp4-mm/submission_prewarm.py` (a16wfp4 + K7168 config + prewarm)
- MLA: `mixed-mla/submission_pg8_v2.py` (pg8 for kv>=8192, pg1 for kv<=1024)
- MoE: `moe-mxfp4/submission_optimized_v2.py` (CK injection for E<=64, use_nt=False)

## What Works (proven)
- gemm_a16wfp4 with config= parameter for K=7168 tuning
- gemm_afp4wfp4 for K=1536 (separate quant path)
- Library defaults for K=512 shapes (ALL custom configs worse)
- MLA pg8 for kv>=8192 + pg1 for kv<=1024
- MoE CK 2-stage pipeline with monkey-patched kernel injection (E<=64 d<2048 only)
- Prewarm all Triton shapes to avoid JIT penalty
- GEMM batch config sweeper (3 configs/submission, occupancy filter >= 128 blocks)

## Dead Ends (Session 15 confirmed — DON'T retry)
- PR #2261 configs on old runner (2-7x regression)
- qseqlen4/2 MLA fold (hangs or fails accuracy on old runner)
- MoE d=2048 kernel injection (GPU crash, S2_256 not on runner)
- MoE block_m override for d>=2048 (GPU crash)
- afp4wfp4 for all shapes (64-89% worse than a16wfp4 for K=512)
- Custom Triton GEMM kernels (JIT overhead negates speed gains)
- HIP MFMA kernel (correct but slower than Triton)
- MLA micro-optimizations (BLOCK=2048, kv view caching = <1% difference)
- pg2 for kv<=1024 (fails accuracy on secret seeds)
- GEMM per-shape hand-tuned configs for K=512 (all worse than defaults)

## Runner State (as of Mar 28 probe)
- Commit: f3be04a12 (#2156) — OLD
- qseqlen2/4 kernel .co files: EXIST but dispatch code doesn't use them
- FlyDSL: 27 references in code, 0 binaries
- Total .co files: ~200+
- MLA kernels: a16w8_ps, a8w8_qseqlen1_ps (current), a8w8_qseqlen2_ps, a8w8_qseqlen4_ps (unused)

## Leaderboard Intel
- Top GEMM/MLA have EXPLOIT entries (timing artifact via tinygrad comgr)
- Organizers scrub periodically but exploits reappear
- ZainHaider20 (1μs GEMM), HorizonLiang (4.36μs GEMM) = confirmed exploits
- Danishlynx (26.66μs MLA) = borderline, might be exploit
- josusanmartin (7.74μs GEMM, 31.16μs MLA) = legit via 4814+ submissions of config sweeping
- Phibi stuck at 147μs MoE despite extensive effort (FlyDSL, aiter kernels, custom)
- Aggregate top 10 cutoff: ~1275/3750

## Research — Search BROADLY
- AMD blogs (rocm.blogs.amd.com), salykova.github.io
- ROCm/aiter GitHub PRs, issues, commits
- Triton docs, triton-lang GitHub
- GPU MODE Discord logs in `discord-logs/`
- Leaderboard mirror: leaderboard.ooousay.com
- Any blog, paper, tutorial with MI355X/gfx950/CDNA4 optimization insights

## What Would Change Everything
1. **Runner aiter update** → PRs #2261, #2440, #2497 unlock all 3 problems
2. **MLA qseqlen fold working** → 10-15μs improvement on MLA
3. **Better GEMM K=7168 config** → each 1μs saves ~0.15μs on geomean
4. **FlyDSL binaries deployed** → MoE improvement

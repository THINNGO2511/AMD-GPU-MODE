---
name: Project Status
description: AMD GPU MODE hackathon standings, bottlenecks, and session history as of 2026-03-27
type: project
---

**Deadline**: 2026-04-07 07:59 UTC (10 days from Mar 27)
**Goal**: Top 10 (ideally top 5) on all 3 leaderboards for Phase 2 advancement
**GitHub**: https://github.com/THINNGO2511/AMD-GPU-MODE

## Current Standings (as of Session 10, Mar 27 2026)

| Problem | Leaderboard | Our Score | Our Rank | Top Score | Gap |
|---------|------------|-----------|----------|-----------|-----|
| GEMM | amd-mxfp4-mm | 16.2μs | ~130th | ~7.7μs (josusanmartin) | 2x |
| MoE | amd-moe-mxfp4 | 169μs | ~52nd | ~110μs (Ananda Sai A) | 1.5x |
| MLA | amd-mixed-mla | ~42μs→pending | ~10th | ~33μs (Ananda Sai A) | 1.3x |

**MLA leaderboard PASSED**: `exp_pg1_safe.py` ranked 45.5μs (pg1 safe approach, was 42.5μs with pg2 risky)
**MoE no-opus benchmark**: 170μs — same as baseline, opus sorting is NOT the bottleneck

## Best Submissions on Leaderboard
- **GEMM**: `submission_prewarm.py` (16.2μs) — gemm_a16wfp4 + K7168 config + K1536 afp4wfp4
- **MoE**: `submission_opus_sort.py` (169μs) — opus sorting + CK injection for E≤64 d<2048
- **MLA**: `exp_optimal_splits.py` (41.6μs benchmark, leaderboard pending) — a16w8+pg2 kv≤1024, a8w8+pg8 kv≥8192, optimal kv_splits

## Key Bottlenecks (Session 10 assessment)
- GEMM: aiter wrappers exhausted at ~10μs. Need CUSTOM Triton MXFP4 kernel from scratch. 2x gap.
- MoE: d=2048 at 333μs kills geomean. Every injection attempt failed. Try blockPerCu, cktile, native runner update.
- MLA: pg1 safe at 45.5μs. Hybrid pg2/pg1 by batch size could recover to ~42μs. Need radical approach for 33μs.

**Why:** Top 5 advance to Phase 2 ($1.1M prize pool, production kernel work on DeepSeek-R1)
**How to apply:** Maximum parallel effort with multiple agents. Write new code, don't tune existing wrappers. Focus on breakthroughs.

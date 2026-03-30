---
name: Session 10 Results
description: Mar 27 2026 PM — MLA pg1 leaderboard pass, GEMM/MoE stalled, strategic pivot planned
type: project
---

## Session 10 Final Summary (Mar 27 2026)

### Results
- **MLA**: pg1-safe LEADERBOARD PASSED at 45.5μs (was 42.5μs with pg2). pg2 for kv=1024 FAILS leaderboard.
- **MoE**: No improvement found. No-opus (170μs), quant threshold (168.8μs), PR2261 wider stage2 (175μs) all same or worse.
- **GEMM**: deepgemm_ck dead end. Triton env vars already default. batched/fused APIs not useful for our shapes.
- **Research**: Found PR #2261 (tuned configs), PR #2440 (MLA kernel update), unofficial leaderboard data.

### Strategic Assessment — ALL 3 PROBLEMS STALLED
- GEMM: 16μs, need 8μs (2x gap). Easy optimizations exhausted. Need custom Triton kernel.
- MoE: 169μs, need 136μs (20% gap). d=2048 at 333μs is bottleneck. All injection attempts failed.
- MLA: 45.5μs, need 33μs (27% gap). Regressed from 42.5μs by switching to pg1 for reliability.

### Strategic Pivot Agreed (end of session)
User approved maximum-effort parallel approach:
1. **GEMM: Write custom Triton MXFP4 kernel from scratch** (highest priority, highest risk/reward)
2. **GEMM: Check if runner aiter updated** with PR #2261 retuned KSPLIT=16 configs
3. **MoE: d=2048 blitz** — blockPerCu, cktile, native runner config check
4. **MLA: Hybrid pg2/pg1 by batch size** — pg2 for bs<=32 (safe), pg1 for bs>=64
5. **Research: Competitor reverse engineering** — what do 8μs GEMM and 27μs MLA competitors use
6. **Research: aiter source deep dive** — find hidden optimization parameters

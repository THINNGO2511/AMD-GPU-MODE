# Session Pickup Prompt
Copy-paste everything below into a new Claude Code session:

---

Resume autonomous GPU kernel optimization research for the AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline is April 7, 2026.

**Mode: Fully autonomous. Don't ask permission. Just execute. Submit experiments, do web research, iterate.**

## Context
Read KNOWLEDGE.md and check your memory files for full context. Key points:
- 3 problems: GEMM (amd-mxfp4-mm), MoE (amd-moe-mxfp4), MLA (amd-mixed-mla)
- popcorn-cli v1.3.6 at ~/.local/bin/popcorn-cli, use --no-tui flag
- Rate limits: 10 benchmark/hr, 1 leaderboard/hr per problem

## Current Standings
- **GEMM**: 16.2μs (rank ~130), target ~8μs. Triton path near-optimal.
- **MoE**: 169μs (rank ~52), target ~130μs. d=2048 shape is bottleneck.
- **MLA**: 42.5μs (rank #10), target ~33μs. Submitted ~40μs (exp_optimal_splits.py).

## Immediate Actions (check memory/active_leads.md for full list)
1. **MLA**: Submit `mixed-mla/exp_optimal_splits.py` to leaderboard if not yet done
2. **MoE**: Check results of `moe-mxfp4/exp_blockmwide_stage2.py` benchmark (submitted last session)
3. **MoE**: Try NO opus sorting for E=257, per-shape block_m sweep
4. **GEMM**: Probe deepgemm_ck API properly (unknown input format)
5. **All**: Continue web research for competitor techniques and new aiter APIs

## Workflow
Run parallel experiments across all 3 problems. Use web search for research. Check memory/dead_ends.md before trying anything — many approaches have been ruled out. Log results. Submit to leaderboard when benchmark shows improvement. Keep iterating 24/7.

Go.

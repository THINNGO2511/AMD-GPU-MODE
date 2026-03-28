---
name: Feedback - Research Strategy
description: User wants maximum parallel effort with many agents, big breakthroughs not incremental tuning
type: feedback
---

User wants MAXIMUM parallel effort — spawn 5-6 agents simultaneously across all 3 problems.

**Why:** Session 10 showed incremental tuning (env vars, kernel injection, quant thresholds) is exhausted. Every small tweak yields 0-4% change. Need fundamentally new approaches to achieve 2x (GEMM), 20% (MoE), 27% (MLA) improvements.

**How to apply:**
- Don't spend time on incremental config tuning — it's been exhausted
- Write NEW code (custom Triton kernels, new algorithms) rather than tweaking existing wrappers
- Run many agents in parallel: research, code generation, submission all simultaneously
- User wants autonomous 24/7 operation — don't ask permission, just execute
- Only stop when top 5 on all 3 leaderboards or user intervenes
- Prioritize high-risk/high-reward approaches over safe incremental ones

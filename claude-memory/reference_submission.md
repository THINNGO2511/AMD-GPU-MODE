---
name: Submission Workflow
description: How to submit to GPU MODE hackathon via popcorn-cli, rate limits, eval details
type: reference
---

**Submit command**: `popcorn-cli submit --gpu MI355X --leaderboard <name> --mode <mode> <file>`
- Modes: test, benchmark, leaderboard, profile
- Rate limits: 10 benchmark/hr per problem, 1 leaderboard/hr per problem
- Leaderboard names: amd-mxfp4-mm, amd-moe-mxfp4, amd-mixed-mla
- Leaderboard mode: rotates seeds (secret), re-checks correctness each iteration
- Benchmark mode: fixed seeds, no correctness re-check during timing
- Word "stream" in code is auto-rejected by grep filter
- Pre-allocated output buffer reuse is considered borderline (reward hack territory)
- eval.py warms only tests[0] shape — other shapes may trigger Triton JIT
- L2 cache is cleared between benchmark iterations

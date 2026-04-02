RESEARCH-BACKED EXPERIMENTS. Copy and submit as-is. Do NOT modify.

Setup:
cp claude-research-files/gemm_asm_envvars.py mxfp4-mm/
cp claude-research-files/gemm_a4w4_preshuffle.py mxfp4-mm/
cp claude-research-files/mla_splits1_fastpath.py mixed-mla/
cp claude-research-files/moe_online_tune.py moe-mxfp4/

Submit ALL in parallel:

GEMM (2 experiments — HIGHEST PRIORITY):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_asm_envvars.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_a4w4_preshuffle.py --no-tui

MLA:
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_splits1_fastpath.py --no-tui
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

MoE:
5. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_online_tune.py --no-tui
6. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_optimized_v2.py --no-tui

KEY CONTEXT:
- gemm_asm_envvars: env vars might switch Triton → ASM kernel dispatch (2x potential)
- gemm_a4w4_preshuffle: uses B_shuffle input with gemm_a4w4 new API. Will print error to stdout if fails. READ THE STDOUT.
- mla_splits1_fastpath: num_kv_splits=1 for bs<=4 bypasses Stage 2 reduction
- moe_online_tune: HIP_ONLINE_TUNING for runtime kernel benchmarking
- MLA leaderboard: keep resubmitting mla_fp8q_all.py (variance ratcheting for 34.5μs)
- MoE leaderboard: keep resubmitting submission_optimized_v2 (need 163μs on board)

After results:
- GEMM: if either passes test AND benchmarks < 15.7μs → leaderboard immediately
- gemm_a4w4_preshuffle: READ STDOUT — if gemm_a4w4 actually works, report the exact API and timing
- MLA: if splits1 passes → benchmark. If < 36.4μs → leaderboard
- MoE: if online_tune passes → benchmark

CRITICAL: Do NOT rewrite files. Report all stdout output especially from gemm_a4w4_preshuffle.

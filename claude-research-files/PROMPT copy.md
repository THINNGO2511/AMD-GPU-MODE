New round of experiments. Copy to correct dirs, submit as-is. Do NOT modify files.

Setup:
cp claude-research-files/mla_fp8q_all.py mixed-mla/
cp claude-research-files/mla_splits8.py mixed-mla/
cp claude-research-files/moe_d2048_bm32.py moe-mxfp4/
cp claude-research-files/gemm_afp4_k2048.py mxfp4-mm/
cp claude-research-files/gemm_lite_probe.py mxfp4-mm/

Submit ALL in parallel (separate rate limit pools per problem):

MLA:
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_fp8q_all.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_splits8.py --no-tui

MoE:
3. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_d2048_bm32.py --no-tui

GEMM:
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_afp4_k2048.py --no-tui
5. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/gemm_lite_probe.py --no-tui

After results:
- MLA: whichever passes test AND benchmarks faster than 41.5μs → leaderboard it
- MoE: if moe_d2048_bm32 passes test → benchmark it. If benchmark < 163μs → leaderboard
- GEMM: gemm_afp4_k2048 if passes test → benchmark. gemm_lite_probe → read stdout for deepgemm/ASM paths
- If combining MLA improvements, ask me first

CRITICAL: Do NOT rewrite files. Do NOT add new Triton kernels. Submit exactly as-is.
Report per-shape times if visible in stdout.

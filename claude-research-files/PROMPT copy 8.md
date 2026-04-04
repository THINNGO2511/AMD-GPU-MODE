GEMM BREAKTHROUGH — ASM kernel with perfect accuracy confirmed. Test and benchmark IMMEDIATELY.

Setup:
cp claude-research-files/gemm_a4w4_clean.py mxfp4-mm/
cp claude-research-files/gemm_a4w4_hybrid.py mxfp4-mm/

Submit BOTH as test first, then BENCHMARK the one that passes:

1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_a4w4_clean.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_a4w4_hybrid.py --no-tui

After test results:
- If BOTH pass: benchmark BOTH, compare scores
- If only one passes: benchmark that one
- Whichever benchmarks FASTER than 15.7μs: LEADERBOARD IMMEDIATELY
- Report exact per-shape times if available

Also keep running:
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

EXPLANATION:
- gemm_a4w4_clean: ALL shapes via ASM kernel (gemm_a4w4 with shuffled A_scale)
- gemm_a4w4_hybrid: K=7168/2048 via ASM, K=512 via Triton, K=1536 via afp4wfp4

The "clean" version might be faster overall (single code path, fewer branches).
The "hybrid" version hedges in case ASM is slower for small K.

CRITICAL: Do NOT modify these files. The data format recipe is PROVEN (0 errors).

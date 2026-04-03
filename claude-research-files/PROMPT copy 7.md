CRITICAL GEMM EXPERIMENT. gemm_a4w4 RUNS — we just need correct data format.

Setup:
cp claude-research-files/gemm_a4w4_fix.py mxfp4-mm/
cp claude-research-files/moe_csv_merged.py moe-mxfp4/

Submit:

GEMM (HIGHEST PRIORITY — READ ALL STDOUT):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_a4w4_fix.py --no-tui

MoE:
2. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_csv_merged.py --no-tui

MLA (ratchet):
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

CRITICAL — GEMM stdout:
The gemm_a4w4_fix file tries 6 different combinations of:
- A raw vs A shuffled (shuffle_weight)
- A_scale as e8m0 vs A_scale shuffled (e8m0_shuffle)
- B_shuffle+bpre=1 vs B_q+bpre=0

For EACH combo it prints: max_err, mean_err, and count of wrong elements.
REPORT ALL 6 LINES. If ANY combo shows fewer than 100 wrong elements, that is the fix.

After results:
- GEMM: if any combo passes accuracy, we have the 8us kernel. Benchmark immediately.
- MoE: if csv_merged passes (check E=257 shapes still work), benchmark it.
- MLA: check ratchet score.

Do NOT rewrite files. Report ALL [GEMM] stdout lines.

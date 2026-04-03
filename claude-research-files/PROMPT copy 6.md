LAST UNTRIED API PATHS. Copy and submit. Do NOT modify.

Setup:
cp claude-research-files/gemm_fp4x2_view.py mxfp4-mm/
cp claude-research-files/moe_csv_e33_correct.py moe-mxfp4/

Submit:

GEMM (KEY EXPERIMENT — read stdout carefully!):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_fp4x2_view.py --no-tui

MoE:
2. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_csv_e33_correct.py --no-tui

MLA (keep ratcheting):
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

After results:
- GEMM fp4x2_view: READ ALL STDOUT. It tries 4 different API signatures for gemm_a4w4.
  If ANY attempt says "SUCCESS": benchmark immediately, then leaderboard.
  If all fail: report the exact error messages for each attempt.
  Also report what torch.ops.aiter a4w4/asm ops are listed.
- MoE csv_e33: if passes test → benchmark. If < 163μs → leaderboard.
- MLA: variance ratchet. Check if score < 36.4μs.

CRITICAL: Do NOT rewrite files. Report ALL stdout output from gemm_fp4x2_view — every [GEMM] line matters.

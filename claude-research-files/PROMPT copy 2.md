New experiments. Copy to correct dirs, submit as-is. Do NOT modify files.

Setup:
cp claude-research-files/mla_tuned_v2.py mixed-mla/
cp claude-research-files/mla_pg2_fp8q.py mixed-mla/
cp claude-research-files/moe_e257_bm_tune.py moe-mxfp4/

Submit ALL in parallel:

MLA:
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_tuned_v2.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_pg2_fp8q.py --no-tui

MoE:
3. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_e257_bm_tune.py --no-tui

After results:
- MLA: if mla_tuned_v2 passes test → benchmark → leaderboard if < 36.4μs
- MLA: if mla_pg2_fp8q passes test → benchmark → leaderboard (HIGH RISK: might fail secret seed like all pg2 attempts. Only leaderboard if TEST shows <3% mismatch on kv=1024)
- MoE: if moe_e257_bm_tune passes test → benchmark → leaderboard if < 163μs

CRITICAL: Do NOT rewrite files. Submit exactly as-is. Report mismatch ratios from test output for mla_pg2_fp8q (we need to see kv=1024 accuracy before risking a leaderboard slot).

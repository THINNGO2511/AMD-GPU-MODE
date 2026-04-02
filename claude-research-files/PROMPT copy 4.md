CRITICAL: These experiments test specific hypotheses. Submit as TEST first and report ACCURACY details (mismatch ratios, max errors) BEFORE benchmarking. Do NOT modify files.

Setup:
cp claude-research-files/mla_pg2_fix_v1.py mixed-mla/
cp claude-research-files/mla_pg2_fix_v2.py mixed-mla/
cp claude-research-files/gemm_lib_defaults.py mxfp4-mm/
cp claude-research-files/moe_sort_policy.py moe-mxfp4/

Submit ALL in parallel:

MLA (THE CRITICAL EXPERIMENTS — check accuracy carefully!):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_pg2_fix_v1.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_pg2_fix_v2.py --no-tui

GEMM:
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_lib_defaults.py --no-tui

MoE:
4. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_sort_policy.py --no-tui

AFTER MLA TEST RESULTS — THIS IS CRITICAL:
For mla_pg2_fix_v1 and v2, report EXACTLY:
- Does it PASS test?
- What is the kv=1024 mismatch ratio? (look for "mismatch" in stdout)
- What is the max error on kv=1024 shapes?
- If mismatch < 2% on kv=1024: BENCHMARK immediately, then LEADERBOARD
- If mismatch 2-4%: benchmark but DON'T leaderboard (too risky)
- If mismatch > 4% or FAILS: report the error and move on

The pg2_fix_v1 tests a potential BUG in our kv_indptr handling.
If it works, this is the breakthrough that gets us from 36μs to 32-33μs.

CRITICAL: Do NOT rewrite files. Report exact accuracy numbers.

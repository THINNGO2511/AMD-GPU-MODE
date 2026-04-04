SHOTGUN ROUND — 7 experiments + 1 ratchet. Copy and submit ALL in parallel. Do NOT modify files.

Setup:
cp claude-research-files/gemm_lib_defaults.py mxfp4-mm/
cp claude-research-files/gemm_preshuffle.py mxfp4-mm/
cp claude-research-files/gemm_lazy_prewarm.py mxfp4-mm/
cp claude-research-files/gemm_afp4wfp4_all.py mxfp4-mm/
cp claude-research-files/moe_1stage_d2048.py moe-mxfp4/
cp claude-research-files/moe_splitk_d2048.py moe-mxfp4/
cp claude-research-files/mla_mxfp4_kv.py mixed-mla/

Submit ALL as TEST first:

GEMM (4 experiments):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_lib_defaults.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_preshuffle.py --no-tui
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_lazy_prewarm.py --no-tui
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_afp4wfp4_all.py --no-tui

MoE (2 experiments):
5. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_1stage_d2048.py --no-tui
6. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_splitk_d2048.py --no-tui

MLA (1 experiment + ratchet):
7. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_mxfp4_kv.py --no-tui
8. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

AFTER TEST RESULTS — benchmark anything that passes:

GEMM: benchmark ALL that pass test. Report per-shape times for each.
- gemm_lib_defaults: if benchmark < 9.6μs geomean → leaderboard
- gemm_preshuffle: READ STDOUT — reports API probing results
- gemm_lazy_prewarm: compare vs current 15.7μs leaderboard score
- gemm_afp4wfp4_all: if K=7168 time drops significantly → leaderboard

MoE: benchmark both if they pass.
- moe_1stage_d2048: check if d=2048 shape gets faster
- moe_splitk_d2048: check if d=2048 shape gets faster

MLA:
- mla_mxfp4_kv: READ STDOUT for "[MLA] MXFP4 KV SUCCESS" or failure reason. If succeeds → benchmark → if < 36μs → leaderboard

CRITICAL: Do NOT rewrite files. Report ALL stdout from gemm_preshuffle and mla_mxfp4_kv.

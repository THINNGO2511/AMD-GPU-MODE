I have new submission files in claude-research-files/. Copy them to the correct problem directories and submit. Do NOT modify any file contents.

Setup:
cp claude-research-files/mla_safe_fast.py mixed-mla/
cp claude-research-files/mla_skip_amax_ratchet.py mixed-mla/
cp claude-research-files/moe_d2048_medtile.py moe-mxfp4/
cp claude-research-files/gemm_stages3_k512.py mxfp4-mm/

Then run ALL of these (don't wait between problems — they use separate rate limit pools):

1. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_safe_fast.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_skip_amax_ratchet.py --no-tui
3. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_d2048_medtile.py --no-tui
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/gemm_stages3_k512.py --no-tui

AFTER RESULTS:
- If mla_safe_fast passes test: benchmark it, then leaderboard it
- If moe_d2048_medtile passes test: benchmark it
- If mla_skip_amax_ratchet fails secret seed (67% chance): retry it next hour
- If mla_skip_amax_ratchet PASSES: that's 38.7μs locked in, huge win

CRITICAL: Do NOT rewrite or improve any file. Submit exactly as-is. No new Triton kernels.

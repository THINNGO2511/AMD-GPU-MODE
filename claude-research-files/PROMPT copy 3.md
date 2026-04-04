New experiments + auto-ratchet setup. Copy to correct dirs, submit as-is. Do NOT modify files.

Setup:
cp claude-research-files/moe_d2048_s1small.py moe-mxfp4/
cp claude-research-files/moe_d2048_s2large.py moe-mxfp4/
cp claude-research-files/gemm_l2_prewarm.py mxfp4-mm/
cp claude-research-files/auto_mla_ratchet.sh .

Start the MLA auto-ratchet in background (submits mla_fp8q_all.py to leaderboard every 65 min):
chmod +x auto_mla_ratchet.sh
# NOTE: mla_fp8q_all.py should already be in mixed-mla/ from the previous round.
# Start the ratchet loop:
bash auto_mla_ratchet.sh &

Submit experiments (all different rate limit pools):

MoE (2 experiments targeting d=2048 at 333μs):
1. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_d2048_s1small.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe-mxfp4/moe_d2048_s2large.py --no-tui

GEMM (L2 cache warming):
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_l2_prewarm.py --no-tui

MLA (auto-ratcheting handles this — also submit manually once now):
4. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

After results:
- MoE: whichever passes test → benchmark. If < 163μs → leaderboard immediately
- GEMM: if passes test → benchmark. If < 15.7μs → leaderboard
- MLA: auto-ratchet handles continuous submissions. Check log: cat auto_mla_ratchet.log

CRITICAL: Do NOT rewrite files. Submit exactly as-is. No new Triton kernels.

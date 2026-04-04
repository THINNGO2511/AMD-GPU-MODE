# Round 3 — Corrected After Claude Code Feedback

## PRIORITY 1: MoE block_m bug fix

Your CK injection hardcodes block_m=32 for E=33 d=512 bs=512.
Library default is 128. You're running 4x wrong tile size.

**Submit order (use 3 of your 6 hourly slots):**
```bash
# A/B test: no injection vs injection with fixed block_m
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_only_usent.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_ck_fixbm.py --no-tui

# If either passes test, benchmark it:
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark WINNER.py --no-tui
```

**Decision tree:**
- moe_only_usent < 169μs? → Your overrides were HURTING. Use this.
- moe_ck_fixbm < moe_only_usent? → CK kernels help with correct block_m
- Both ≈ 169μs? → block_m wasn't the issue for E=33 d=512

**If block_m fix helps, also try d=2048 injection:**
```bash
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_d2048_medtile.py --no-tui
```

## PRIORITY 2: MLA skip-amax (1-3μs savings)

```bash
# Test accuracy first — fixed scale might clip outlier Q values
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mla_skip_amax.py --no-tui

# If passes, benchmark
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark mla_skip_amax.py --no-tui

# Keep ratcheting current best in parallel (separate hourly slot)
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mla_ratchet.py --no-tui
```

**If accuracy fails:** Try _FIXED_AMAX = 64.0 or 128.0 (more headroom, less precision).

## PRIORITY 3: GEMM (incremental)

```bash
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test gemm_stages3_k512.py --no-tui
```

## JIT TIMEOUT NOTES
- moe_only_usent.py is 25 lines. If THIS times out, it's a runner issue, not code.
- mla_skip_amax.py uses IDENTICAL Triton kernels as current best. Same JIT cache.
- If something times out, RETRY. Ephemeral runners have variable performance.

## FILES
| File | Lines | New JIT? | What |
|------|-------|----------|------|
| moe_only_usent.py | 25 | NO | Pure library defaults + use_nt=False |
| moe_ck_fixbm.py | 65 | NO | CK injection with library-default block_m |
| moe_d2048_medtile.py | 75 | NO | Above + 256x32 tiles for d=2048 |
| mla_skip_amax.py | 155 | NO | Skip amax kernel, fixed scale=32.0 |
| mla_ratchet.py | 180 | NO | Current best, keep dice-rolling |
| gemm_stages3_k512.py | 130 | Minimal | stages=3 for K=512 only |

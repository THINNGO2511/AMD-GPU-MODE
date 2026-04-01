# Experiment Submission Guide — April 1, 2026

## SUBMIT ORDER (highest impact first)

### STEP 1: Probes (submit as BENCHMARK first to see stdout)
```bash
# MoE d=2048 probe — learn what kernel runs for the bottleneck shape
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe_exp_probe_d2048.py --no-tui

# GEMM deep probe — discover ASM/deepgemm API paths
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark gemm_exp_deep_probe.py --no-tui
```

### STEP 2: MoE experiments (submit as TEST first, then BENCHMARK)
```bash
# MoE minimal (no overrides, just use_nt=False) — test if our overrides are HURTING
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_minimal.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe_minimal.py --no-tui

# MoE sort caching — save ~16% from sort overhead
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_sort_cache.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe_sort_cache.py --no-tui

# MoE custom CSV — force configs for d=2048
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode test moe_custom_csv.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode benchmark moe_custom_csv.py --no-tui
```

### STEP 3: MLA experiments (submit as TEST first — accuracy matters!)
```bash
# MLA pg2 for ALL shapes — THE key question: does corrected formula fix kv=1024?
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mla_pg2_all.py --no-tui

# MLA fixed-scale quant — saves one kernel launch for kv>=8192
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mla_fixed_scale.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark mla_fixed_scale.py --no-tui

# MLA aggressive — pg2 everywhere + fp8 Q everywhere + fixed scale (dice roll)
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mla_aggressive.py --no-tui
```

### STEP 4: GEMM experiments
```bash
# GEMM stages=3 for K=512 only (proven -6% for K=512, no regression on K=2048)
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test gemm_stages3_k512.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark gemm_stages3_k512.py --no-tui

# GEMM afp4wfp4 for K=2048 — test if separate quant+GEMM is faster
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test gemm_afp4_k2048.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark gemm_afp4_k2048.py --no-tui
```

### STEP 5: LEADERBOARD (only submit winners)
```bash
# Whichever test+benchmark shows improvement:
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard WINNER.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard WINNER.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard WINNER.py --no-tui
```

## RATE LIMITS
- 6 submissions/hr per problem (test + benchmark + leaderboard SHARE this pool)
- 1 leaderboard submission/hr per problem
- Budget carefully: 2 probes + 3 MoE + 3 MLA + 2 GEMM = 10 submissions across 3 problems

## FILES
| File | Problem | Type | What it tests |
|------|---------|------|---------------|
| moe_exp_probe_d2048.py | MoE | Probe | d=2048 kernel selection + CSV format |
| moe_sort_cache.py | MoE | Experiment | Cache sort results between stages |
| moe_minimal.py | MoE | Experiment | No overrides (test if we're hurting ourselves) |
| moe_custom_csv.py | MoE | Experiment | Force configs via AITER_CONFIG_FMOE |
| mla_pg2_all.py | MLA | Experiment | pg2 for ALL shapes with correct formula |
| mla_fixed_scale.py | MLA | Experiment | Fixed-scale fp8 quant (1 kernel vs 2) |
| mla_aggressive.py | MLA | Experiment | pg2+fp8+fixed scale everywhere (max risk) |
| gemm_exp_deep_probe.py | GEMM | Probe | Discover ASM/deepgemm paths |
| gemm_stages3_k512.py | GEMM | Experiment | stages=3 for K=512 only |
| gemm_afp4_k2048.py | GEMM | Experiment | afp4wfp4 path for K=2048 |

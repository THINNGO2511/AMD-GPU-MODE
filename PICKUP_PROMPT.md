# Pickup Prompt — Session 17

Continue GPU kernel optimization for AMD x GPU MODE hackathon. I'm noobmaster69_og. Deadline April 7, 2026.

## CURRENT SCORES (Mar 30)
- **GEMM**: 16.06μs rank ~158 → need <9μs for top 10 (2x gap)
- **MLA**: 42.18μs rank ~12 → need <37μs for top 10 (pg2+bf16Q approach works)
- **MoE**: 169.13μs rank ~61 → need <129μs for top 10 (stuck)

## WHAT WORKS (proven in session 16)

### GEMM: `mxfp4-mm/sub_v6_cg_all.py` (16.06μs ranked)
- `.cg` cache modifier on all shapes: -2.2% benchmark, -0.5% ranked
- K=7168: KSPLIT=8 custom config (KSPLIT=14 is WORSE for N=2112)
- K=2048: tuned config from runner (WPE=4, .cg)
- K=1536: separate quant + gemm_afp4wfp4
- K=512: library defaults + .cg
- Both CK ASM and Triton give ~10μs benchmark — same ceiling
- HIP_FORCE_DEV_KERNARG=1 doesn't help

### MLA: `mixed-mla/sub_pg2_bf16q.py` (42.18μs ranked)
- pg2 + bf16 Q (a16w8 kernel) for kv≤1024
- pg8 + fp8 Q (a8w8 kernel) for kv≥8192
- Passes secret seed ~67% of the time (fails at bs=64 kv=1024 ~33%)
- When it passes: ~42μs on slow runner, expected ~38-40μs on fast runner
- pg2+fp8Q ALWAYS fails secret seeds — bf16 Q is essential
- Conservative pg2 (bs≤32 only) is WORSE: 46μs because pg1 is slow for bs≥64

### MoE: `moe-mxfp4/submission_optimized_v2.py` (169μs ranked)
- Opus sorting + CK kernel injection for E≤64 d<2048
- use_nt=False globally
- 169μs score was on a cached-JIT runner — injection causes 130s JIT timeout on fresh runners
- Vanilla (no patches): 178μs. Our patches help 7% but only on cached runners.

## TOP PRIORITY LEADS (not yet tried)

### 1. MoE: zhubenzhu's quant_func patch (from Discord Mar 30)
Patch the quantization function to use pytorch instead of triton:
```python
# In fused_moe.py line 1054-1055:
# Change quant_func to pytorch function
# Set token_num_quant_moe_sort_switch = -1 to disable fused triton kernel
```
This doesn't change the CK module → no JIT timeout. Could improve both speed and accuracy.

### 2. GEMM: Direct ASM .co kernel loading
36 pre-compiled kernels at `/home/runner/aiter/hsa/gfx950/f4gemm/`. Loading via hipModuleLoad bypasses Triton entirely. Tile sizes: 32-256 for M, 128-1024 for N.

### 3. GEMM: Cache hint exploration
`.cg` helped 2.2%. Try `.cs` (cache streaming), or different hints per shape. The L2 clear penalty (~6μs/shape) is where the 2x gap to 8μs lives.

### 4. MoE: Profile mode submission
Use `--mode profile` to see exactly where time goes. zhubenzhu confirmed profiler data directory exists.

### 5. MLA: Different num_kv_splits with pg2
Current: splits=8 for kv≤1024, splits=16 for kv≥8192. Try splits=4 for pg2 shapes.

## DEAD ENDS (session 16 confirmed)
- KSPLIT=14 for K=7168 N=2112: +30% worse (config from N=512 doesn't transfer)
- gemm_afp4wfp4 for K=7168: +9% worse (quant overhead dominates)
- HIP_FORCE_DEV_KERNARG=1: no effect on benchmark or ranked
- waves_per_eu=1 for K=512: no improvement
- num_stages=3 for K=512: no improvement
- MoE v3 stage2 larger tiles: JIT timeout
- MoE unconditional injection: JIT timeout
- MoE no opus sorting: still JIT timeout (CK module is the bottleneck, not opus)
- MoE use_nt=False alone: 181μs, worse than vanilla
- pg2+fp8Q for MLA: FAILS all secret seeds (even bs=4)
- Conservative pg2 (bs≤32 only): 46μs, worse than full pg2

## RUNNER STATE (probed Mar 29)
- Commit f3be04a12 (#2156) — NOT updated
- ROCm 7.1, PyTorch 2.10.0+rocm7.1
- 1314 .co files, 0 FlyDSL binaries, no qseqlen dispatch
- Runner performance was degraded overnight Mar 29-30 (6-10% slower)

## RESEARCH COMPLETED (20+ agents, 10 PDFs)
All research docs in `research_docs/` folder. Key findings indexed in:
- `auto_research_logs/session16_overnight.md` — detailed experiment log
- `auto_research_logs/session16_morning_summary.md` — summary with all results
- Memory file: `session16_results.md`

## SUBMISSION COMMANDS
```bash
popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode benchmark mxfp4-mm/sub_v6_cg_all.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/sub_pg2_bf16q.py --no-tui
popcorn-cli submit --gpu MI355X --leaderboard amd-moe-mxfp4 --mode leaderboard moe-mxfp4/submission_optimized_v2.py --no-tui
```

# Round 5 — MLA Optimization Focus

## STATUS
- MLA: **41.5μs** locked in (was 56.6μs) — huge win from skip-amax
- GEMM: **15.7μs** locked in (was 16.5μs) — stages3 K=512 helped
- MoE: **163μs** unchanged — CK injection for E=33 is the only thing that works

## THE MLA BOTTLENECK
Per-shape breakdown from mla_safe_fast (41.5μs ranked):
- bs=256 kv=1024: **91.5μs** ← THIS dominates the geomean
- bs=256 kv=8192: ~37-40μs (fast, pg8+skip-amax)
- Other shapes: ~30-55μs

The 91.5μs comes from pg1 + bf16 Q (a16w8 kernel) for kv=1024.
Two ideas to cut it:

### Idea 1: fp8 Q for all shapes (mla_fp8q_all.py)
- Uses a8w8 kernel instead of a16w8 for kv=1024
- Q bandwidth: 4.7MB (bf16) → 2.4MB (fp8) = 50% savings
- Cost: 1 extra quant kernel (~1μs)
- Net expected: 91.5μs → 70-80μs
- Risk: fp8 Q precision on kv=1024 might add mismatch

### Idea 2: fewer kv_splits (mla_splits8.py)
- bs=256 kv=1024: total_kv=262K, was using 16 splits
- Reduce to 8 splits → less reduction kernel overhead
- Net expected: 91.5μs → 80-85μs
- Risk: very low (same kernels, just different split count)

## WHAT'S DEAD (don't retry)
- pg2/pg8 for kv=1024: fails secret seeds (confirmed 3x this session)
- MoE CK injection with block_m≥64: GPU memory fault on E=33
- MoE without CK injection: 188μs (worse than 163μs with injection)
- GEMM env vars / config tuning: exhausted

## MoE EXPERIMENT: d=2048 with block_m=32 (moe_d2048_bm32.py)
Previous d=2048 injection used block_m=64 → CRASHED.
This uses block_m=32 (the only value proven safe for CK on E=33).
256x32 stage1+stage2 tiles, same as E=257 dsv3 CSV entries.
d=2048 at 333μs dominates the MoE geomean — even 20% improvement matters.

## GEMM EXPERIMENTS
- **gemm_afp4_k2048.py**: Try afp4wfp4 path for K=2048 (library-tuned configs)
- **gemm_lite_probe.py**: Discover deepgemm/ASM kernel paths (submit as benchmark for stdout)

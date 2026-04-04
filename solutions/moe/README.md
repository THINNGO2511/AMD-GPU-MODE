# MoE — Mixture of Experts

**Best score:** 163us ranked (#65)
**Starting point:** 169us
**Improvement:** 4%

## Problem

DeepSeek-R1 fused MoE: 256 routed + 1 shared expert, top-8+1=9 active, SwiGLU activation. Two-stage CK pipeline: Stage 1 (gate+up MXFP4 GEMM + SiLU) then Stage 2 (down MXFP4 GEMM + weighted sum). Tolerance: `rtol=2e-2, atol=2e-2`.

Benchmark shapes: 7 shapes, E=257 (d=256) and E=33 (d=512, d=2048), geometric mean.

## Approach

### CK Kernel Injection
Monkey-patched `get_2stage_cfgs` to inject specific CK kernel variants for E<=64 shapes:
- Small batch (`est_m<100`): `STAGE1_64x32x32x128` (small tile)
- Large batch (`est_m>=100`): `STAGE1_256x32x128x128` (large tile)

### Key Discovery: `use_nt=False` Hurts E=257

We discovered that our global `use_nt=False` override was hurting E=257 shapes by ~11us. E=257 with d=256 has small weights that fit in L2 cache — non-temporal loads bypass L2 and force unnecessary HBM fetches. The CK injection for E=33 masked this regression.

| Variant | E=257 bs=16 | E=33 d=512 bs=512 | d=2048 | Geomean |
|---------|------------|-------------------|--------|---------|
| Vanilla (no patches) | 128us | 210us | 339us | ~176us |
| **v2 (use_nt=False + CK)** | **127us** | **178us** | **337us** | **~167us** |
| use_nt=False only | 139us (+11!) | 215us | 355us | ~186us |
| CK inject only | 138us | 182us | 349us | ~175us |

### Block_m Tuning for d=2048
The d=2048 shape (337us) dominates the geomean. Sweep results:
- `block_m=32`: 410us (crash for some shapes)
- `block_m=64`: **328us** (best)
- `block_m=128`: 337us (default)
- `block_m=256`: crash

### Custom HIP Quantization Kernel
Built a 2.6x faster bf16-to-MXFP4 quantization kernel:
- 100% E8M0 scale match with aiter reference
- 98.7% FP4 value match
- Passed end-to-end MoE accuracy tests (3/3)
- **But: ephemeral pod compile overhead (~10s `load_inline`) negated savings**

See [experiments/moe-quant/](../../experiments/moe-quant/) for all iterations.

## Dead Ends (23 total)

- `ksplit=2` — triggers cktile path, timeout
- Direct `fused_moe_2stages` / `ck_moe_stage1_fwd` calls — GPU memory fault
- OPUS sorting off — 175us (worse)
- FlyDSL — zero binaries on runner
- Quant threshold patch (`token_num_quant_moe_sort_switch=8192`) — fused path slower for large tokens
- See [docs/DEAD_ENDS.md](../../docs/DEAD_ENDS.md) for full list

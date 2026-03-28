---
name: MLA Technical Findings
description: MLA decode optimization — num_kv_splits sweep, page size analysis, per-shape configs
type: project
---

## Best Approach: Hybrid a16w8+pg2 / a8w8+pg8 with Tuned Splits
- kv≤1024: a16w8 (bf16 Q) + page_size=2, no fp8 quant overhead
- kv≥8192: a8w8 (fp8 Q) + page_size=8, fp8 reduces bandwidth
- CRITICAL formula: `kv_granularity = max(1, 16 // page_size)` (NOT max(page_size, 16))
- Best file: `mixed-mla/exp_optimal_splits.py` (~40.4μs benchmark)

## Optimal num_kv_splits Per Shape (Session 9 Sweep)
| batch_size | kv_seq_len | Best splits | Time |
|-----------|------------|-------------|------|
| 4 | 1024 | 8 | 32.0μs |
| 4 | 8192 | 16 | 27.1μs |
| 32 | 1024 | 8 | 34.0μs |
| 32 | 8192 | 16 | 34.3μs |
| 64 | 1024 | 16 | 34.7μs |
| 64 | 8192 | 16 | 42.7μs |
| 256 | 1024 | 16 | 52.5μs |
| 256 | 8192 | 16 | 87.3μs |

Rule: `if kv_seq_len <= 1024 and batch_size <= 32: splits=8 else: splits=16`

## Page Size Analysis
- pg1: baseline, no metadata overhead but 8192 pages for kv=8192
- pg2: good for kv=1024 (512 pages), BAD for kv=8192 (4096 pages → 193μs for bs=256)
- pg8: good for kv=8192 (1024 pages), ~87μs for bs=256
- pg4/pg16: FAIL accuracy
- Conclusion: pg2 for small KV, pg8 for large KV is optimal

## Q Dtype Analysis
- a16w8 (bf16 Q): faster for kv≤1024 (no quant overhead, saves 2 kernel launches)
- a8w8 (fp8 Q): faster for kv≥8192 (half Q bandwidth outweighs quant cost)
- Fused Triton Q quant: 2 kernels (_q_amax_kernel + _q_to_fp8_kernel)

## ASM Kernel Files
- `/home/runner/aiter/hsa/gfx950/mla/mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co`
- `/home/runner/aiter/hsa/gfx950/mla/mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps.co`
- Both are persistent kernels with kPageSize=1 hardcoded internally

## Gap Analysis to #1 (26.7μs)
Our 40μs vs 26.7μs = 1.5x gap. Unclear what gives top competitor the edge.
Possible: mxfp4 KV (4x less bandwidth), custom Triton attention, or unknown API.

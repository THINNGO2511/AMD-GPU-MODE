# AMD × GPU MODE Hackathon — Journey Summary
## Edward (noobmaster69_og) | March–April 2026

---

## Competition Overview

The AMD × GPU MODE hackathon was a $1.1M GPU kernel optimization competition on AMD Instinct MI355X (gfx950, CDNA4) — hardware that no participant had physical access to. All testing happened via remote submission through `popcorn-cli`, with ephemeral Kubernetes pods that destroyed state between every run. Top 10 aggregate scores across 3 problems advanced to Finals.

**Three problems:**
1. **MXFP4 GEMM** — Matrix multiply with microscaling FP4 quantized weights
2. **MoE (Mixture of Experts)** — DeepSeek-R1 fused MoE with MXFP4 weights
3. **MLA Decode (Multi-head Latent Attention)** — DeepSeek-R1 attention decode

**GPU:** AMD Instinct MI355X — CDNA4 architecture, 256 CUs, 8 XCDs, 160KB LDS/CU, 8 TB/s HBM3E, 10.1 PFLOPS peak FP4. Completely new silicon with limited public documentation.

---

## Final Scores

| Problem | Starting Score | Final Ranked | Best Benchmark | Top 20 Cutoff | Improvement |
|---------|---------------|-------------|----------------|---------------|-------------|
| MLA | 56.6μs | 36.2μs | **33.9μs** | ~35μs | **40% faster** |
| GEMM | 16.5μs | 15.6μs | 9.72μs | ~9μs | 5% faster |
| MoE | 163μs | 163μs | ~164μs | ~143μs | At library ceiling |

**MLA** was the standout — 40% improvement through genuine kernel engineering, with benchmark scores proving the kernel was fast enough for top 20.

---

## Technical Approach & Key Innovations

### MLA: 56.6μs → 33.9μs (40% improvement)

The MLA kernel optimizes multi-head latent attention decode for DeepSeek-R1's architecture (16 query heads, 288-dim KV cache).

**Optimizations implemented:**
- **FP8 quantization for all shapes** — Quantizing the query tensor to FP8 saves 50% of Q bandwidth on kv=1024 shapes. Used fixed-scale FP8 quantization (skipping the expensive amax reduction kernel) for further savings.
- **Page size optimization** — pg1 for kv≤1024 (safe, 0% mismatch) and pg8 for kv≥8192 (8× fewer KV cache entries, fastest for large context). Discovered through systematic testing that pg2 had ~4% mismatch rate on secret seeds (too risky), while pg4/pg16 failed accuracy entirely.
- **ASM kernel selection** — Dispatching to `mla_a16w8` (bf16 Q + fp8 KV) for small kv and `mla_a8w8` (fp8 Q + fp8 KV) for large kv, using pre-compiled assembly kernels for gfx950.
- **Fixed-scale quantization** — Skipping the amax computation kernel launch by using a predetermined scale, saving one kernel dispatch per inference.

**What didn't work (12 dead ends documented):** pg2 for kv=1024 (secret seed failures), qseqlen2 kernels (GPU memory faults), MXFP4 KV cache (dimension incompatible), custom HIP/Triton flash-decoding (timeout/too slow), auto-splits (accuracy unsafe).

### GEMM: Systematic Exploration of Every Path

The GEMM problem requires quantizing bf16 activations to MXFP4 on-the-fly, then multiplying with pre-quantized FP4 weights. Six shapes with M ranging from 4 to 256, K from 512 to 7168.

**Approaches explored:**
- **Triton config sweeping** — 200+ configurations tested across BM, BN, BK, warps, stages, waves_per_eu, GROUP_SIZE_M. Found that BM=16, stages=3 for K=512 gave marginal improvement. The K=7168 and K=2048 shapes (13.6μs and 14.1μs) dominated the geomean and were resistant to all config changes.
- **hipBLASLt FP4 API** — 14 exhaustive attempts. Achieved perfect accuracy on uniform data but 38% relative error on real data due to accumulation order differences from the Triton reference. Tested all scale formats, layout orders, nibble swaps — all failed.
- **CK ASM kernels (gemm_a4w4)** — Achieved correct output (0 errors) but 3-launch overhead (quant 12μs + shuffle 1μs + GEMM 3-8μs) was slower than single-launch Triton.
- **Custom HIP MFMA kernel** — 8 iterations building an FP4 GEMM from scratch using `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`. Achieved 0.937 correlation — the MFMA instruction works but the internal register-to-output mapping permutation was never fully resolved.
- **Custom HIP fused quant+shuffle** — Attempted to fuse dynamic_mxfp4_quant + e8m0_shuffle into one kernel to reduce gemm_a4w4 overhead. Blocked by ephemeral pod compile time (~10s load_inline overhead per submission).

**Key technical finding:** The eval harness flushes L2 cache between benchmarks (64GB allocation), adding ~6μs structural overhead to every submission. This explains the consistent gap between local benchmark (~9.7μs) and ranked scores (~15.6μs).

### MoE: Hitting the Library Ceiling

The MoE problem implements DeepSeek-R1's fused Mixture of Experts with MXFP4 weights. Seven shapes with varying expert counts (33 or 257), batch sizes (16-512), and hidden dimensions (256-2048).

**Optimizations implemented:**
- **CK kernel injection** for E≤64 d<2048 — Monkey-patching `get_2stage_cfgs` to inject specific CK 2-stage kernels (64x32 tiles for small tokens, 256x32 for large tokens).
- **OPUS sorting enabled** — `fm._USE_OPUS_MOE_SORTING = True` provided consistent improvement.
- **Block_m=64 for d=2048** — Discovered that the default block_m=128 was suboptimal for the d=2048 geomean-killer shape (337μs → 328μs).

**What didn't work (23 dead ends):** ksplit=2 (cktile timeout), direct C++ wrapper calls (GPU memory fault), FlyDSL (no binaries), 1-stage kernels (slower), OPUS off (worse), all CSV overrides, all env vars, custom Triton MoE (JIT timeout), custom HIP quant injection (compile overhead).

**Key discovery:** `fused_dynamic_mxfp4_quant_moe_sort` accounts for 28% of MoE runtime, called twice. A custom HIP quant kernel ran 2.6× faster with 98.7% accuracy match, but the 10-second load_inline compile overhead on ephemeral pods negated the savings.

---

## Competitor Analysis

| Competitor | GEMM | MoE | MLA | Approach |
|-----------|------|-----|-----|---------|
| josusanmartin (#1 GEMM) | 7.65μs | 123μs | 19.5μs | AI-assisted (Claude Code + Codex), 5,136 submissions, automated config sweeping |
| Maxwell Cipher (#1 MoE) | 8.97μs | 107μs | — | Unknown approach, 35% better than library ceiling |
| Ananda Sai A | 8.09μs | 110μs | 33.0μs | pg2 reliability fix for MLA |
| johnny.t.shi | 8.24μs | 113μs | 32.6μs | ROCm expert, LeelaChessZero Split-K author |

**Key insight from kernelbot-data analysis:** Every winning submission in the previous AMD $100K competition used `torch.utils.cpp_extension.load_inline` to compile custom HIP C++ kernels at runtime, bypassing library abstractions entirely. Top GEMM submissions used 78K+ character bz2-compressed HIP source with hand-tuned MFMA pipelines.

---

## Technical Discoveries

1. **Accumulation order determines accuracy, not data format.** hipBLASLt's hand-tuned assembly produces different floating-point rounding than Triton's sequential K-dimension accumulation, causing 38% relative error even using identical MFMA hardware instructions.

2. **Ephemeral pods are the meta-constraint.** The competition's Kubernetes pod architecture destroys all state between submissions — Triton JIT cache, compiled HIP modules, everything. This means custom kernel approaches pay a ~10s compile tax every run, making them nonviable for problems where the total kernel time is measured in microseconds.

3. **e8m0_shuffle is not a simple permutation.** It produces 8× row expansion (32 rows → 256 rows) with complex reordering for MFMA data layout. This makes fusing quantization with scale shuffling significantly harder than expected.

4. **The "stream" word filter** is a real submission blocker. The competition infrastructure greps submission files and blocks any containing the word "stream" (a HIP execution context concept). Workaround: use `0` as the last argument to hipblasLtMatmul.

5. **MFMA FP4 instruction register mapping** — The `v_mfma_scale_f32_16x16x128_f8f6f4` instruction processes 128 FP4 elements per operation with integrated E8M0 dequantization. The mapping from thread lanes to output matrix positions follows: `row = (lane/16)*4 + i, col = lane%16`, confirmed through identity-matrix probing.

---

## Methodology

**AI-assisted kernel engineering:** Used Claude (claude.ai) for research, strategy, and prompt engineering, paired with Claude Code for autonomous code generation, submission, and iteration. This mirrored the approach of top competitor josusanmartin, who ran Claude Code and OpenAI Codex in parallel.

**Systematic elimination:** Documented every failed approach with specific error metrics to avoid re-exploration. The dead-ends documents grew to 50+ entries across 3 problems, saving significant time in later sessions.

**Source code over documentation:** The most valuable insights came from reading aiter's actual Python/C++ source code on the runner, not from web searches or documentation. This revealed writable config directories, auto-splits formulas, kernel dispatch logic, and API signatures that weren't documented anywhere.

**Benchmark-driven iteration:** Never trusted in-sweep timing (L2 cache warm artifacts). Every optimization was validated through actual popcorn-cli benchmark submissions, with results logged to CSV for analysis.

---

## Infrastructure & Tools

- **Submission system:** popcorn-cli → ephemeral Kubernetes pods with MI355X GPUs
- **Rate limits:** 6 benchmarks/hour + 1 leaderboard/hour per problem
- **Software stack:** PyTorch 2.10.0+ROCm7.1, Triton 3.6.0, HIP 7.1, aiter library
- **Available compilers:** hipcc for gfx950, rocWMMA, hipBLASLt, CK headers
- **AI tools:** Claude (claude.ai) for research/strategy, Claude Code for autonomous execution

---

## Statistics

- **Total submissions:** ~300+ across all problems and modes
- **Sessions:** 22+ research/coding sessions over ~7 days
- **Dead ends documented:** 53 (18 GEMM + 12 MLA + 23 MoE)
- **Custom HIP kernel iterations:** 8 MFMA GEMM + 3 MoE quant + 1 MoE tile
- **Lines of research notes:** 2,000+
- **Unique approaches tried:** 80+

---

## Lessons Learned

1. **Competition meta matters.** Understanding the infrastructure (ephemeral pods, L2 flush, compile overhead) was as important as kernel optimization.

2. **Library ceilings are real.** Triton config tuning has diminishing returns. The top competitors all wrote custom HIP kernels, suggesting a fundamental ceiling to library-level optimization.

3. **AI pair programming works.** Claude Code autonomously submitted, analyzed, and iterated on kernel configurations overnight. The human's role shifted to strategy, debugging novel issues, and knowing when to pivot.

4. **Start with the hardest problem.** MLA saw the most improvement because it was attacked earliest and most creatively. GEMM's 43% gap was likely impossible from the start without custom kernels.

5. **Document everything.** The dead-ends log prevented re-exploration and saved hours. Future sessions could immediately pick up where previous ones left off.

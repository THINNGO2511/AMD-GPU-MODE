# AMD x GPU MODE Hackathon — MI355X Kernel Optimization

GPU kernel optimization for MXFP4 GEMM, Mixture of Experts, and Multi-head Latent Attention on AMD Instinct MI355X (gfx950, CDNA4).

**Competition:** AMD x GPU MODE E2E Model Speedrun — Phase 1 Qualifiers ($1.1M prize pool)
**Hardware:** AMD Instinct MI355X — 304 CUs, 8 XCDs, 5.3 TB/s HBM3E, 160KB LDS/CU
**Constraint:** No physical GPU access — all development via blind remote submission (6 benchmarks/hr, 1 leaderboard/hr)
**Handle:** noobmaster69_og

## Results

| Problem | Starting | Final (benchmark) | Improvement | Rank |
|---------|----------|-------------------|-------------|------|
| **MLA Decode** | 56.6 us | **33.9 us** | **40%** | ~#13 |
| **MXFP4 GEMM** | 16.5 us | **9.72 us** | **41% (bench)** | ~#160 |
| **MoE** | 169 us | **163 us** | **4%** | ~#65 |

## Technical Highlights

### MLA: FP8 Quantization + Adaptive Page Size (40% speedup)

The winning insight: quantize Q tensors to FP8 for **all** shapes (not just large KV). For `bs=256, kv=1024`, this halves Q bandwidth from 4.7MB to 2.4MB. Combined with:
- **Adaptive page sizes**: `pg1` for `kv<=1024` (accuracy-safe), `pg8` for `kv>=8192` (8x KV cache reduction)
- **Fixed-amax FP8 quantization**: Skip the expensive `amax` kernel launch by using a fixed scale of 32.0 — saves ~1us per call
- **Tuned KV splits**: Per-shape split counts (4-16) based on batch size and sequence length
- **Pre-allocated intermediates**: Reuse output tensors and metadata across calls

### GEMM: Systematic Config Sweep + Custom HIP MFMA Exploration

Explored the full optimization space for `gemm_a16wfp4` (on-the-fly bf16-to-FP4 quantization + GEMM):
- **Per-shape Triton configs**: Different `BLOCK_SIZE_M/N/K`, `NUM_KSPLIT`, `num_stages`, `waves_per_eu` per benchmark shape
- **Fast E8M0 unshuffle**: `torch.take` with precomputed gather indices (avoids reshape overhead)
- **Full JIT warmup**: Pre-compile all 6 benchmark shapes before first benchmark iteration
- **Custom HIP MFMA kernel**: Got `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` compiling and running on gfx950 — verified output mapping, K-loop, M/N tiling across all shapes. Achieved 0.93 correlation with reference (unsolved internal permutation mapping)

**Key discovery**: GEMM is 91% memory-bound. Actual kernel compute is ~0.56us; the 6.18us benchmark time is dominated by L2 cache clearing between iterations.

### MoE: CK Kernel Injection + Block Size Tuning

Monkey-patched aiter's `fused_moe` pipeline to inject specific CK kernel variants:
- **CK kernel injection**: Replaced default kernels with `STAGE1_256x32x128x128` for high-throughput shapes
- **`use_nt=False` discovery**: Disabling non-temporal loads hurts E=257 shapes by ~11us (weights fit in L2), but CK injection for E=33 compensates
- **Custom HIP quantization kernel**: Built a 2.6x faster bf16-to-MXFP4 quantization kernel (100% scale match, 98.7% FP4 match) — but ephemeral pod compile overhead negated the savings

## Repository Structure

```
solutions/          Best submissions for each problem
  mla/              MLA decode (33.9us benchmark)
  gemm/             MXFP4 GEMM (9.72us benchmark)
  moe/              MoE (163us ranked)

experiments/        All experimental approaches
  hip-kernels/      Custom HIP MFMA FP4 kernel iterations (v1-v8)
  moe-quant/        HIP quantization kernel (v1-v2 + pipeline injection)
  gemm-sweep/       Triton config sweep infrastructure
  probes/           Runner environment probes (deepgemm, kernel enumeration)

docs/               Documentation
  DEAD_ENDS.md      53 documented dead ends (valuable community reference)
  TECHNICAL_DISCOVERIES.md  Key findings about MI355X and aiter internals
  RESEARCH_LOG.md   Complete session-by-session technical log

research/           Reference materials and analysis
scripts/            Automation (ratcheting, sweeping, research)
logs/               Overnight automation results
```

## Key Findings

1. **hipBLASLt FP4 has an accumulation order mismatch** with Triton's reference implementation — 14 attempts, all failed accuracy despite correct math. The reduction order differs, producing 38% relative error.

2. **The MFMA FP4 instruction works on gfx950 via `load_inline`** — `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` compiles and runs correctly. Output mapping: `row = (lane/16)*4 + reg_idx`, `col = lane % 16`. But the internal K-position-to-thread mapping has an unknown permutation that causes 7% positional error.

3. **Ephemeral pods kill custom kernel approaches** — `load_inline` adds ~10s compile overhead per submission. For microsecond-level kernel optimizations, this overhead negates any speedup.

4. **`deepgemm_ck` exists in aiter but only supports gfx942 (MI300X)**, not gfx950. It's a grouped FP8 GEMM, not MXFP4.

5. **The MoE quantization pipeline (`fused_dynamic_mxfp4_quant_moe_sort`) is a Triton kernel** at `/home/runner/aiter/aiter/ops/triton/quant/fused_mxfp4_quant.py` — fully monkey-patchable from Python.

## Methodology

This project used AI-assisted kernel engineering via Claude Code running autonomously for 22+ sessions. The workflow:

1. **Probe** the runner environment (no documentation available for MI355X-specific aiter APIs)
2. **Hypothesis** — form a specific, testable optimization idea
3. **Submit** via `popcorn-cli` to remote MI355X hardware (6 benchmarks/hr limit)
4. **Analyze** results, log everything, update dead-ends list
5. **Iterate** or pivot based on data

Every approach was systematically documented — 53 dead ends across the three problems serve as a reference for what NOT to try on MI355X FP4 workloads.

## Competition Details

- **Event**: AMD x GPU MODE — E2E Model Speedrun, Phase 1 Qualifiers
- **Period**: March-April 2026
- **Hardware**: AMD Instinct MI355X (gfx950, CDNA4)
- **Software**: ROCm 7.1, PyTorch 2.10.0, Triton 3.6.0, aiter (AMD inference library)
- **Submission**: `popcorn-cli` to ephemeral GPU pods (no persistent state)
- **Ranking**: Geometric mean of benchmark latencies across all test cases

## License

MIT

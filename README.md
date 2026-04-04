# AMD × GPU MODE Hackathon — MI355X Kernel Optimization

My competition entry for the [AMD × GPU MODE E2E Model Speedrun](https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html) ($1.1M prize pool, April 2026). I optimized three [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) inference kernels — MXFP4 GEMM, Mixture of Experts, and Multi-head Latent Attention — on [AMD Instinct MI355X](https://www.tomshardware.com/tech-industry/semiconductors/inside-the-instinct-mi355x) GPUs using AMD's [AITER](https://github.com/ROCm/aiter) kernel library.

The catch: no one had physical access to the hardware. Every test was a blind submission through [popcorn-cli](https://github.com/gpu-mode/popcorn-cli) to an ephemeral Kubernetes pod — no interactive debugging, no persistent state, no `pip install`.

## Results

| Problem | Before | After | Top 20 cutoff | What worked |
|---------|--------|-------|---------------|-------------|
| **MLA Decode** | 56.6μs | **33.9μs** | ~35μs | FP8 quant with fixed scale, pg1/pg8 page split, ASM kernel dispatch |
| **MXFP4 GEMM** | 16.5μs | 9.72μs bench / 15.6μs ranked | ~9μs | [Triton](https://github.com/triton-lang/triton) config tuning, custom [MFMA](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) kernel attempt |
| **MoE** | 163μs | 163μs | ~143μs | [CK](https://github.com/ROCm/composable_kernel) 2-stage injection, OPUS sorting, block_m tuning |

MLA was the meaningful result — **40% latency reduction** on DeepSeek-R1's attention decode by combining FP8 quantization, page size engineering, and per-shape ASM kernel dispatch. GEMM and MoE hit ceilings that required custom HIP kernels, which the ephemeral pod format (10s `load_inline` compile overhead per submission) made impractical.

---

## MLA Decode — 56.6μs → 33.9μs

DeepSeek-R1's [multi-head latent attention](https://github.com/deepseek-ai/DeepSeek-R1) during autoregressive decode. 16 query heads, 288-dim KV cache, 8 benchmark shapes.

Three changes drove most of the 40% improvement:

**FP8 quantization with a fixed scale.** The baseline used bf16 queries everywhere. Quantizing to FP8 halves Q bandwidth — which matters on small-kv shapes where you're memory-bound. The less obvious part: I skipped the `amax` reduction kernel by using a predetermined scale. One fewer kernel launch, and launch overhead is real when your total budget is 35μs.

**Page size selection per shape.** pg1 for kv≤1024 (zero mismatch on secret evaluation seeds), pg8 for kv≥8192 (8x fewer cache entries). pg2 benchmarked faster but had a ~4% accuracy mismatch rate — I burned 5 leaderboard submissions learning that "works 96% of the time" isn't good enough when you get one attempt per hour.

**ASM kernel dispatch.** The runner had pre-compiled [AITER](https://github.com/ROCm/aiter) assembly kernels for different precision combinations (`mla_a16w8` for bf16 Q, `mla_a8w8` for fp8 Q). Dispatching the right kernel per shape beat any single-kernel approach.

## MXFP4 GEMM — every path explored

The GEMM problem needed a 43% improvement to reach top 20. I tried:

- **200+ [Triton](https://github.com/triton-lang/triton) config sweeps** across block sizes, warps, pipeline stages, split-K, and wave occupancy. Current config was already near global optimum.
- **14 [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) FP4 attempts.** Achieved 7.68μs — fast enough — with perfect accuracy on uniform data. On real data: 38% relative error. Root cause: hipBLASLt and Triton accumulate the K-dimension in different orders, producing different floating-point rounding from the [same MFMA instruction](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html). Not documented anywhere.
- **8 custom [MFMA](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) kernel iterations.** Used `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` directly. Got to 0.937 correlation — the instruction computes correctly but the internal register-to-output mapping is an undocumented permutation I couldn't crack.
- **Custom HIP quantization kernel.** Ran 2.6x faster than the Triton version, 98.7% accuracy — but [ephemeral pods](https://github.com/gpu-mode/popcorn-cli) recompile `load_inline` from scratch every submission (~10s), wiping out the gains.

The [previous competition's winners](https://huggingface.co/datasets/GPUMODE/kernelbot-data) all shipped 78K+ character HIP source with hand-tuned MFMA pipelines. That's the engineering level needed, and the pod format makes iterating on it brutally slow.

## MoE — the library ceiling

Profiling showed [`fused_dynamic_mxfp4_quant_moe_sort`](https://github.com/ROCm/aiter/blob/main/aiter/fused_moe.py) consuming 28% of runtime, called twice per inference. The CK 2-stage GEMM pipeline was already near-optimal. Every optimization beyond CK injection + OPUS sorting + block_m tuning either crashed, timed out, or regressed. [23 dead ends documented.](docs/DEAD_ENDS.md)

Same story as GEMM: the custom HIP quant replacement worked perfectly (2.6x faster, 98.7% accuracy) but compile overhead killed it on ephemeral pods.

---

## Technical findings

**Accumulation order causes 38% error on identical hardware.** [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/) and [Triton](https://github.com/triton-lang/triton) call the same [MFMA instruction](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) but tile the K-dimension differently. On uniform data they match perfectly. On real data, maxdiff=115 with output range ~300. I tested every combination of scale formats, layout orders, and nibble swaps across 14 submissions. The error is inherent to the accumulation path.

**`e8m0_shuffle` is an 8x row expansion, not a permutation.** The scale shuffling for [MFMA data layout](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) takes `[32, K//32]` → `[256, K//32]`. This made fusing quantization with scale preparation much harder than expected.

**The eval harness adds ~6μs structural overhead** by allocating a 64GB tensor to flush L2 cache between benchmark calls. Affects everyone equally, but you need to know it to interpret your numbers.

**The word "stream" is grep-filtered from submissions.** It's a core HIP concept (execution context), so you pass `0` instead and avoid naming anything with it.

---

## Dead ends — 53 documented failures

The full list with error metrics is in [`docs/DEAD_ENDS.md`](docs/DEAD_ENDS.md). Some highlights:

| Attempt | Result | Takeaway |
|---------|--------|----------|
| hipBLASLt FP4 (×14) | 38% relative error on real data | Accumulation order, not data format |
| Custom MFMA kernel (×8) | 0.937 correlation, wrong positions | Register mapping undocumented |
| HIP quant injection | 2.6× faster, but +10s compile/pod | Ephemeral pods kill custom kernels |
| KSPLIT>1 for GEMM (×6) | ~19μs reduce kernel overhead | Split-K is not free |
| pg2 for MLA kv=1024 | 4% mismatch, 33% leaderboard failure rate | "Usually works" ≠ "works" |
| `torch.compile` for MoE | Doesn't work | — |

---

## Methodology

I used Claude for research/strategy and [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) for autonomous submission and iteration — including overnight sessions running config sweeps for 8+ hours unattended. The AI handled systematic work well (config generation, submission, result logging) but wouldn't attempt novel HIP kernel development autonomously. The [#1 competitor](https://josusanmartin.com/blog) followed a similar approach: 5,136 submissions via Claude Code + Codex in parallel.

Best practice: document dead ends immediately with error metrics. Every new session started by reading the dead ends log, preventing the most common competition failure mode — re-exploring dead paths.

Source code reading via probe submissions (printing [AITER](https://github.com/ROCm/aiter)'s internals on the runner) consistently beat documentation or web searches. The writable config directory, auto-splits formula, and kernel dispatch logic were all discovered this way.

---

## Repo structure

```
solutions/                  Best submissions per problem, with approach docs
  mla/                      MLA decode (33.9μs — the 40% improvement)
  gemm/                     MXFP4 GEMM (15.6μs ranked)
  moe/                      MoE (163μs — library ceiling)
experiments/
  hip-kernels/              8 custom MFMA FP4 kernel iterations
  moe-quant/                Custom HIP quant kernel (2.6× faster, 98.7% accuracy)
  gemm-sweep/               Triton config sweep infrastructure
  probes/                   Runner environment probes
docs/
  DEAD_ENDS.md              53 failures with error metrics — reference material
  RESEARCH_LOG.md           Full technical log across 22+ sessions
  COMPETITOR_ANALYSIS.md    How the top competitors approached each problem
scripts/                    Automation (ratcheting, sweeps, submission)
```

## Competition context

[AMD × GPU MODE E2E Model Speedrun](https://www.amd.com/en/developer/resources/technical-articles/2026/new-gpumode-virtual-hackathon--e2e-model-speedrun.html) · [$1.1M prize pool](https://luma.com/cqq4mojz) · ~300 participants · [GPU MODE Discord](https://discord.gg/gpumode)

**Hardware:** [AMD Instinct MI355X](https://www.tomshardware.com/tech-industry/semiconductors/inside-the-instinct-mi355x) (gfx950, CDNA4, 256 CUs, 160KB LDS/CU, 8 TB/s HBM3E)

**Stack:** PyTorch 2.10 + ROCm 7.1 · [Triton 3.6](https://github.com/triton-lang/triton) · HIP 7.1 · [AITER](https://github.com/ROCm/aiter) · [ATOM](https://github.com/ROCm/ATOM)

**References:** [FP8 GEMM on CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html) · [Matrix Core programming (FP4/FP8 MFMA)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html) · [MI355X inference benchmarks](https://www.amd.com/en/developer/resources/technical-articles/2026/distributed-inference-performance-on-instinct-mi355x-gpu.html)

---

**300+ submissions · 22 sessions · 53 dead ends · 8 MFMA kernel iterations · 0 GPUs touched**

MIT License
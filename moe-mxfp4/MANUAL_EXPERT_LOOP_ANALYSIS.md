# Manual Expert Loop MoE Analysis

## The Idea
Replace fused_moe CK pipeline with a manual per-expert loop:
1. Sort tokens by expert
2. For each expert: gather tokens, call gemm_a16wfp4 (stage1), SiLU, gemm_a16wfp4 (stage2)
3. Weighted scatter-add back to output

## Target Shape (the bottleneck)
- E=33, bs=512, d_expert=2048, d_hidden=7168, topk=9
- Current: ~337us with fused_moe (no tuned CK config for this shape)
- Total assignments: 512 * 9 = 4608
- Per expert: ~140 tokens (assuming uniform routing)

## Detailed Cost Analysis

### Per-Expert GEMM Sizes
- Stage1: M=~140, N=4096 (2*d_expert_pad), K=7168 (d_hidden_pad) -- gate_up projection
- Stage2: M=~140, N=7168 (d_hidden_pad), K=2048 (d_expert_pad) -- down projection

### Per-Expert GEMM Timing Estimates (from GEMM benchmark data)
GEMM benchmark reference points:
- M=64, N=7168, K=2048: ~14us (gemm_a16wfp4 default)
- M=16, N=2112, K=7168: ~15us (gemm_a16wfp4 with custom config)
- M=256, N=3072, K=1536: ~16us (gemm_afp4wfp4)

For M=140 (larger M = proportionally more compute):
- Stage1 (M=140, N=4096, K=7168): estimate ~25-40us per call
  - N=4096 is larger than benchmark N values, K=7168 is large
  - Split-K would help but adds overhead
- Stage2 (M=140, N=7168, K=2048): estimate ~15-25us per call
  - Similar to benchmark (64, 7168, 2048) but with 2x more rows

### Total Time Estimate
- 33 experts * (35us stage1 + 20us stage2) = 33 * 55us = **~1815us**
- Plus overhead per expert:
  - Token gathering (index_select): ~2-5us
  - SiLU activation: ~2us
  - Weighted scatter-add (index_add_): ~2-5us
  - Kernel launch overhead: ~5-10us
- Per-expert overhead: ~15-20us
- Total overhead: 33 * 18us = ~594us
- **GRAND TOTAL: ~2400us = 7x WORSE than fused_moe's 337us**

## Why fused_moe is Better (Fundamental Architecture Advantage)

### 1. Single Kernel Launch vs 66 Launches
fused_moe launches 2 CK kernels total (stage1 + stage2). The manual loop launches 66 (33*2).
Each kernel launch has 5-10us host-side overhead on AMD GPUs.
Launch overhead alone: 66 * 7us = ~462us.

### 2. GPU Occupancy and Parallelism
fused_moe's CK kernel processes ALL experts simultaneously across 304 CUs.
- Stage1: all 33 experts' tokens processed in parallel across CUs
- The kernel internally assigns different M-blocks to different CUs/XCDs
- With M*topk=4608 tokens and block_m=32: ~144 M-blocks across 33 experts
- 144 blocks / 304 CUs = high occupancy

The manual loop processes experts SEQUENTIALLY:
- Expert 0's GEMM runs, then expert 1's, then expert 2's...
- Each M=~140 GEMM only generates ~4-5 M-blocks (at BM=32)
- 5 blocks across 304 CUs = <2% occupancy!
- The GPU is mostly idle during each per-expert GEMM

### 3. Triton JIT Compilation Risk
New (N, K) combinations not seen by the GEMM kernels trigger JIT recompilation:
- Stage1 N=4096, K=7168: likely NO pre-cached config on runner
- Stage2 N=7168, K=2048: possibly has cached config from GEMM benchmark
- Each new shape: 30-120s JIT compilation
- Combined: 60-240s JIT overhead (out of 720s test timeout)

Even if JIT succeeds, the FIRST benchmark run would include compilation time.

### 4. Data Movement Overhead
Manual loop has extra data movement:
- Token gathering: hidden_states[indices] = random-access reads
- Intermediate buffer allocation per expert
- Weighted scatter-add: atomic-like operations on output
None of these exist in fused_moe (it uses sorted_ids internally).

## Alternative: Batch All Experts Together

Could we avoid per-expert loops by padding all experts to max tokens and using batched GEMM?

### batched_gemm_a16wfp4
- Already proven: only batches SAME weight matrix across batch dimension
- Cannot handle different weight matrices per expert
- DEAD END

### torch.bmm with dequantized weights
- Dequant all 33 experts' weights to bf16: 33 * 4096 * 7168 * 2 bytes = ~1.8GB
- Plus 33 * 7168 * 2048 * 2 bytes = ~880MB
- Total ~2.7GB just for weight dequant
- And dequant time: ~100-300us per expert * 33 = prohibitive

### Custom Triton Grouped GEMM
- aiter already has fused_moe_mxfp4_silu which IS a grouped Triton GEMM
- All 6 submission attempts (v1-v6) TIMED OUT due to tl.dot_scaled JIT
- This IS the right architecture but the JIT compiler blocks it

## Verdict: NOT VIABLE

The manual expert loop approach is fundamentally ~7x slower than fused_moe because:
1. 66 sequential kernel launches vs 2 parallel ones
2. <2% GPU occupancy per expert vs full occupancy
3. Extra data movement for gather/scatter
4. Triton JIT risk for new (N,K) shapes

**The fused_moe CK pipeline's architecture (grouped GEMM across all experts in one launch) is exactly designed to avoid these problems.** The 337us for d=2048 is slow only because there's no tuned CK config for that shape -- not because the architecture is wrong.

## Real Path to Improving d=2048

The only paths that could improve d=2048:
1. **Find/create CK config for E=33 d=2048**: The dsv3 tuned CSV has no entry. If organizers update aiter (PR #2261), new configs may appear.
2. **Use the Triton MoE kernels (fused_moe_mxfp4_silu + fused_moe_mxfp4)**: These ARE grouped GEMMs that process all experts in one launch. But JIT compilation blocks them.
3. **Pre-compile Triton kernels offline**: Cache the JIT output so benchmark runs don't pay compilation cost. This requires submitting a warm-up-heavy script that compiles during test phase.
4. **Wait for runner to have cached Triton modules**: If another user's submission compiled the same kernel, it might be cached.

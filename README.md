# AMD x GPU MODE Hackathon - E2E Model Speedrun

Phase 1 Qualifiers: Optimizing GPU kernels for AMD Instinct MI355X

## Problems

### 1. MXFP4 GEMM (`mxfp4-mm/`)
Quantize bf16 activations to MXFP4 + block-scaled matrix multiplication.
- Input: bf16 A, pre-quantized MXFP4 B with E8M0 scales
- Output: bf16 C
- Baseline: aiter CK `gemm_a4w4`

### 2. MXFP4 MoE (`moe-mxfp4/`)
DeepSeek-R1 style fused Mixture-of-Experts with MXFP4 weights.
- 2-stage pipeline: gate+up GEMM with SwiGLU, then down GEMM with weighted reduction
- 256 routed + 1 shared expert, top-k=8
- Baseline: aiter `fused_moe`

### 3. MLA Decode (`mixed-mla/`)
DeepSeek-R1 Multi-head Latent Attention decode kernel.
- 16 query heads, 1 shared KV head (MQA)
- KV cache in bf16/fp8/mxfp4 formats
- Baseline: aiter `mla_decode_fwd` (a8w8 persistent kernel)

## Submission

```bash
# Install popcorn-cli
curl -fsSL https://raw.githubusercontent.com/gpu-mode/popcorn-cli/main/install.sh | bash
popcorn register discord

# Test
popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/submission.py

# Leaderboard
popcorn submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode leaderboard mxfp4-mm/submission.py
```

## Structure

```
mxfp4-mm/
  submission.py      # CK baseline
  submission_v3.py   # Triton GEMM variant
  reference.py       # Reference implementation
  task.py/task.yml   # Problem definition
moe-mxfp4/
  submission.py      # aiter fused_moe baseline
  reference.py
mixed-mla/
  submission.py           # Naive torch baseline
  submission_optimized.py # aiter persistent kernel (fast)
  submission_mxfp4.py     # mxfp4 KV experiment
  reference.py
```

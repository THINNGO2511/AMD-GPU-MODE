# AMD x GPU MODE Hackathon — E2E Model Speedrun

Phase 1 Qualifiers | Deadline: April 7, 2026 | GPU: AMD Instinct MI355X (gfx950, CDNA4)

## Current Standings (Mar 28, 2026)
| Problem | Score | Rank | Top 10 | Gap |
|---------|-------|------|--------|-----|
| MXFP4 GEMM | 16.222μs | 151 | 8.4μs | 2x |
| MLA Decode | 42.488μs | 11 | 41.8μs | 0.7μs |
| MXFP4 MoE | 169.131μs | ~57 | 136μs | 20% |

## Quick Start — Mac/Linux

```bash
git clone <repo-url> && cd AMD-GPU-MODE
pip install popcorn-cli          # or: cargo install popcorn-cli
popcorn-cli register             # login with Discord
bash restore_memory.sh           # restore Claude Code memory
python3 autosweep_gemm.py &     # GEMM config sweep
python3 autosweep_moe.py &      # MoE config sweep
python3 autoresearch.py &       # continuous monitoring
bash auto_submit.sh &           # hourly leaderboard submissions
# Open Claude Code → paste PICKUP_PROMPT.md
```

## Quick Start — Windows

```cmd
git clone <repo-url>
cd AMD-GPU-MODE
pip install popcorn-cli
popcorn-cli register
restore_memory.bat               &REM restore Claude Code memory
start python autosweep_gemm.py   &REM GEMM config sweep
start python autosweep_moe.py    &REM MoE config sweep
start python autoresearch.py     &REM continuous monitoring
start auto_submit.bat            &REM hourly leaderboard submissions
REM Open Claude Code → paste PICKUP_PROMPT.md
```

> **Note**: Python scripts auto-detect `popcorn-cli` location via `shutil.which()`. Make sure it's on your PATH.

## Project Structure

```
AMD-GPU-MODE/
├── PICKUP_PROMPT.md          # ⭐ Paste this into Claude Code to resume work
├── KNOWLEDGE.md              # Full technical knowledge base
├── CLAUDE.md                 # Competition rules + proven techniques
├── README.md                 # This file
│
├── autosweep_gemm.py         # 🔄 GEMM automated config sweep (runs forever)
├── autosweep_moe.py          # 🔄 MoE automated config sweep (runs forever)
├── autoresearch.py           # 🔄 Continuous monitoring (runs forever)
├── auto_submit.sh            # 🔄 Hourly leaderboard submissions
│
├── mxfp4-mm/                 # GEMM submissions
│   ├── submission_prewarm.py           # Current best (16.222μs)
│   ├── submission_stages3_all.py       # num_stages=3 experiment
│   ├── submission_envvars.py           # Triton env vars experiment
│   ├── submission_best_combined.py     # Envvars + fused K=512 kernel
│   ├── submission_fused_all.py         # Fused kernel all shapes
│   ├── submission_fused_k512_v2.py     # Fused kernel K=512 only
│   ├── submission_tutorial_gemm_v3.py  # Custom Triton tl.dot_scaled
│   ├── submission_custom_triton.py     # First custom kernel attempt
│   ├── submission_splitk_v1.py         # Split-K version
│   └── exp_*.py                        # Experimental probes
│
├── mixed-mla/                # MLA submissions
│   ├── submission_a16w8_pg2_pg8.py     # Current best (42.488μs) ⭐
│   ├── submission_pg2_pingpong.py      # pg2 + Triton BLOCK_PINGPONG
│   ├── submission_pg2_pg8_splits_fix.py # pg2 with fixed splits
│   ├── submission_pg8_v3.py            # Safe pg1+pg8 approach
│   ├── submission_pg1_all.py           # Ultra-safe pg1 all shapes
│   └── exp_*.py                        # Experimental probes
│
├── moe-mxfp4/                # MoE submissions
│   ├── submission_opus_sort.py         # Current best (169.131μs)
│   ├── submission_inject_metadata.py   # CK kernel injection (~167μs)
│   ├── submission_envvars_moe.py       # Triton env vars
│   ├── submission_compile.py           # torch.compile experiment
│   └── exp_*.py                        # Experimental probes
│
├── auto_research_logs/       # Sweep results + submission logs
│   ├── submissions.jsonl     # All submission history
│   ├── gemm_sweep.jsonl      # GEMM config sweep results
│   ├── moe_sweep.jsonl       # MoE config sweep results
│   └── research.jsonl        # Auto-research findings
│
└── discord-logs/             # Discord conversation archives
```

## Three Problems

### 1. MXFP4 GEMM (`mxfp4-mm/`)
- Quantize bf16 A → MXFP4, multiply with pre-quantized MXFP4 B → bf16 C
- 6 benchmark shapes: M=4/16/32/64/256, various N/K
- Key API: `gemm_a16wfp4(A, w, w_scales, dtype, y=None, config=None)`
- **Strategy**: Automated config sweep (BM/BN/BK/KSPLIT/stages/waves)

### 2. MLA Decode (`mixed-mla/`)
- DeepSeek-R1 Multi-head Latent Attention with mixed precision KV cache
- 8 benchmark shapes: bs=4/32/64/256, kv=1024/8192
- Key API: `mla_decode_fwd(...)` with pre-compiled ASM kernels
- **Strategy**: pg2 with reliability fix + Triton BLOCK_PINGPONG

### 3. MXFP4 MoE (`moe-mxfp4/`)
- DeepSeek-R1 fused MoE: 256+1 experts, top-8+1, SwiGLU, 2-stage pipeline
- 7 benchmark shapes, d=2048 dominates geomean at 333μs
- Key API: `fused_moe(...)` with CK kernel monkey-patching
- **Strategy**: Automated CK kernel × block_m sweep. Remove block_m override!

## Submission Commands
```bash
# Test (check accuracy, no timing)
popcorn-cli submit --gpu MI355X --leaderboard <name> --mode test <file> --no-tui

# Benchmark (timing, no leaderboard ranking)
popcorn-cli submit --gpu MI355X --leaderboard <name> --mode benchmark <file> --no-tui

# Leaderboard (official ranking — 1/hr limit!)
popcorn-cli submit --gpu MI355X --leaderboard <name> --mode leaderboard <file> --no-tui
```

Leaderboard names: `amd-mxfp4-mm`, `amd-mixed-mla`, `amd-moe-mxfp4`

## Rate Limits
- 6 submissions/hr per problem (test + benchmark + leaderboard share this)
- 1 leaderboard submission/hr per problem
- The word "stream" is GREP-FILTERED from submissions — don't use it!

## Key Learnings (14+ rounds of work)
- Custom Triton kernels work but JIT overhead negates speed in benchmarks
- Competitors use automated config sweeping (thousands of submissions)
- `num_stages=3` may be optimal on gfx950 (PR #2434)
- MoE: removing block_m override may improve score (competitor insight)
- MLA pg2: competitors made it reliable (unknown fix)
- Triton BLOCK_PINGPONG env var used by MLA competitor "pingpong.py"

## Claude Code Memory
Memory files are backed up in `claude-memory/` in this repo.
On a new machine, restore them:
```bash
bash restore_memory.sh
```
This copies 20+ memory files (learnings, dead ends, active leads, session results) into Claude Code's memory directory so it picks up exactly where we left off.

Key memory files:
- `active_leads.md` — current priorities
- `dead_ends.md` — what NOT to try (saves HOURS)
- `gemm_custom_kernel.md` — custom Triton kernel details
- `session9-14_results.md` — per-session results
- `feedback_nonstop.md` — never stop mode instructions

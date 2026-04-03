# ROUND 13: qseqlen2 MLA + preshuffle GEMM + continued exploration

## PART 1: Submit experiments

Setup:
cp claude-research-files/mla_qseqlen2.py mixed-mla/
cp claude-research-files/gemm_preshuffle_v2.py mxfp4-mm/

Submit as TEST:
1. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test mixed-mla/mla_qseqlen2.py --no-tui
2. popcorn-cli submit --gpu MI355X --leaderboard amd-mxfp4-mm --mode test mxfp4-mm/gemm_preshuffle_v2.py --no-tui
3. popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode leaderboard mixed-mla/mla_fp8q_all.py --no-tui

## PART 2: CONTINUE READING SOURCE CODE (while waiting for results)

These are the MOST IMPORTANT reads we haven't done yet:

### 2A. Read how mla_decode_fwd dispatches to qseqlen kernels
```bash
# Find the actual dispatch logic that selects qseqlen1 vs qseqlen2 vs qseqlen4
grep -n "qseqlen\|max_seqlen\|qseq" /home/runner/aiter/aiter/mla.py | head -30
# Read the full function that loads .co kernels
grep -n "def.*mla_decode\|def.*stage1\|hipModule\|hsaco\|\.co" /home/runner/aiter/aiter/mla.py | head -30
# Read the kernel selection code — what parameter triggers qseqlen2?
cat /home/runner/aiter/aiter/mla.py | head -400
```
REPORT: What EXACT condition dispatches to qseqlen2 vs qseqlen1? Is it max_seqlen_q? batch_size? A flag?

### 2B. Read the preshuffle Triton kernel to understand WHY it fails on fp4x2
```bash
cat /home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4_preshuffle.py 2>/dev/null
# Also check if there's a SEPARATE preshuffle function
grep -n "def.*preshuffle\|def.*preshuf" /home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py
```
REPORT: What does the preshuffle variant do differently? Can we work around the fp4x2 issue?

### 2C. Read the PRESHUFFLED JSON config for N=2112 K=7168
```bash
cat /home/runner/aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-A16WFP4_PRESHUFFLED-N=2112-K=7168.json
cat /home/runner/aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-A16WFP4_PRESHUFFLED.json
# Also read the generic A16WFP4 config to compare
cat /home/runner/aiter/aiter/ops/triton/configs/gemm/gfx950-GEMM-A16WFP4.json
```

### 2D. Read how the config loader works
```bash
grep -n "def get_gemm_config\|def _get_config\|def get_config" /home/runner/aiter/aiter/ops/triton/gemm/basic/gemm_a16wfp4.py | head -10
# Read the config loading function
grep -n "get_gemm_config\|load_config\|json.load" /home/runner/aiter/aiter/ops/triton/gemm/ -r | head -20
# Can we override configs via env var?
grep -n "environ\|AITER_CONFIG\|config_dir\|config_path" /home/runner/aiter/aiter/ops/triton/gemm/ -r | head -20
```
REPORT: Can we write shape-specific JSON configs that the loader picks up? Where?

### 2E. Read fused_moe get_2stage_cfgs logic
```bash
# Lines around get_2stage_cfgs that determine 1-stage vs 2-stage
grep -n "def get_2stage\|run_1stage\|stage1\|1stage\|one_stage" /home/runner/aiter/aiter/fused_moe.py | head -20
# Read the actual function
sed -n '/def get_2stage_cfgs/,/^def /p' /home/runner/aiter/aiter/fused_moe.py | head -100
```
REPORT: What conditions trigger 1-stage? Can we force it for E=33 d=2048?

### 2F. Check the _page kernel variant for MLA
```bash
# What's different about _page vs _ps kernels?
grep -n "_page\|page_table\|native_page" /home/runner/aiter/aiter/mla.py | head -20
# How to dispatch to _page variant?
```

### 2G. Can we write to the config directory?
```bash
ls -la /home/runner/aiter/aiter/ops/triton/configs/gemm/
touch /tmp/test_write && echo "CAN WRITE /tmp" || echo "CANNOT WRITE"
touch /home/runner/aiter/aiter/ops/triton/configs/gemm/test.json 2>&1 && rm /home/runner/aiter/aiter/ops/triton/configs/gemm/test.json && echo "CAN WRITE CONFIGS" || echo "CANNOT WRITE CONFIGS"
```

## After test results:
- MLA qseqlen2: if passes → benchmark immediately. READ ALL STDOUT.
- GEMM preshuffle: READ STDOUT for success/failure messages.
- If either works → LEADERBOARD IMMEDIATELY.

CRITICAL: Report ALL source code findings from Part 2 IN FULL. Every line matters.

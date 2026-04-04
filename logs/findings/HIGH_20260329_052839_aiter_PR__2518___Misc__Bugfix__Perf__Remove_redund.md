# [HIGH] aiter PR #2518: [Misc][Bugfix][Perf] Remove redundant bpreshuffle tuner, fix type hints warnings, add GLM-5 tuned configs

URL: https://github.com/ROCm/aiter/pull/2518

## Summary
- Remove redundant `gemm_a8w8_blockscale_bpreshuffle_tune.py` — `gemm_a8w8_blockscale_tune.py --preshuffle` already covers both ck and cktile backends
- Fix noisy `type hints mismatch` warnings caused by `typing.SupportsIndex` in pybind signatures
- Add 80 tuned a8w8_blockscale_bpreshuffle GEMM configs for GLM-5-FP8 (hidden_size=6144) on MI355X

## Test plan
- [ ] `python3 -c "import aiter"` — no type hints mismatch warnings
- [ ] `bash .github/scripts/op_tune.sh tune ck_gemm_a8w8_blo

Found: 2026-03-29 05:28:39.297523

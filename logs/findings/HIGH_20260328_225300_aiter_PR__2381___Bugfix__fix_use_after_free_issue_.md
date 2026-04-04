# [HIGH] aiter PR #2381: [Bugfix] fix use after free issue in moe_sorting_fwd

URL: https://github.com/ROCm/aiter/pull/2381

## Motivation

This PR fixed the use after free issue with torch::Tensor ws. 

## Technical Details

Declare torch::Tensor ws outside of curly braces {} to make the lifecycle of the buffer pointed to by ws_ptr valid.

## Test Plan
1) Run op_tests/test_moe_sorting.py tests.
2) Run Qwen3.5 inference to trigger this code snippet without any memory issue such as crash.

## Test Result
1) op_tests/test_moe_sorting.py passed.
2) Qwen3.5 inference with sglang, no memory issue such as cras

Found: 2026-03-28 22:53:00.168187

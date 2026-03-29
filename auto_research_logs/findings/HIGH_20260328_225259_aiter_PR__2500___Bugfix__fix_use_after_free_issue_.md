# [HIGH] aiter PR #2500: [Bugfix] fix use after free issue in moe_sorting_opus_fwd

URL: https://github.com/ROCm/aiter/pull/2500

## Motivation

This PR fixed the use after free issue with "torch::Tensor ws"

## Technical Details

Declare torch::Tensor ws outside of curly braces {} to make the lifecycle of the buffer pointed to by ws_ptr valid.

## Test Plan

1. Run op_tests/test_moe_sorting.py tests by:  AITER_USE_OPUS_MOE_SORTING=1 python test_moe_sorting.py

## Test Result

All cases passed.

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/devel

Found: 2026-03-28 22:52:59.662021

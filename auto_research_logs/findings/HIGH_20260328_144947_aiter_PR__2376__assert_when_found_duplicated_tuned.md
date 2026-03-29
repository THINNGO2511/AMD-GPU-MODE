# [HIGH] aiter PR #2376: assert when found duplicated tuned shape

URL: https://github.com/ROCm/aiter/pull/2376

## Motivation

assert error when there is duplicate tuned shape in configs

## Technical Details
1.
 Duplicate shape detection and auto-dedup in config merge                                                           
                                                                                                                     
  When update_config_files() merges tuned config CSVs from configs/ and model_configs/, it detects and handles       
  duplicate shape entries as follows: 

Found: 2026-03-28 14:49:47.547841

---
name: MoE Technical Findings
description: MoE fused_moe optimization — monkey-patching, kernel injection, dead ends
type: project
---

## Best Approach: submission_opus_sort.py (167-169μs)
- OPUS sorting enabled (`fm._USE_OPUS_MOE_SORTING = True`)
- use_nt=False for E≤64
- CK kernel injection for E≤64 d<2048 (S1_64/S1_256 + S2_V1 with block_m=32)
- d≥2048: NO injection (default heuristic)
- Custom block_size_M: 32 for est_m<50, 64 for est_m≥50

## Per-Shape Benchmark (submission_opus_sort.py)
| Shape | Time | Notes |
|-------|------|-------|
| E=257 bs=16 d=256 | 127μs | CK injected |
| E=257 bs=128 d=256 | 206μs | CK injected |
| E=257 bs=512 d=256 | 241μs | CK injected |
| E=33 bs=16 d=512 | 87μs | CK injected |
| E=33 bs=128 d=512 | 111μs | CK injected |
| E=33 bs=512 d=512 | 178μs | CK injected |
| E=33 bs=512 d=2048 | **337μs** | **GEOMEAN KILLER** — default heuristic |

## d=2048 Bottleneck
- CSV lookup on runner has NO entry for E=33, topk=9, d=2048
- fc0c54bb commit targets E=32, topk=8 — wrong shape
- CSV is exact 13-field match — will never hit
- Heuristic default DOES set a kernelName — `if not kernelName` check skips injection
- Large-tile injection (256x128x128x128): 348μs — NO improvement
- Custom AITER_CONFIG_FMOE CSV: 188μs — WORSE (overrides good defaults for other shapes)

## Monkey-Patching Pattern (proven working)
```python
import aiter.fused_moe as fm
fm.use_nt = lambda t, k, e: False if e <= 64 else orig_use_nt(t, k, e)
fm._USE_OPUS_MOE_SORTING = True
fm.get_block_size_M = lambda t, k, e, d: ...
orig_get_2stage = fm.get_2stage_cfgs.__wrapped__
# Override with @functools.lru_cache, return fm.MOEMetadata(...)
fm.cfg_2stages = None  # Clear cache
```

## Remaining Leads
- block_m=128 for E=33 bs=512 d=512 (est_m~140): experiment submitted, pending
- Stage2 v3 kernel for larger shapes: experiment submitted, pending
- Try NOT using opus sorting for E=257 (competitor "noopus" in filename)
- Per-shape use_nt: maybe NT is better for some E=257 shapes

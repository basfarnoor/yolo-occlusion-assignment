# Termination Comparison: Fixed vs. Adaptive Lifetime (Task 10)

Recall = fraction of 10 synthetic occlusion gaps (lengths 1-10 frames) successfully bridged. Ghost duration = how many frames a track that will NEVER return stays alive (capped at 20) -- a real cost, measured as a duration, not just a yes/no rate.

## Fixed-lifetime sweep

| max_missing_frames | recall | ghost duration (frames) |
|---:|---:|---:|
| 1 | 0.10 | 1 |
| 2 | 0.20 | 2 |
| 3 | 0.30 | 3 |
| 4 | 0.40 | 4 |
| 5 | 0.50 | 5 |
| 6 | 0.60 | 6 |
| 7 | 0.70 | 7 |
| 8 | 0.80 | 8 |
| 9 | 0.90 | 9 |
| 10 | 1.00 | 10 |

## Adaptive-lifetime sweep (frozen beta=0.15, alpha=0.01, uncertainty_ceiling=500.0)

| existence_floor | recall | ghost duration (frames) |
|---:|---:|---:|
| 0.6 | 0.20 | 2 |
| 0.4 | 0.30 | 3 |
| 0.25 | 0.40 | 4 |
| 0.15 | 0.60 | 6 |
| 0.1 | 0.70 | 7 |
| 0.07 | 0.70 | 7 |
| 0.05 | 0.80 | 8 |
| 0.03 | 0.90 | 9 |
| 0.02 | 1.00 | 10 |
| 0.01 | 1.00 | 11 |

## Comparison at matched ghost risk (<= 5 ghost frames)

| Policy | Operating point | Recall | Ghost duration |
|---|---|---:|---:|
| Fixed lifetime | max_missing_frames=5 | 0.50 | 5 |
| Adaptive | existence_floor=0.25 | 0.40 | 4 |

The fixed-lifetime policy actually recovers MORE genuine occlusions at this matched ghost-risk budget in this synthetic setup -- reported honestly. With constant-velocity motion and no real detector noise, a fixed frame count and an uncertainty ceiling end up drawing a very similar line; the adaptive policy's advantage should be expected to show up more clearly with noisier, more variable real motion than this idealized synthetic sweep provides.

## Frozen configuration

`beta=0.15`, `alpha=0.01`, `existence_floor=0.05`, `uncertainty_ceiling=500.0` (`configs/termination.yaml`) were chosen from this sweep and must not be re-tuned after any evaluation-scene result is opened in later tasks.

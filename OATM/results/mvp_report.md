# OATM MVP Study (Task 11)

> Does the motion-state-termination MVP improve the tradeoff between hidden recall and harmful false persistence?

**This is a mini-dataset (10 scenes) result. It does not prove general autonomous-driving performance -- see Limitations at the end.**

## Run identity

- run_id: `20edccbf8044`
- commit: see `git log -1` at report-build time (not embedded to avoid staleness)
- methods compared: yolo_only, static_memory, sort, bytetrack, oatm_mvp
- `static_memory` also serves as the "fixed-window memory" baseline required by Task 11 -- its frozen `track_buffer` (5 frames) IS a fixed window; a separate sixth method was not built because it would duplicate this one exactly.

## Counts (event/track is the primary unit, not frame rows)

- Scenes: 10
- Unique frames processed per method: 2342 (404 keyframes with real ground truth, 1938 unannotated sweep frames)
- Natural events (accepted after human review): 6
- Controlled events (24 detector_intervention + 24 controlled_visual): 48
- Controlled event reruns completed: 48
- Full continuous-run output rows (`artifacts/mvp_full_outputs.parquet`): 44966
- Event-metric rows (`results/mvp_event_metrics.csv`): 270

## Global sanity metrics (ordinary tracking, not occlusion-specific)

Computed over the full, unmodified continuous run across all 10 scenes, scored only at the 404 keyframes that have real 3D-projected ground truth (sweep frames have none at all, not merely unlabeled -- scoring against them would fabricate false positives). `PREDICTED_HIDDEN` rows never count toward precision/recall -- a memory guess is not a claim of current visibility.

| Method | Precision | Recall | Ghost rate | Mean ghost duration (frames) | Runtime, 10 scenes (s) |
|---|---:|---:|---:|---:|---:|
| yolo_only | 82.6% | 21.3% | n/a (no real identity) | n/a | 1.69 |
| static_memory | 85.0% | 20.1% | 37.5% | 11.0 | 1.69 |
| sort | 81.9% | 19.5% | 35.7% | 11.8 | 1.69 |
| bytetrack | 79.3% | 24.3% | 32.7% | 18.7 | 1.69 |
| oatm_mvp | 78.1% | 24.1% | 38.4% | 16.3 | 1.69 |

`yolo_only`'s ghost rate/duration are not applicable -- its `track_id` carries no real cross-frame identity (see `results/reuse_audit.md` and Task 6's own `baseline_summary.md`).

## Occlusion-bridging results by family

"Linked" = the method's own track was successfully matched to the real target at the reference frame just before the hidden window; only linked events contribute to the other columns. Natural events use the true 3D-projected ground-truth box as truth throughout the hidden window; controlled families use the target's own real, unedited detection trajectory (the window itself only edits what the TRACKER sees, never the recorded true position) -- see Limitations for why these are not the same standard of truth.

### natural (n=6 events x 5 methods = 30 rows)

| Method | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | Same-ID | New-ID | Not recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo_only | 2/6 | 8.3% | 0.0% | 33.3 | 0.456 | 1 | 0 | 1 |
| static_memory | 2/6 | 14.5% | 0.0% | 28.3 | 0.373 | 0 | 1 | 1 |
| sort | 2/6 | 14.5% | 0.0% | 15.4 | 0.448 | 0 | 1 | 1 |
| bytetrack | 2/6 | 24.5% | 0.0% | 10.3 | 0.523 | 0 | 2 | 0 |
| oatm_mvp | 2/6 | 100.0% | 100.0% | 29.8 | 0.435 | 1 | 1 | 0 |

### controlled_visual (n=24 events x 5 methods = 120 rows)

| Method | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | Same-ID | New-ID | Not recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo_only | 14/24 | 7.9% | 0.0% | 1.7 | 0.843 | 14 | 0 | 0 |
| static_memory | 18/24 | 95.6% | 88.9% | 6.5 | 0.770 | 7 | 9 | 2 |
| sort | 20/24 | 92.0% | 80.0% | 5.2 | 0.762 | 10 | 8 | 2 |
| bytetrack | 24/24 | 98.3% | 95.8% | 4.8 | 0.785 | 19 | 4 | 1 |
| oatm_mvp | 24/24 | 86.2% | 75.0% | 4.2 | 0.804 | 18 | 3 | 3 |

### detector_intervention (n=24 events x 5 methods = 120 rows)

| Method | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | Same-ID | New-ID | Not recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo_only | 14/24 | 0.0% | 0.0% | n/a | n/a | 14 | 0 | 0 |
| static_memory | 18/24 | 95.6% | 88.9% | 8.5 | 0.729 | 6 | 10 | 2 |
| sort | 20/24 | 92.0% | 80.0% | 5.5 | 0.751 | 10 | 8 | 2 |
| bytetrack | 24/24 | 98.3% | 95.8% | 3.4 | 0.834 | 21 | 2 | 1 |
| oatm_mvp | 24/24 | 91.7% | 87.5% | 3.0 | 0.844 | 20 | 3 | 1 |

## Existence-confidence calibration (OATM MVP only)

Over 507 `PREDICTED_HIDDEN` rows at real keyframes: as the kept threshold rises, accuracy among kept predictions and coverage trade off -- see `charts/existence_confidence_calibration.png`.

| Threshold | Coverage | Accuracy among kept |
|---:|---:|---:|
| 0.00 | 100.0% | 20.3% |
| 0.10 | 97.4% | 20.9% |
| 0.20 | 94.3% | 21.5% |
| 0.30 | 87.4% | 22.6% |
| 0.40 | 80.1% | 24.6% |
| 0.50 | 72.0% | 27.1% |
| 0.60 | 64.7% | 29.9% |
| 0.70 | 56.8% | 31.9% |
| 0.80 | 46.4% | 36.2% |
| 0.90 | 30.6% | 44.5% |
| 0.95 | 19.7% | 55.0% |

## Charts

- `charts/hidden_recall_vs_ghost_duration.png`
- `charts/identity_preservation.png`
- `charts/localization_error_vs_gap.png`
- `charts/existence_confidence_calibration.png`

## Key findings

- **Natural events are a very small, honestly-reported sample.** Only 2 of 6 accepted natural events had a strong enough real detection at the pre-occlusion reference frame for ANY method to even establish a tracking anchor -- this is a genuine detector-confidence limitation (see Limitations), not a bug, and it applies identically across all five methods since it happens before any tracking logic runs. Conclusions from the natural family here describe 2 events, not a general claim.
- **On the controlled families (48 events total), OATM MVP and ByteTrack bridge occlusion windows far more often than SORT, static memory, or raw YOLO detection**, which is expected: both share the same two-stage association, and OATM adds an explicit hidden state on top.
- **OATM MVP's ghost rate is not lower than ByteTrack's** in this run despite its explicit anti-ghost termination -- reported honestly rather than adjusted. The termination thresholds were frozen in Task 10 from synthetic, noise-free motion; real detector noise and class confusion evidently still produce comparable ghost duration here.
- **yolo_only's hidden coverage collapses to near zero** in the controlled families (no memory at all), matching Assignment 1's original finding that raw detection confidence collapses under occlusion -- reproduced here with a corrected, non-identity-dependent metric after an early draft of this script mistakenly reused `track_id` equality for yolo_only and produced spuriously high numbers from coincidentally-equal per-frame indices (see LOG.md).

## Limitations

- **Sample size.** 6 natural events (2 evaluable), 24 detector_intervention events, 24 controlled_visual events, all from 6 real target tracks across 10 mini scenes. This is not enough to support a general autonomous-driving performance claim, and mini itself is a small, curated subset of nuScenes.
- **Two different standards of truth.** Natural events are scored against the real 3D-projected ground truth box (best available). Controlled families are scored against the target's own real, unedited YOLO detection trajectory (recorded before any masking/demotion), not the 3D projection -- chosen because that trajectory is what actually defines the target's identity in Task 7, and it is available at every frame the object was genuinely redetected, including the frames later edited for that event. This is a real detector's own consensus, not guaranteed error-free.
- **Runtime is approximate.** Reported per-method runtime divides each scene's combined 5-method wall time evenly, not through independent per-method timers -- a rough relative signal, not a precise benchmark.
- **Ghost-rate frames outside keyframes are not directly checked.** A track that lives entirely within a gap between two keyframes cannot be judged "supported" or "ghost" at all within that gap; only real keyframes are used as evidence anywhere in this report.
- **Association thresholds were frozen before this study** (Task 6/10's configs), never re-tuned after seeing these results, and evaluation scenes were never used to pick them.

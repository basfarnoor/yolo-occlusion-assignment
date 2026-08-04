# Task 12 Appearance-Memory Ablation

> Does a frozen clear-view appearance anchor reconnect the correct identity better than motion alone?

- run_id: `30614cf75bbc`
- Events run: 48 x 3 modes
- Unique embeddings computed: 1057 (embedding computation bounded to a 15-frame lead-in before each event's pre-occlusion reference frame through the recovery-search horizon -- reconnection can only ever fire inside that window, so embedding the rest of each scene would cost real time for zero effect on any measured outcome)
- Total elapsed: 79.1s

Same 48 controlled events (24 detector_intervention + 24 controlled_visual) Task 11 used. `motion_only` is Task 11's OATM MVP unchanged (the ablation's baseline arm) -- differences from Task 11's own `oatm_mvp` numbers reflect only the narrower embedding-window scoping of the detections fed to it here, not a behavior change.

### detector_intervention (n=24 events x 3 modes = 72 rows)

| Mode | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | Same-ID | New-ID | Not recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| motion_only | 24/24 | 91.7% | 87.5% | 3.0 | 0.844 | 20 | 3 | 1 |
| appearance_only | 24/24 | 91.7% | 87.5% | 14.4 | 0.745 | 17 | 5 | 2 |
| dual | 24/24 | 91.7% | 87.5% | 5.1 | 0.807 | 18 | 5 | 1 |

### controlled_visual (n=24 events x 3 modes = 72 rows)

| Mode | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | Same-ID | New-ID | Not recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| motion_only | 24/24 | 86.2% | 75.0% | 4.2 | 0.804 | 18 | 3 | 3 |
| appearance_only | 24/24 | 86.2% | 75.0% | 17.3 | 0.734 | 15 | 6 | 3 |
| dual | 24/24 | 86.2% | 75.0% | 6.9 | 0.770 | 16 | 5 | 3 |

## Charts

- `charts/appearance_ablation_recovery.png`
- `charts/appearance_ablation_coverage.png`

## Finding

**Appearance did not earn its complexity here -- reported honestly, not adjusted.** The best appearance mode (`dual`) reached 70.8% same-ID recovery, actually below `motion_only`'s 79.2%. On this small mini sample, the frozen MobileNetV3-Small embedding does not appear to discriminate these specific same-class objects well enough to help, and may occasionally cause a wrong reconnection instead.

## Limitations

- Same small-sample caveats as `mvp_report.md`: 48 controlled events from 6 real target tracks across 10 mini scenes.
- The frozen embedder (MobileNetV3-Small, generic ImageNet features) was never fine-tuned or validated as a re-identification model specifically -- a purpose-built re-id embedding might perform differently.
- Embedding computation was deliberately scoped to a bounded window around each event (documented above); this cannot affect any measured event's outcome but does mean this script cannot answer questions about appearance drift over much longer gaps than tested here.

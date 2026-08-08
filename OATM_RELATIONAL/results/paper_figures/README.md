# Final OATM Paper Figure Pack

This directory contains the current eight-figure paper pack. Numerical figures
use only the scene-disjoint validation population from run
`lidar-fixes-20260805`: 1,873 projected car/pedestrian annotations, identical
frozen camera detections for all methods, and class-aware Hungarian matching at
IoU 0.30.

Build and validate the complete pack from `OATM_RELATIONAL/`:

```bash
UV_CACHE_DIR=/private/tmp/oatm-uv-cache \
MPLCONFIGDIR=/private/tmp/oatm-mpl \
uv run --frozen --no-sync python scripts/build_paper_figures.py
```

Run a non-retaining validation build:

```bash
UV_CACHE_DIR=/private/tmp/oatm-uv-cache \
MPLCONFIGDIR=/private/tmp/oatm-mpl \
uv run --frozen --no-sync python scripts/build_paper_figures.py --check
```

Every figure is available as editable SVG, vector PDF, and 600-DPI PNG. Figure
titles and explanatory caveats belong in the manuscript captions, not inside
the drawing canvas.

## Current figure manifest and captions

### Figure 1 — Online OATM architecture

- Stem: `oatm_online_architecture`
- Recommended section: Method
- Evidence type: causal camera-only method diagram
- Caption: **OATM online architecture.** Standard detection and association
  handle matched observations. When an established track remains unmatched,
  OATM evaluates occlusion evidence, initializes one primary target--occluder
  relation, compares independent motion prediction with occluder-relative
  reconstruction, and either emits a relation-supported prediction, restores
  the same identity, or terminates unsupported persistence.
- Boundary: YOLO, Kalman prediction, and ByteTrack association are the tracking
  foundation; the highlighted conditional branch is OATM.

### Figure 2 — Offline evaluation boundary

- Stem: `oatm_offline_evaluation`
- Recommended section: Experimental protocol
- Evidence type: evaluation-boundary diagram
- Caption: **Strictly separated offline evaluation.** Frozen camera-only OATM
  outputs are combined with projected nuScenes annotations, visibility labels,
  and LiDAR/radar support metadata only after online tracking has completed.
  Class-aware Hungarian matching produces the reported evaluation metrics; no
  privileged information returns to the tracker.
- Boundary: visibility and LiDAR/radar support are offline metadata, not online
  OATM inputs or exact pixel-level occlusion masks.

### Figure 3 — Final validation metric profile

- Stem: `figure_03_final_validation_metric_profile`
- Recommended section: Main results
- Source: `lidar-fixes-20260805`, validation, IoU 0.30, 1,873 annotations
- Caption: **Final scene-disjoint validation profile.** OATM achieved the
  highest precision, F1, MOTA, and IDF1 among the matched methods, while both
  ByteTrack settings retained slightly higher recall. Values are percentages.
- Boundary: the result supports the strongest balanced metric profile in this
  nuScenes-mini comparison, not universal OATM superiority.

### Figure 4 — Selective persistence tradeoff

- Stem: `figure_04_selective_persistence_tradeoff`
- Recommended section: Main results / persistence reliability
- Source: `lidar-fixes-20260805`, validation, IoU 0.30, 1,873 annotations
- Caption: **Selective persistence tradeoff.** OATM combines the highest
  predicted-hidden precision with the lowest unsupported-track rate and only
  40 false predicted-hidden matches, compared with 150 for ByteTrack-5 and 393
  for ByteTrack-12.
- Boundary: unsupported-track rate is a sparse-annotation proxy and must not be
  described as fully verified ghost duration.

### Figure 5 — Identity and localization comparison

- Stem: `figure_05_identity_localization_comparison`
- Recommended section: Identity and localization results
- Source: `lidar-fixes-20260805`, validation, IoU 0.30, 1,873 annotations
- Caption: **Identity continuity and localization.** OATM produced the fewest
  identity switches and fragmentations and the lowest mean center error among
  the evaluated methods. Lower values are better in all three panels.
- Boundary: localization is measured only on matched projected annotations at
  annotated keyframes.

### Figure 6 — Recall under severe visibility

- Stem: `figure_06_recall_under_severe_visibility`
- Recommended section: Limitations
- Source: `lidar-fixes-20260805`, validation, IoU 0.30, 1,873 annotations
- Caption: **Recall under severe visibility.** Recall falls substantially for
  every method in nuScenes visibility token 1, the most-occluded coarse bin.
  OATM reaches 8.3% in this bin versus 9.3% for ByteTrack-5 and 10.7% for
  ByteTrack-12, exposing the principal remaining recall limitation.
- Boundary: token 1 means 0--40% annotated visibility; it is not an exact
  `CAM_FRONT` pixel-occlusion mask.

### Figure 7 — Occluder-relative geometry

- Stem: `figure_07_occluder_relative_geometry`
- Recommended section: Method / relational memory
- Evidence type: conceptual geometry diagram
- Caption: **Occluder-relative geometry.** At relation initialization, OATM
  stores the target's normalized center offset and scale relative to one
  primary occluder. On later frames, the visible occluder reconstructs the
  hidden target, and support is accepted only when that reconstruction agrees
  with the independent target-motion prediction in center and scale.
- Boundary: the geometry is causal and camera-derived; it does not use offline
  projected annotations.

### Figure 8 — Temporal recovery sequence

- Stem: `figure_08_temporal_recovery_sequence`
- Recommended section: Method overview or qualitative explanation
- Evidence type: conceptual causal sequence
- Caption: **Causal recovery sequence.** A matched observation becomes
  temporarily unmatched behind a visible occluder. OATM emits explicitly
  labeled relation-supported predictions until expected clearance, then either
  reconnects a valid observation to the original identity or terminates the
  unsupported track.
- Boundary: dashed blue boxes are temporal predictions, not current visual
  detections.

## Legacy figures

Files named `figure_1_synthetic_*` through `figure_6_natural_*`, and the older
`oatm_architecture_pipeline` assets, are retained for historical traceability.
They mix synthetic mechanism studies or the earlier two-linkable-event pilot
and are not part of the current main-paper figure pack. Do not use their numbers
to support the final scene-disjoint claims.

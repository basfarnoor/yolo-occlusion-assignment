# OATM Project Log

## Current status

- **Phase:** Environment and data-pipeline validation.
- **Active branch:** `codex/organize-assignments`.
- **Methodology:** Defined in `METHODOLOGY.md`.
- **Dataset:** nuScenes mini is available locally at `../data/nuscenes/`.
- **Immediate milestone:** Prove that a clean OATM pipeline can load nuScenes
  mini, reconstruct chronological `CAM_FRONT` sequences, project annotations,
  and produce a verified candidate-occlusion index.

## 2026-07-29 — Installed nuScenes mini on IBEX

### Change

Downloaded and extracted the official `v1.0-mini.tgz` archive into the
Git-ignored repository data directory.

### Reason

Use the 10-scene mini split to validate the complete pipeline before committing
storage and compute to the trainval split.

### Validation

- Archive size matched the server response: 4,167,696,325 bytes.
- Gzip integrity and archive-path safety checks passed.
- Required `maps`, `samples`, `sweeps`, and `v1.0-mini` directories exist.
- Parsed 10 scenes, 404 keyframe samples, and 31,206 sample-data records.
- Found 2,342 `CAM_FRONT` records.
- Every sensor file referenced by `sample_data.json` exists.
- Git confirms that `data/nuscenes/` is excluded by `.gitignore`.

### Decision and next step

Keep the downloaded archive temporarily for reproducibility. Next, implement a
read-only dataset audit that checks temporal ordering, camera calibration,
ego-pose linkage, annotation projection, and candidate visibility transitions.

## 2026-07-28 — Created the OATM workspace

### Change

Moved the research methodology out of the completed coursework and into the
root-level `OATM/` workspace. Organized three earlier experiments under
`assignments/`.

### Reason

Separate completed learning assignments from the active research project while
preserving their evidence and Git history.

### Validation

- Assignment files, scripts, results, presentations, and samples remain present.
- Internal references were updated after the reorganization.
- Private source data remained untouched and excluded from Git.

### Decision and next step

Use the assignment results as baseline evidence, but implement the research
pipeline cleanly under `OATM/`.

## Prior evidence from completed assignments

1. A pretrained YOLO detector loses targets during full visual occlusion.
2. Static last-seen memory preserves existence briefly but becomes spatially
   stale and can create ghost predictions.
3. SORT-style motion prediction provides a stronger baseline than a frozen box,
   but simple motion assumptions remain limited by camera motion, turns, and
   association errors.

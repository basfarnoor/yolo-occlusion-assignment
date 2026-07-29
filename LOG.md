# OATM Project Log

## Current status

- **Phase:** Environment and data-pipeline validation.
- **Active branch:** `codex/organize-assignments`.
- **Methodology:** Defined in `OATM/METHODOLOGY.md`.
- **Dataset:** nuScenes mini is available locally at `data/nuscenes/`.
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

## 2026-07-27 — Experiment 2: Static last-seen memory

### Question

Can a simple memory prevent YOLO from immediately forgetting a fully occluded
object, and how accurately does a frozen last-seen box represent its later
position?

### Method

Selected five valid nuScenes samples from the first experiment. When YOLO lost
the chosen target, the experiment retained its most recent bounding box without
moving or resizing it. Memory was explicitly labeled as a prediction rather
than current camera evidence and expired after two missing stages. The frozen
box was compared with the new YOLO box when the target reappeared.

### Results

- Five samples were valid and three were rejected because target identity or
  true occlusion could not be confirmed reliably.
- Memory kept a target report alive through the one-stage occlusion gap in all
  five valid samples.
- Mean center error at reappearance was 330.6 px; median error was 422.8 px.
- Mean IoU was 0.085; median IoU was 0.
- Three of five frozen boxes had zero overlap with the reappearing target.
- Best case: `sample_003`, with 82.9 px center error and 0.339 IoU.
- Worst case: `sample_006`, with 589 px center error and 0 IoU.
- The human helpful/misleading and ghost-risk columns were not completed, so no
  subjective counts were inferred.

### Decision

Static memory proves that object existence can persist beyond a missed
detection, but a frozen location becomes stale quickly and can create ghost
objects. OATM therefore needs motion prediction, uncertainty growth, explicit
occlusion state, and termination logic.

### Evidence

- `assignments/02_last_seen_memory/results/final_report.md`
- `assignments/02_last_seen_memory/results/summary.csv`
- `assignments/02_last_seen_memory/results/comparisons/`

## 2026-07-26 — Experiment 1: YOLO occlusion sensitivity

### Question

How does increasing visual occlusion affect a pretrained YOLO detector's ability
to detect the same road user?

### Method

Ran pretrained `yolo26n.pt` in prediction-only mode on 30 organized
`CAM_FRONT` images from eight manually selected nuScenes samples. Images
represented up to five stages: visible, first partial occlusion, full
occlusion, first partial reappearance, and full reappearance. One target per
sample was manually reviewed.

### Results

- Detection was 8/8 before occlusion and 3/3 in the available first-partial
  stage.
- Detection fell to 1/8, or 12%, during full occlusion.
- Mean target confidence fell from 0.77 before occlusion to 0.59 during first
  partial occlusion and 0.03 during full occlusion.
- Detection recovered to 7/7 at full appearance, with mean usable confidence
  0.84.
- The first-partial-appearance result was 1/2, too small to generalize.
- The only reported full-occlusion detection, `sample_012` at confidence 0.23,
  may have been a box on the occluder; incomplete review fields prevented a
  definitive classification.

### Decision

The small study supports the core problem statement: single-frame detection
confidence collapses when visual evidence disappears and recovers when the
target returns. The sample is not large enough for statistical claims, so OATM
must build a larger, scene-split occlusion evaluation set.

### Evidence

- `assignments/01_yolo_occlusion/results/final_report.md`
- `assignments/01_yolo_occlusion/results/run_summary.md`
- `assignments/01_yolo_occlusion/results/analysis_summary.csv`

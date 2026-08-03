# OATM Project Log

## Current status

- **Phase:** Environment and data-pipeline validation.
- **Active branch:** `master`.
- **Methodology:** Defined in `OATM/METHODOLOGY.md`.
- **Implementation plan:** Defined in `OATM/IMPLEMENTATION_PLAN.md`.
- **Dataset:** nuScenes mini is available locally at `data/nuscenes/`.
- **Immediate milestone:** Prove that a clean OATM pipeline can load nuScenes
  mini, reconstruct chronological `CAM_FRONT` sequences, project annotations,
  and produce a verified candidate-occlusion index.

## 2026-08-03 — Added the student-facing OATM implementation assignment

### Change

Created `OATM/STUDENT_IMPLEMENTATION_ASSIGNMENT.md`, a Claude Code-ready master
assignment that divides the OATM implementation into 17 ordered tasks with
plain-language explanations, required outputs, automated checks, student
questions, and mentor stop points.

### Reason

The methodology and implementation plan define the research rigorously, but a
student needs a guided workflow like the four completed assignments: Claude
handles code and terminal work one task at a time while the student reviews the
scientific meaning and makes explicit human decisions.

### Validation

- Cross-checked the task order and terminology against `METHODOLOGY.md` and
  `IMPLEMENTATION_PLAN.md`.
- Preserved the Phase 0–1 stopping gate, causal camera-only boundary,
  scene-disjoint splitting, offline privileged-label boundary, separate event
  families, and recall-versus-ghost evaluation.
- Matched the instructional pattern of the earlier assignments with task goals,
  deliverables, completion checks, LLM rules, and student checkpoints.
- Ran `git diff --check` successfully.

### Decision and next step

Give the complete new assignment to Claude Code, begin with Task 0, and stop
after Task 2 for mentor review before projection or tracking work starts.

## 2026-08-01 — Tightened OATM methodology before experiments

### Change

Revised `OATM/METHODOLOGY.md` and `OATM/IMPLEMENTATION_PLAN.md` using the
evidence and limitations from Assignments 1–4. Separated strong detections,
weak detections, and prediction-only outputs; corrected the camera-only
ego-motion design; narrowed the first MVP; separated confidence quantities;
defined controlled visual occlusion versus detector intervention; and made
ghost-risk tradeoffs and valid experimental units explicit.

### Reason

The prior proposal combined too many unvalidated components, treated low
confidence too much like direct occlusion evidence, and contradicted its
camera-only claim by naming recorded ego pose as online input.

### Validation

- Cross-checked terminology, phase gates, schemas, metrics, and tomorrow's
  starting checklist across both OATM documents.
- Preserved causal inference and the privileged-data boundary.
- Grounded the revised scope in the small, preliminary assignment evidence
  without promoting pilot results into general claims.

### Decision and next step

Begin with Phase 0 and Phase 1 only. Establish the reproducible scaffold and
read-only chronological nuScenes-mini audit before projection, detection, or
tracking experiments.

## 2026-08-01 — Added a reproducible Assignment 4 environment

### Change

Added an assignment-local `pyproject.toml` for uv with the runtime and test
dependencies used by the ByteTrack experiment, plus an explicit `.venv`
ignore rule.

### Reason

Make the completed experiment and its automated checks reproducible without
depending on unrecorded system Python packages.

### Validation

- `uv lock` resolved 73 packages with standard PyTorch wheels, supporting CPU
  execution and CUDA when a compatible allocated resource is available.
- The assignment-local environment passed all 42 automated tests.
- The environment remains local and excluded from Git.

### Decision and next step

Use `uv sync` and `uv run pytest` from the Assignment 4 `experiment/`
directory whenever its saved results or implementation are audited.

## 2026-07-29 — Planned the OATM implementation

### Change

Created `OATM/IMPLEMENTATION_PLAN.md` with the pipeline, data contracts,
repository structure, phases, quality gates, experiment matrix, tests, risks,
definition of done, and a continuation checklist.

### Reason

Turn the methodology into a durable engineering and experimental handoff that
can resume without reconstructing decisions.

### Validation

- Fast-forwarded local `master` to merged `origin/master`.
- Grounded the plan in the methodology and earlier experiments.
- Kept privileged nuScenes evaluation data outside camera-only inference.
- Defined observed mini checks: 10 scenes, 404 keyframes, and 2,342
  `CAM_FRONT` records.

### Decision and next step

Implement Phase 0 and Phase 1 only: scaffold the project and build a read-only
chronological mini audit. Do not begin detection or tracking until that dataset
foundation passes its quality gate.

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

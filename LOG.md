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

## 2026-08-04 — OATM Task 6: rebuild and verify baselines

### Change

Wrote `results/reuse_audit.md` auditing every component before reuse from
Assignments 2-4. Reused Assignment 4's geometry/Hungarian-association/
timestamp-aware-Kalman code unchanged into `src/oatm/tracking/`; rebuilt
static memory (Assignment 2 had no reusable tracker-shaped code) and adapted
SORT/ByteTrack to emit OATM's canonical `TrackerOutputRecord` instead of
Assignment 4's lighter-weight output type. All four baselines (YOLO-only,
static memory, SORT, ByteTrack) now share one interface and ran over all 10
scenes from one `configs/tracker.yaml`.

### Reason

Later phases (motion memory, the occlusion state machine, the MVP report)
need a settled, identically-fed baseline layer to compare against -- and
Assignment 2's static memory in particular never had a real lifecycle, so it
could not previously be fairly compared to SORT/ByteTrack's buffer-based one.

### Validation

- 27 new tests, covering: exact
  IoU=1/IoU=0 cases, one-to-one assignment, dt-scaled Kalman prediction,
  birth/missing/reactivation/expiry for all three stateful baselines,
  strong-before-weak ByteTrack ordering, unmatched-weak-cannot-birth,
  scene-boundary track ID isolation, structural inaccessibility of ground
  truth (checked both by source-string search and by `update()`'s exact
  parameter set), determinism, and growing localization uncertainty during
  missing frames (a real Kalman covariance-trace value, not a placeholder).
  91 tests total, all passing. `ruff check` clean.
- Ran all four baselines over all 10 scenes (34,042 output rows, 4.2s).
  Caught and fixed a real bug in my own summary script: naively grouping
  YOLO-only's per-frame `track_id` by `(scene, track_id)` silently manufactured
  fake 97-row-long "tracks" out of unrelated single-frame detections that
  only shared a reused index number -- exactly the kind of misleading metric
  this project exists to prevent. Fixed by reporting those two columns as
  explicitly not applicable for a method with no real identity, rather than
  computing a number that would be misread as persistence.
- Sanity signal even at this pre-OATM stage: ByteTrack produced fewer, longer
  tracks (425 tracks, mean length 23.84) than SORT (470 tracks, 18.44) --
  consistent with reduced fragmentation, though this is baseline behavior
  only, not yet an evaluated result (that's Task 11).

### Decision and next step

Next: Task 7 (build the detector-intervention and controlled-visual-occlusion
event families -- two explicitly separate experiment types, extending
Assignment 4's controlled-experiment pattern rather than calling detection
edits "visual occlusion").

## 2026-08-04 — OATM Task 8: motion memory and growing uncertainty

### Change

Added `oatm.memory.{motion,motion_regimes,motion_comparison}` and
`scripts/run_motion_comparison.py`. Built seven synthetic motion-regime
fixtures with EXACTLY known ground truth (stationary, smooth, slow, unequal
timestamp gaps, turning, abrupt, missing-then-reappear) and compared
`StationaryPredictor` (frozen box) against the existing timestamp-aware
Kalman filter on each. Wrote the committed `results/motion_regime_report.md`
and the required diagram, `results/charts/uncertainty_growth.png`.

### Reason

Before adding any occlusion classifier or appearance memory, the project
needs to know, honestly and per motion regime, whether constant-velocity
prediction actually beats freezing the box -- and whether the system's own
uncertainty estimate behaves sensibly while evidence is missing.

### Validation

- 15 new tests (133 total), all passing: uncertainty grows monotonically
  during every regime's gap (not just some), stationary is exact on a
  non-moving object, Kalman clearly wins on smooth motion, the two models
  are close on slow motion, Kalman wins on irregular timestamps (the exact
  payoff of Assignment 4's dt-aware repair), and the abrupt/turning regimes
  are checked for a REPORTABLE result without asserting a predetermined
  winner.
- Caught and fixed an honesty issue in my own first draft: the report
  initially labeled the turning-motion result "negative/mixed" without
  checking the actual numbers -- Kalman still won there (51.35px vs.
  66.56px), just by a much smaller margin than on smooth motion (23% vs.
  100% error reduction). Rewrote that section to report the real numbers and
  explain the margin, rather than asserting a loss that didn't happen.
- Kalman won on mean center error in all seven regimes tested (including
  turning/abrupt, where its assumption is violated) -- an honest result, not
  a forced negative one; the report explains why that is not the same claim
  as "motion prediction always helps" and flags turning/abrupt as the
  regimes most likely to flip with real noise or sharper turns.
- Required diagram confirms uncertainty visibly grows (P-trace 20.7 -> 48.4
  -> 88.8) after 1, 3, and 5 missing frames.

### Decision and next step

Per this task's rule ("do not add visual ego-motion compensation or
appearance embeddings yet"), stopping the motion work here. Next: Task 9
(the OBSERVED_STRONG / OBSERVED_WEAK / PREDICTED_HIDDEN / LOST / EXITED
state machine).

## 2026-08-04 — OATM Task 7: controlled-occlusion event families

### Change

Implemented `oatm.dataset.controlled_occlusion` and
`scripts/build_controlled_events.py`. Selected 6 real target tracks (via the
ByteTrack baseline's now-correct raw detection boxes) and built two
explicitly separate event families: `detector_intervention` (demote/remove
the target's detection row, pixels untouched) and `controlled_visual` (paint
a seeded gray mask over the target on a local image copy, then rerun the
real frozen detector on that copy). 48 events total (24 each), varying
duration (2, 5 frames) and coverage (0.5, 1.0).

### Reason

Detector intervention and visual occlusion test fundamentally different
things (tracker logic in isolation vs. real re-detection under an actual
covered target) and must never be conflated or compared as if they were the
same experiment.

### Validation

- Found and fixed a determinism bug before it shipped: the original seed
  derivation used Python's built-in `hash()`, which is randomized per
  process (`PYTHONHASHSEED`) -- every rerun would have produced different,
  unreproducible masks. Replaced with a SHA-256-based deterministic digest.
- 11 new tests (same-seed same-mask, full-coverage mask matches the target
  box size, deterministic target selection/windowing, all four baselines
  receiving the identical unmutated detection list) plus 3 integration tests
  against the real manifest (both families present and never merged, every
  event has a seed and traceable target, visual events additionally record
  mask box and cache key). 103 tests total, all passing.
- **Visual checkpoint actually performed**, not just described: zoomed into
  full-resolution modified frames and confirmed the gray mask lands exactly
  on the intended target (a pedestrian at a bus stop -- shoes visible below
  the mask edge; a car at a distant intersection). Sent the full six-target
  before/after comparison to the student directly for her own confirmation.
- Confirmed the source nuScenes files are only ever opened for reading
  (`Image.open()` + `.copy()` before any drawing) -- never written to.

### Decision and next step

Next: Task 8 (motion memory -- compare stationary vs. timestamp-aware
constant-velocity prediction across synthetic motion regimes, with growing
localization uncertainty).

## 2026-08-04 — OATM Task 5: run detector once, cache observations

### Change

Implemented `oatm.detection.cache` (content-hash cache: image + model +
weights + image size + confidence floor + package versions) and
`scripts/run_detector.py`. Ran the same pretrained `yolo26n.pt` used in
Assignments 1-4 (prediction only, no training) over all 2,342 CAM_FRONT
frames across all 10 mini scenes, confidence floor 0.05.

### Reason

Every later baseline (SORT, ByteTrack, OATM) must compare fairly -- that
requires one shared, cached detector observation table, never a per-method
re-run.

### Validation

- First run: 2,342/2,342 cache misses (real inference), 49,436 raw
  detections, 285.4s wall-clock (0.17s/frame benchmark, within budget).
- Second run (identical inputs): 2,342/2,342 cache **hits**, 0 misses,
  identical 49,436-row output, 9.9s wall-clock -- a real, not just unit-tested,
  proof the cache works.
- 9 new tests (cache-key sensitivity to every field, round-trip, a
  call-counting fake-inference test proving a hit skips real work, plus
  integration checks against the real generated artifact: no out-of-range
  confidence, every detection traces to a real frame, weak detections
  actually survive the floor). 64 tests total, all passing. `ruff check`
  clean.
- Confidence-by-class table confirms the target MVP classes are well
  represented: 31,367 car, 5,097 person, 3,193 truck, 1,031 bus detections,
  spanning the full 0.05-0.95+ confidence range.

### Decision and next step

Next: Task 6 (rebuild and verify the YOLO-only / static-memory / SORT /
ByteTrack baselines behind one common interface, auditing Assignments 1-4's
code for reuse rather than copying it blindly).

## 2026-08-04 — OATM Task 4: mine and review natural occlusion events

### Change

Implemented `oatm.dataset.event_mining` (visibility decline-then-recovery
detection requiring a plausible closer occluder as a second independent
signal) and `scripts/mine_natural_events.py` /
`scripts/record_natural_event_review.py`. Assigned a scene-disjoint
development/validation/test split (6/2/2 of the 10 mini scenes, seeded) before
any event was selected. Wrote the committed
`results/natural_event_manifest.csv` and `results/natural_event_selection.md`.

### Reason

A visibility-label change alone is not proof of real occlusion (could be a
labeling artifact, an exit, or truncation) -- Task 4 requires two independent
signals plus an actual human visual review before any event counts as usable
evaluation evidence.

### Validation

- 54 candidate instances found a decline-then-recovery pattern with a
  plausible closer occluder; 19 more had the visibility pattern but no
  occluder and were auto-rejected (logged with reasons, never silently
  dropped, never shown for review).
- Shortlisted the top 15 (by occlusion length, then occluder overlap) and
  built a 15-row visual contact sheet (PRE/START/END/POST per candidate, the
  candidate occluder outlined in cyan).
- **The student reviewed the actual contact sheet** (sent directly, not just
  described) and gave a real per-candidate verdict: 6 accepted, 8 unsure
  (mostly "unclear to my eye," two flagged for multiple overlapping
  occluders), 1 rejected ("wrong objects"). Recorded verbatim, not invented.
- 10 new tests (event-boundary detection, occluder depth/overlap logic,
  deterministic scene-disjoint splitting), all passing (56 total). `ruff
  check` clean.
- Confirmed: every scene maps to exactly one split, every non-accepted event
  records why, and event boundaries are strictly ordered pre < start <= end <
  post (all tested against the real generated manifest).

### Decision and next step

Only the 6 `accepted` events are eligible for the later MVP evaluation
(Task 11) -- `unsure` events stay in the manifest for traceability but are
excluded, not silently promoted. This mini result is explicitly a pilot, not
a statistical conclusion (15 reviewed candidates from one dataset). Next:
Task 5 (run the frozen detector once, cache observations for every baseline).

## 2026-08-04 — OATM Task 3: project 3D ground truth into CAM_FRONT

### Change

Implemented `oatm.dataset.projection` (global -> ego -> camera -> pinhole
transform, behind-camera rejection, image-boundary clipping via shapely,
car/pedestrian evaluation-class mapping) and
`scripts/project_annotations.py` + `scripts/build_projection_overlays.py`.
Wrote local-only `projected_ground_truth.parquet` (5,384 accepted rows) and
`projection_rejections.json`, plus the committed `results/projection_audit.md`.

### Reason

Every later phase needs a trustworthy, independently-derived 2D ground truth
to score the tracker against -- one that can never leak into the camera-only
online path (METHODOLOGY.md's camera-only boundary).

### Validation

- 17 new tests (unit + integration), all passing (60 total in the project):
  known coordinate transform, behind-camera rejection, boundary clipping with
  nonzero truncation, positive finite area, deterministic ordering, MVP
  class-mapping, and identity preservation against the real dataset.
- Independent cross-check: the transform was reimplemented with `scipy`'s
  rotation library (instead of `pyquaternion`) and agrees to 1e-9 across 3
  rotation/translation cases -- satisfies the "compare with an independent
  reference" requirement without standing up the full official devkit.
- 5,384 of 18,538 considered annotations accepted (rest were behind the
  vehicle or outside the frame, as expected since annotations cover the full
  360-degree scene, not just CAM_FRONT); 3,492 map to the MVP's car/pedestrian
  scope.
- Visually reviewed 50 overlay frames (5 contact sheets, all 10 scenes, day/
  night/rain) -- tight alignment throughout, no systematic error, nothing
  needed a silent correction.
- `ruff check` clean; `git add -n` shows only source/tests/docs.

### Decision and next step

Task 3 has no separate student checkpoint in the assignment (only "automated
checks pass and every discrepancy is explained"), so continuing directly.
Next: Task 4 (mine and review natural occlusion events from real visibility
transitions) -- this task DOES have a student checkpoint requiring an actual
contact-sheet review with accept/reject/unsure answers, so it will pause for
that.

## 2026-08-04 — OATM Task 2: read-only mini audit + chronological CAM_FRONT index

### Change

Implemented `oatm.dataset.nuscenes_index` (chain-walking from each scene's
`prev`/`next` links, not filename or timestamp sort) and
`scripts/audit_dataset.py`. Wrote local-only `OATM/artifacts/frame_index.parquet`
(2,342 rows) and `dataset_audit.json`, plus the committed
`OATM/results/dataset_audit_summary.md`.

### Reason

Task 2's question: can the exact chronological CAM_FRONT stream be
reconstructed per scene, from the dataset's own links, with zero missing
files, zero broken links, and zero out-of-order frames? Nothing downstream
(projection, detection, tracking) can be trusted until this foundation is
verified.

### Validation

- Mini quality gate **PASSED** on the first real run against the local
  dataset: 10/10 scenes, 404/404 keyframes, 2,342/2,342 CAM_FRONT records,
  zero missing image files, zero non-monotonic timelines, every scene chain
  complete with exactly one head/tail and fully reciprocal links.
- 26/26 tests pass (9 new unit tests on synthetic fixtures + 3 new
  integration tests against the real local dataset); `ruff check` clean.
- Frame index rebuild is deterministic across two runs (tested).
- Student answered Task 2's checkpoint in her own words after one
  clarification each on two of the three questions (unsafe filename sorting;
  why future frames are forbidden online; why the large index stays local
  while the compact summary is committed).

### Decision and next step

**Stopping here per the assignment's mandatory mentor checkpoint** -- no
annotation projection, YOLO, tracking, or OATM logic begins until a mentor
reviews this Phase 0-1 output. Proposed a checkpoint commit; awaiting
confirmation before committing and before any Task 3 work starts.

## 2026-08-04 — OATM Task 0 (project map) and Task 1 (Phase 0 scaffold)

### Change

Created `OATM/results/project_map.md` (Task 0: plain-language terms, pipeline,
and method-comparison table). Then built the Phase 0 reproducible scaffold
(Task 1): `OATM/pyproject.toml` (pinned `pydantic`/`pyyaml`, dev deps
`pytest`/`ruff`), a project-local `OATM/.venv/`, `OATM/configs/mini.yaml`,
`src/oatm/config.py` (repo-relative data-root discovery, validated config with
readable errors), `src/oatm/records.py` (typed data contracts for every later
phase's frame index / projected ground truth / detector observations /
occlusion events / tracker output), 15 unit + integration tests, and
`OATM/README.md`. Added OATM-specific ignore rules to the root `.gitignore`.

### Reason

Task 0 gives a plain-English map before any code exists. Task 1 proves the
project can be installed and tested by someone else before any dataset,
detector, or tracker work begins -- exactly the assignment's "no experiment
before a trustworthy foundation" ordering.

### Validation

- Student answered Task 0's three checkpoint questions in her own words
  (`PREDICTED_HIDDEN` vs. a detection; why unlimited persistence would mislead
  recall; why staged builds stop one broken/masking component from hiding
  inside a bigger system).
- 15/15 pytest tests pass (`config` and `records` unit tests, one integration
  smoke test); `ruff check` passes with no findings.
- `find_data_root()` resolves the repo's real local `data/` folder (not
  `data/nuscenes/` as `AGENTS.md` describes -- same discrepancy already noted
  for the completed assignments) with no hardcoded absolute path; a missing or
  malformed config raises a readable `OATMConfigError` (tested).
- `git add -n` on `OATM/` shows only source, config, tests, and docs staged --
  no dataset, weights, `.venv/`, or cache directories.
- No YOLO run, no annotation projection, and no tracker code written yet, per
  Task 1's explicit scope limit.

### Decision and next step

Task 1's own checkpoint ("show `mini.yaml`, explain config vs. hardcoded vs.
private path") was answered inline in chat, not requiring a separate written
response. Next: Task 2 (read-only nuScenes mini audit and chronological
`CAM_FRONT` index) -- ending at the assignment's mandatory mentor checkpoint
before any projection, YOLO, or tracking work begins.

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

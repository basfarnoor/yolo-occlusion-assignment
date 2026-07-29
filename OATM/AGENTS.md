# OATM Working Instructions

These instructions apply to all work inside `OATM/`.

## Project purpose

OATM is the active research workspace for **Occlusion-Adaptive Temporal
Memory**, a camera-only object-persistence method for temporary occlusions in
autonomous-driving video.

Treat `METHODOLOGY.md` as the current scientific source of truth. If an
implementation decision materially changes the proposed method, evaluation, or
scope, update the methodology and record the decision in `LOG.md`.

## Scientific constraints

- The deployed method receives camera images only.
- nuScenes LiDAR-supported boxes, visibility labels, calibration, and ego poses
  may be used as privileged supervision or evaluation evidence, but must never
  be described as live camera-only input.
- Every online experiment must be causal: frame `t` may use the current and
  earlier frames, never future frames.
- Report natural and controlled occlusion results separately.
- Distinguish current visual detections from temporal predictions in saved
  outputs and metrics.
- Evaluate ghost tracks and identity errors, not only recovered-object recall.
- Split data by scene, never by neighboring frames.
- Do not claim novelty or benchmark reproduction beyond the evidence produced.

## Data safety

- The local nuScenes data root is `../data/nuscenes/`.
- `data/` is intentionally excluded by the repository `.gitignore`.
- Never add raw datasets, model weights, credentials, private workbooks,
  generated caches, or virtual environments to Git.
- Never modify or delete original nuScenes files. Derived subsets should record
  their source scene, sample token, channel, and timestamp.
- Before staging a commit, run `git status` and confirm no data artifacts are
  present.

## Workspace organization

Use these directories as the implementation grows:

- `src/` — reusable OATM implementation.
- `scripts/` — reproducible command-line entry points.
- `configs/` — versioned experiment settings.
- `tests/` — automated checks.
- `results/` — compact, reviewable reports and metrics.
- `artifacts/` — large or regenerable outputs; keep ignored unless explicitly
  approved for version control.

Avoid copying code from the completed `assignments/` experiments without first
deciding whether it is reusable, tested, and scientifically consistent with the
OATM methodology.

## Working procedure

1. Read `METHODOLOGY.md` and the latest entries in `LOG.md`.
2. State the experiment question and acceptance checks before implementation.
3. Prefer small, deterministic pipeline stages with explicit inputs and outputs.
4. Record random seeds, package versions, configuration, source scenes, and
   runtime environment for experiments.
5. Add or update tests for transformations, geometry, temporal ordering, and
   evaluation metrics.
6. Validate outputs before reporting results; never invent missing measurements.
7. Update `LOG.md` after every material implementation, experiment, decision, or
   blocker.

## Log format

Add new entries at the top of `LOG.md`, below the project status section. Each
entry should include:

- Date.
- Change or experiment.
- Reason.
- Validation or evidence.
- Decision and next step.

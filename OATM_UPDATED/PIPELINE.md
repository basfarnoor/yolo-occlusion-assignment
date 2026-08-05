# Reproducible Pipeline

## Stage 0: environment

`uv sync` creates the environment and locks direct and transitive dependencies.
The audited `OATM` package is an explicit editable path dependency.

## Stage 1: contract tests

Run `uv run pytest`. Tests verify causal state transitions, scene isolation,
selective admission, exit termination, ordinary-miss expiry, and identical
visible behavior between ByteTrack and Selective OATM.

## Stage 2: deterministic development experiment

Run `uv run python scripts/run_selective_study.py`. Seeded scenarios cover:

- Short and long occlusion with a visible overlapping occluder.
- Ordinary detector misses without an occluder.
- Outward field-of-view exit.

The runner writes row-level metrics, aggregate metrics, configuration, package
versions, seed, runtime, and a Markdown report under `results/`.

## Stage 3: real-data controlled visual evaluation

Build fresh detector outputs from unmodified nuScenes images, create pixel-level
controlled occlusions, rerun the detector, and compare all methods on identical
observations. Detector-row deletion remains labeled `detector_intervention`.

## Stage 4: natural-event evaluation

Mine candidates using privileged annotations, manually verify them, freeze
scene-disjoint development/validation/test manifests, and run trackers without
access to privileged fields. Include exits and ordinary misses as negative
events.

## Stage 5: decision

Select thresholds using development scenes only. Freeze them before test-scene
evaluation. Promote a component only when its ablation improves the matched-risk
frontier. Record every material result or blocker in the root `LOG.md`.

# CAM_FRONT LiDAR-Supported Evaluation

This directory provides a reproducible nuScenes-mini evaluation of
Relational OATM and matched ByteTrack baselines. The deployed trackers remain
strictly camera-only and causal.

## Scientific boundary

The pipeline is intentionally split into three stages:

1. The frozen YOLO detector reads `CAM_FRONT` images.
2. Each tracker reads only those camera detections, timestamps, the current
   `CAM_FRONT` image when its camera-motion ablation is enabled, and its own
   causal history.
3. Only after tracker outputs are saved, the evaluator loads projected
   nuScenes 3D annotations, visibility labels, instance tokens, and LiDAR/radar
   point counts for offline scoring.

LiDAR, calibration, annotations, ego poses, and visibility labels never enter
detector or tracker inference. Sweep frames are processed by the tracker but
only the 404 official annotated `CAM_FRONT` keyframes are scored. No future
frame and no interpolated annotation is used for headline metrics.

The ten mini scenes are assigned as whole units to five development and five
validation scenes by a seeded SHA-256 rank. Neighboring frames can never cross
the split. Validation is the headline population; pooled results are labeled
as diagnostics.

## What the run produces

Each job writes an isolated directory under `lidar_eval/results/`:

- `report.md` — readable headline, visibility, point-support, and sensitivity
  tables.
- `summary.csv` — precision, recall, F1, MOTA, IDF1, ID switches,
  fragmentation, IoU, and center error at every configured IoU threshold.
- `stratified_metrics.csv` and `.parquet` — ground-truth-centric results by
  class, visibility, distance, LiDAR support, and truncation.
- `ground_truth_matches.parquet` — one auditable row per projected annotation
  and method at the primary IoU threshold.
- `unmatched_outputs.parquet` — false-positive evidence at annotated
  keyframes.
- `summary.csv` also reports unsupported-keyframe-track rate as a clearly
  labeled sparse-annotation ghost proxy.
- `tracker_outputs.parquet` — all causal outputs across all `CAM_FRONT`
  frames, with observed detections distinguished from temporal predictions.
- `run_metadata.json` — exact config, hashes, package versions, Git state,
  Slurm environment, row counts, and runtimes.
- `detector_cache_audit.json` — whether camera detections were validated and
  reused or regenerated.

Generated results are ignored by Git but remain in place for later inspection.
Nothing under `data/nuscenes/` is modified.

## One-time login-node check

Run from `OATM_RELATIONAL/`:

```bash
test -x .venv/bin/python
uv run --frozen python -m lidar_eval.detector --help
uv run --frozen pytest lidar_eval/tests
```

The job uses the existing `OATM_RELATIONAL/.venv`; it does not create another
environment. If `.venv` is absent, create it once on the login node:

```bash
uv sync --locked
```

The local mini dataset must remain at `../data/nuscenes/`, and the frozen
weights must remain at `../yolo26n.pt`. Both are ignored by Git.

## Submit the four-hour A100 job

From `OATM_RELATIONAL/`:

```bash
sbatch lidar_eval/submit_a100.sbatch
```

The default run validates and reuses `artifacts/detections.parquet` when its
weight hash, frame count, geometry, confidence threshold, and cache keys are
consistent. In that case the GPU is allocated for reproducibility but detector
inference is skipped.

To deliberately regenerate every camera detection on the A100:

```bash
sbatch --export=ALL,REFRESH_DETECTOR=1 lidar_eval/submit_a100.sbatch
```

A single A100 and four hours is comfortably conservative for nuScenes mini.
The earlier CPU detector run took about 15 minutes; tracking and keyframe
evaluation are much smaller stages. The Slurm request uses 16 CPU cores and
64 GB RAM to leave comfortable headroom.

Monitor the job with commands supported by the cluster, commonly:

```bash
squeue -u "$USER"
tail -f lidar_eval/results/slurm-oatm-lidar-eval-JOB_ID.out
```

Results for job `JOB_ID` appear in:

```text
lidar_eval/results/nuscenes-mini-JOB_ID/
```

Some clusters require `#SBATCH --partition=...`, `#SBATCH --account=...`, or a
different GPU resource spelling such as `--gpus=a100:1`. Add only the directives
required by the local scheduler; the supplied file requests
`--gres=gpu:a100:1`.

## Local CPU reuse run

For a quick run that reuses the validated detector cache:

```bash
uv run --frozen python scripts/prepare_nuscenes.py
uv run --frozen python -m lidar_eval.run \
  --config lidar_eval/config.yaml \
  --run-id local-mini \
  --output-dir lidar_eval/results/local-mini \
  --device cpu
```

Output directories are never overwritten. Choose a new `--run-id`, or move an
old result directory somewhere safe before rerunning.

## Interpretation

The primary denominator contains every accepted projected car/pedestrian box,
including zero-LiDAR-point and truncated annotations. Filtering zero-point
objects would preferentially remove difficult occlusions. Point support and
truncation are therefore reported as sensitivity strata instead.

The principal class-aware matching gate is IoU 0.30; IoU 0.10 and 0.50 are
reported as sensitivity checks. This nuScenes-mini study improves the evidence
well beyond the prior two-linkable-event pilot, but it remains a mini-dataset
evaluation—not a full nuScenes benchmark reproduction or a guaranteed
superiority result. Development/validation separation prevents frame leakage,
but does not turn ten scenes into a statistically powered benchmark.

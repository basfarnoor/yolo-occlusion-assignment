"""Create a compact, claim-bounded Markdown report from saved metrics."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _table(frame: pd.DataFrame, columns: list[str], floatfmt: str = ".3f") -> str:
    available = [column for column in columns if column in frame.columns]
    return frame[available].to_markdown(index=False, floatfmt=floatfmt)


def build_report(
    run_id: str,
    summary: pd.DataFrame,
    strata: pd.DataFrame,
    dataset_stats: dict[str, Any],
    primary_threshold: float,
) -> str:
    primary = summary[
        (summary.iou_threshold == primary_threshold) & (summary.population == "validation")
    ].copy()
    pooled = summary[
        (summary.iou_threshold == primary_threshold) & (summary.population == "all")
    ].copy()
    sensitivity = summary[summary.population == "validation"].copy()
    validation_strata = strata[strata.population == "validation"]
    visible = validation_strata[
        (validation_strata.dimension == "visibility") & (validation_strata.value == "4")
    ]
    severe = validation_strata[
        (validation_strata.dimension == "visibility") & (validation_strata.value == "1")
    ]
    lidar_support = validation_strata[validation_strata.dimension == "lidar_support"]
    primary_table = _table(
        primary,
        [
            "method_name",
            "precision",
            "recall",
            "f1",
            "mota",
            "idf1",
            "id_switches",
            "fragmentations",
            "unsupported_keyframe_track_rate",
            "mean_unsupported_keyframe_rows",
            "mean_iou",
            "mean_center_error_px",
        ],
    )
    pooled_table = _table(
        pooled,
        [
            "method_name",
            "precision",
            "recall",
            "f1",
            "mota",
            "idf1",
            "unsupported_keyframe_track_rate",
            "mean_center_error_px",
        ],
    )
    state_table = _table(
        primary,
        [
            "method_name",
            "observed_tp",
            "observed_fp",
            "observed_precision",
            "observed_share_of_gt",
            "predicted_hidden_tp",
            "predicted_hidden_fp",
            "predicted_hidden_precision",
            "predicted_hidden_share_of_gt",
        ],
    )
    stratum_columns = [
        "method_name",
        "gt_annotations",
        "matched",
        "recall",
        "observed_matches",
        "predicted_hidden_matches",
        "mean_center_error_px",
    ]
    visible_table = _table(visible, stratum_columns)
    severe_table = _table(severe, stratum_columns)
    lidar_table = _table(lidar_support, ["method_name", "value", *stratum_columns[1:]])
    sensitivity_table = _table(
        sensitivity,
        [
            "method_name",
            "iou_threshold",
            "precision",
            "recall",
            "f1",
            "mota",
            "idf1",
            "mean_center_error_px",
        ],
    )
    return f"""# CAM_FRONT LiDAR-Supported Evaluation

Run ID: `{run_id}`

## Scientific boundary

All online inference used only current/earlier `CAM_FRONT` images, frozen
camera detections, timestamps, and causal tracker history. Projected nuScenes
3D annotations, visibility labels, calibration, instance tokens, LiDAR/radar
point counts, and ego poses were loaded only after tracker outputs had been
saved. They are privileged offline evaluation evidence, never model input.

## Evaluation population

- Dataset: `nuScenes {dataset_stats['dataset_version']}`
- Scenes: {dataset_stats['scenes']}
- `CAM_FRONT` inference frames: {dataset_stats['camera_frames']}
- Official annotated keyframes scored: {dataset_stats['keyframes']}
- In-scope projected car/pedestrian annotations: {dataset_stats['gt_annotations']}
- Distinct in-scope instances: {dataset_stats['instances']}
- Development scenes: {dataset_stats['development_scenes']}
  ({dataset_stats['development_annotations']} annotations)
- Validation scenes: {dataset_stats['validation_scenes']}
  ({dataset_stats['validation_annotations']} annotations)
- Primary class-aware Hungarian IoU gate: {primary_threshold:.2f}

Sweep frames without official sample annotations were processed causally but
were not treated as empty ground truth. All official projected annotations,
including zero-LiDAR-point and truncated boxes, remain in the headline
denominator. Quality strata below are sensitivity evidence, not cherry-picked
replacement denominators.

## Primary validation results

{primary_table}

An unsupported keyframe track never matched an official annotation on any
annotated frame where it appeared. This is a sparse-label ghost proxy, not a
claim that unannotated sweep-frame duration is verified ground truth.

## Pooled diagnostic results

{pooled_table}

## Observed detections versus temporal predictions

{state_table}

## Visibility-stratified recall

Official visibility token `4` is the most visible bin; token `1` is the most
occluded bin. These labels are coarse annotation evidence, not exact
per-camera pixel-occlusion masks.

Most visible (`4`):

{visible_table}

Most occluded (`1`):

{severe_table}

## LiDAR-point support sensitivity

Zero-point annotations are retained because filtering them would remove many
hard occlusion cases. This table shows whether conclusions depend on point
support without changing the primary population.

{lidar_table}

## Matching-threshold sensitivity

{sensitivity_table}

## Interpretation boundary

This is a reproducible nuScenes-mini evaluation, not a full-benchmark
reproduction or a statistically powered superiority claim. A credible method
improvement requires better occluded-object persistence and identity behavior
without an unacceptable rise in predicted-hidden false positives, ghost
duration, identity switches, or visible-object precision loss. The stored
row-level matches permit every aggregate above to be audited.
"""

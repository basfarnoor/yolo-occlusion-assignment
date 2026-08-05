"""Detection, persistence, localization, and sparse-keyframe identity metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from lidar_eval.matching import OBSERVED_STATES, PREDICTED_STATE


def safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _state_counts(matches: pd.DataFrame, false_positives: pd.DataFrame, state: str) -> dict[str, Any]:
    if state == "observed":
        match_mask = matches.state.isin(OBSERVED_STATES)
        fp_mask = false_positives.state.isin(OBSERVED_STATES) if not false_positives.empty else []
    elif state == "predicted_hidden":
        match_mask = matches.state == PREDICTED_STATE
        fp_mask = false_positives.state == PREDICTED_STATE if not false_positives.empty else []
    else:
        raise ValueError(f"unknown state group {state}")
    true_positives = int(match_mask.sum())
    false_positive_count = int(np.sum(fp_mask))
    predictions = true_positives + false_positive_count
    return {
        f"{state}_tp": true_positives,
        f"{state}_fp": false_positive_count,
        f"{state}_outputs": predictions,
        f"{state}_precision": safe_divide(true_positives, predictions),
        f"{state}_share_of_gt": safe_divide(true_positives, len(matches)),
    }


def identity_metrics(matches: pd.DataFrame, prediction_count: int) -> dict[str, Any]:
    """Compute sparse-keyframe IDF1, ID switches, fragmentation, and coverage classes."""
    matched = matches[matches.matched].copy()
    pair_counts: dict[tuple[tuple[str, str], tuple[str, int]], int] = defaultdict(int)
    for row in matched.to_dict("records"):
        truth_identity = (row["scene_token"], row["instance_token"])
        prediction_identity = (row["scene_token"], int(row["track_id"]))
        pair_counts[(truth_identity, prediction_identity)] += 1
    truth_ids = sorted({pair[0] for pair in pair_counts})
    prediction_ids = sorted({pair[1] for pair in pair_counts})
    id_true_positives = 0
    if truth_ids and prediction_ids:
        truth_index = {identity: index for index, identity in enumerate(truth_ids)}
        prediction_index = {identity: index for index, identity in enumerate(prediction_ids)}
        contingency = np.zeros((len(truth_ids), len(prediction_ids)), dtype=int)
        for (truth_identity, prediction_identity), count in pair_counts.items():
            contingency[truth_index[truth_identity], prediction_index[prediction_identity]] = count
        rows, columns = linear_sum_assignment(contingency, maximize=True)
        id_true_positives = int(contingency[rows, columns].sum())
    id_false_positives = prediction_count - id_true_positives
    id_false_negatives = len(matches) - id_true_positives

    switches = 0
    fragmentations = 0
    coverage_ratios = []
    for _, trajectory in matches.sort_values("frame_index").groupby(
        ["scene_token", "instance_token"], sort=False
    ):
        previous_track: int | None = None
        seen_match = False
        in_unmatched_gap = False
        for row in trajectory.to_dict("records"):
            if bool(row["matched"]):
                track_id = int(row["track_id"])
                if previous_track is not None and track_id != previous_track:
                    switches += 1
                if seen_match and in_unmatched_gap:
                    fragmentations += 1
                previous_track = track_id
                seen_match = True
                in_unmatched_gap = False
            elif seen_match:
                in_unmatched_gap = True
        coverage_ratios.append(float(trajectory.matched.mean()))

    mostly_tracked = sum(ratio >= 0.8 for ratio in coverage_ratios)
    mostly_lost = sum(ratio < 0.2 for ratio in coverage_ratios)
    partially_tracked = len(coverage_ratios) - mostly_tracked - mostly_lost
    id_precision = safe_divide(id_true_positives, id_true_positives + id_false_positives)
    id_recall = safe_divide(id_true_positives, id_true_positives + id_false_negatives)
    return {
        "idtp": id_true_positives,
        "idfp": id_false_positives,
        "idfn": id_false_negatives,
        "id_precision": id_precision,
        "id_recall": id_recall,
        "idf1": safe_divide(
            2 * id_true_positives,
            2 * id_true_positives + id_false_positives + id_false_negatives,
        ),
        "id_switches": switches,
        "fragmentations": fragmentations,
        "trajectories": len(coverage_ratios),
        "mostly_tracked": mostly_tracked,
        "partially_tracked": partially_tracked,
        "mostly_lost": mostly_lost,
    }


def unsupported_track_metrics(
    matches: pd.DataFrame, false_positives: pd.DataFrame
) -> dict[str, Any]:
    """Keyframe-only ghost proxy; never pretends sparse labels verify every sweep."""
    appearances: list[dict[str, Any]] = []
    for row in matches[matches.matched].to_dict("records"):
        appearances.append(
            {
                "scene_token": row["scene_token"],
                "track_id": int(row["track_id"]),
                "supported": True,
            }
        )
    for row in false_positives.to_dict("records"):
        appearances.append(
            {
                "scene_token": row["scene_token"],
                "track_id": int(row["track_id"]),
                "supported": False,
            }
        )
    if not appearances:
        return {
            "keyframe_tracks": 0,
            "unsupported_keyframe_tracks": 0,
            "unsupported_keyframe_track_rate": None,
            "mean_unsupported_keyframe_rows": None,
        }
    frame = pd.DataFrame(appearances)
    grouped = frame.groupby(["scene_token", "track_id"]).supported.agg(["any", "size"])
    unsupported = grouped[~grouped["any"]]
    return {
        "keyframe_tracks": len(grouped),
        "unsupported_keyframe_tracks": len(unsupported),
        "unsupported_keyframe_track_rate": safe_divide(len(unsupported), len(grouped)),
        "mean_unsupported_keyframe_rows": (
            float(unsupported["size"].mean()) if not unsupported.empty else None
        ),
    }


def summarize_method(
    matches: pd.DataFrame,
    false_positives: pd.DataFrame,
    iou_threshold: float,
) -> dict[str, Any]:
    method_names = matches.method_name.unique().tolist()
    if len(method_names) != 1:
        raise ValueError("summarize_method accepts exactly one method")
    method_name = method_names[0]
    true_positives = int(matches.matched.sum())
    false_negatives = len(matches) - true_positives
    false_positive_count = len(false_positives)
    prediction_count = true_positives + false_positive_count
    precision = safe_divide(true_positives, prediction_count)
    recall = safe_divide(true_positives, len(matches))
    matched = matches[matches.matched]
    summary = {
        "method_name": method_name,
        "iou_threshold": iou_threshold,
        "gt_annotations": len(matches),
        "keyframe_outputs": prediction_count,
        "tp": true_positives,
        "fp": false_positive_count,
        "fn": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": (
            safe_divide(2 * precision * recall, precision + recall)
            if precision is not None and recall is not None
            else None
        ),
        "mota": 1.0,
        "mean_iou": float(matched.iou.mean()) if not matched.empty else None,
        "median_iou": float(matched.iou.median()) if not matched.empty else None,
        "mean_center_error_px": (
            float(matched.center_error_px.mean()) if not matched.empty else None
        ),
        "median_center_error_px": (
            float(matched.center_error_px.median()) if not matched.empty else None
        ),
    }
    summary.update(_state_counts(matches, false_positives, "observed"))
    summary.update(_state_counts(matches, false_positives, "predicted_hidden"))
    identity = identity_metrics(matches, prediction_count)
    summary.update(identity)
    summary.update(unsupported_track_metrics(matches, false_positives))
    summary["mota"] = 1.0 - safe_divide(
        false_negatives + false_positive_count + identity["id_switches"], len(matches)
    )
    return summary


def _depth_bin(value: float, edges: list[float]) -> str:
    if value < edges[0]:
        return f"< {edges[0]:g} m"
    if value < edges[1]:
        return f"{edges[0]:g}-{edges[1]:g} m"
    return f">= {edges[1]:g} m"


def _lidar_bin(value: int, edges: list[int]) -> str:
    if value < edges[0]:
        return "0 points"
    if value < edges[1]:
        return f"{edges[0]}-{edges[1] - 1} points"
    return f">= {edges[1]} points"


def stratified_metrics(matches: pd.DataFrame, strata_config: dict[str, Any]) -> pd.DataFrame:
    """GT-centric strata; intentionally does not misassign subgroup false positives."""
    enriched = matches.copy()
    depth_edges = [float(item) for item in strata_config["depth_edges_m"]]
    lidar_edges = [int(item) for item in strata_config["lidar_point_edges"]]
    if len(depth_edges) != 2 or depth_edges != sorted(depth_edges):
        raise ValueError("depth_edges_m must contain two sorted values")
    if len(lidar_edges) != 2 or lidar_edges != sorted(lidar_edges):
        raise ValueError("lidar_point_edges must contain two sorted values")
    truncation_threshold = float(strata_config["truncation_threshold"])
    enriched["depth_bin"] = enriched.center_depth_m.map(lambda value: _depth_bin(value, depth_edges))
    enriched["lidar_support_bin"] = enriched.num_lidar_pts.map(
        lambda value: _lidar_bin(int(value), lidar_edges)
    )
    enriched["truncation_bin"] = np.where(
        enriched.truncation_fraction <= truncation_threshold,
        f"<= {truncation_threshold:g}",
        f"> {truncation_threshold:g}",
    )
    dimensions = {
        "class": "evaluation_class",
        "visibility": "visibility_token",
        "depth": "depth_bin",
        "lidar_support": "lidar_support_bin",
        "truncation": "truncation_bin",
    }
    rows: list[dict[str, Any]] = []
    for dimension, column in dimensions.items():
        for value, group in enriched.groupby(column, sort=True):
            matched = group[group.matched]
            rows.append(
                {
                    "method_name": group.method_name.iloc[0],
                    "dimension": dimension,
                    "value": str(value),
                    "gt_annotations": len(group),
                    "matched": int(group.matched.sum()),
                    "recall": float(group.matched.mean()),
                    "observed_matches": int(matched.state.isin(OBSERVED_STATES).sum()),
                    "predicted_hidden_matches": int((matched.state == PREDICTED_STATE).sum()),
                    "mean_iou": float(matched.iou.mean()) if not matched.empty else None,
                    "mean_center_error_px": (
                        float(matched.center_error_px.mean()) if not matched.empty else None
                    ),
                }
            )
    return pd.DataFrame(rows)

"""Class-aware Hungarian matching on official annotated CAM_FRONT keyframes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from oatm.tracking.geometry import center_error, iou
from scipy.optimize import linear_sum_assignment

DETECTOR_TO_EVALUATION_CLASS = {"car": "car", "person": "pedestrian"}
OBSERVED_STATES = {"OBSERVED_STRONG", "OBSERVED_WEAK"}
PREDICTED_STATE = "PREDICTED_HIDDEN"


def validate_ground_truth(ground_truth: pd.DataFrame, keyframe_tokens: set[str]) -> pd.DataFrame:
    required = {
        "scene_token",
        "sample_data_token",
        "instance_token",
        "annotation_token",
        "evaluation_class",
        "visibility_token",
        "x1",
        "y1",
        "x2",
        "y2",
        "center_depth_m",
        "num_lidar_pts",
        "truncation_fraction",
        "projection_status",
    }
    if missing := required - set(ground_truth.columns):
        raise ValueError(f"projected ground truth is missing columns: {sorted(missing)}")
    scoped = ground_truth[
        ground_truth.evaluation_class.isin(["car", "pedestrian"])
        & (ground_truth.projection_status == "accepted")
    ].copy()
    unknown = set(scoped.sample_data_token) - keyframe_tokens
    if unknown:
        raise ValueError(f"projected ground truth contains {len(unknown)} non-keyframe tokens")
    if scoped.duplicated(["sample_data_token", "instance_token"]).any():
        raise ValueError("projected ground truth repeats an instance within a keyframe")
    numeric = scoped[["x1", "y1", "x2", "y2", "center_depth_m", "truncation_fraction"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("projected ground truth contains non-finite geometry")
    if ((scoped.x2 <= scoped.x1) | (scoped.y2 <= scoped.y1)).any():
        raise ValueError("projected ground truth contains non-positive boxes")
    return scoped


def _box(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"]))


def _hungarian_pairs(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    if not predictions or not ground_truth:
        return []
    scores = np.zeros((len(predictions), len(ground_truth)), dtype=float)
    for prediction_index, prediction in enumerate(predictions):
        prediction_box = _box(prediction)
        for truth_index, truth in enumerate(ground_truth):
            scores[prediction_index, truth_index] = iou(prediction_box, _box(truth))
    prediction_indices, truth_indices = linear_sum_assignment(scores, maximize=True)
    return [
        (int(prediction_index), int(truth_index), float(scores[prediction_index, truth_index]))
        for prediction_index, truth_index in zip(prediction_indices, truth_indices)
        if scores[prediction_index, truth_index] >= iou_threshold
    ]


def evaluate_method(
    method_outputs: pd.DataFrame,
    ground_truth: pd.DataFrame,
    keyframes: pd.DataFrame,
    iou_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one GT-centric row per annotation and one row per unmatched output."""
    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    if method_outputs.empty:
        raise ValueError("method_outputs cannot be empty")
    method_names = method_outputs.method_name.unique().tolist()
    if len(method_names) != 1:
        raise ValueError("evaluate_method accepts exactly one method")
    method_name = method_names[0]
    token_to_index = keyframes.set_index("sample_data_token").frame_index.astype(int).to_dict()
    token_to_scene = keyframes.set_index("sample_data_token").scene_token.to_dict()
    keyframe_tokens = set(token_to_index)
    predictions = method_outputs[method_outputs.sample_data_token.isin(keyframe_tokens)].copy()
    predictions["evaluation_class"] = predictions.class_name.map(DETECTOR_TO_EVALUATION_CLASS)
    predictions = predictions[predictions.evaluation_class.notna()]

    predictions_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (token, evaluation_class), rows in predictions.groupby(
        ["sample_data_token", "evaluation_class"], sort=False
    ):
        predictions_by_key[(token, evaluation_class)] = rows.to_dict("records")
    truth_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for (token, evaluation_class), rows in ground_truth.groupby(
        ["sample_data_token", "evaluation_class"], sort=False
    ):
        truth_by_key[(token, evaluation_class)] = rows.to_dict("records")

    truth_rows: list[dict[str, Any]] = []
    false_positive_rows: list[dict[str, Any]] = []
    for token in keyframes.sort_values(["scene_token", "frame_index"]).sample_data_token:
        for evaluation_class in ("car", "pedestrian"):
            frame_predictions = predictions_by_key.get((token, evaluation_class), [])
            frame_truth = truth_by_key.get((token, evaluation_class), [])
            pairs = _hungarian_pairs(frame_predictions, frame_truth, iou_threshold)
            prediction_to_pair = {
                prediction_index: (truth_index, score)
                for prediction_index, truth_index, score in pairs
            }
            truth_to_pair = {
                truth_index: (prediction_index, score)
                for prediction_index, truth_index, score in pairs
            }

            for truth_index, truth in enumerate(frame_truth):
                base = {
                    "method_name": method_name,
                    "scene_token": truth["scene_token"],
                    "sample_data_token": token,
                    "frame_index": token_to_index[token],
                    "instance_token": truth["instance_token"],
                    "annotation_token": truth["annotation_token"],
                    "evaluation_class": evaluation_class,
                    "visibility_token": str(truth["visibility_token"]),
                    "center_depth_m": float(truth["center_depth_m"]),
                    "num_lidar_pts": int(truth["num_lidar_pts"]),
                    "num_radar_pts": int(truth.get("num_radar_pts", 0)),
                    "truncation_fraction": float(truth["truncation_fraction"]),
                    "gt_x1": float(truth["x1"]),
                    "gt_y1": float(truth["y1"]),
                    "gt_x2": float(truth["x2"]),
                    "gt_y2": float(truth["y2"]),
                    "matched": truth_index in truth_to_pair,
                    "track_id": None,
                    "state": None,
                    "evidence_source": None,
                    "iou": None,
                    "center_error_px": None,
                    "pred_x1": None,
                    "pred_y1": None,
                    "pred_x2": None,
                    "pred_y2": None,
                }
                if truth_index in truth_to_pair:
                    prediction_index, score = truth_to_pair[truth_index]
                    prediction = frame_predictions[prediction_index]
                    base.update(
                        {
                            "track_id": int(prediction["track_id"]),
                            "state": prediction["state"],
                            "evidence_source": prediction["evidence_source"],
                            "iou": score,
                            "center_error_px": center_error(_box(prediction), _box(truth)),
                            "pred_x1": float(prediction["x1"]),
                            "pred_y1": float(prediction["y1"]),
                            "pred_x2": float(prediction["x2"]),
                            "pred_y2": float(prediction["y2"]),
                        }
                    )
                truth_rows.append(base)

            for prediction_index, prediction in enumerate(frame_predictions):
                if prediction_index in prediction_to_pair:
                    continue
                false_positive_rows.append(
                    {
                        "method_name": method_name,
                        "scene_token": token_to_scene[token],
                        "sample_data_token": token,
                        "frame_index": token_to_index[token],
                        "track_id": int(prediction["track_id"]),
                        "evaluation_class": evaluation_class,
                        "state": prediction["state"],
                        "evidence_source": prediction["evidence_source"],
                        "x1": float(prediction["x1"]),
                        "y1": float(prediction["y1"]),
                        "x2": float(prediction["x2"]),
                        "y2": float(prediction["y2"]),
                    }
                )
    truth_frame = pd.DataFrame(truth_rows)
    false_positive_frame = pd.DataFrame(false_positive_rows)
    if len(truth_frame) != len(ground_truth):
        raise RuntimeError("evaluation did not produce exactly one row per scoped annotation")
    if int(truth_frame.matched.sum()) + len(false_positive_frame) > len(predictions):
        raise RuntimeError("matching accounting exceeded the number of keyframe predictions")
    return truth_frame, false_positive_frame

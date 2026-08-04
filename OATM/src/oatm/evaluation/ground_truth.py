"""Task 11: loads the privileged, offline-only projected ground truth
(Phase 2 / Task 3) and indexes it two ways -- by frame (for precision/recall
and for resolving which real object a controlled-experiment target
corresponds to) and by (frame, instance) (for center-error/IoU while a track
is hidden). This module is evaluation-only: nothing here is ever fed into an
online tracker's `update()`.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# The detector's raw COCO class names differ from the evaluation_class labels
# used in projected_ground_truth.parquet -- "person" (detector) is
# "pedestrian" (ground truth). Only car/pedestrian have ground truth at all
# (see projection.py's EVALUATION_CLASS_MAP); everything else the detector
# reports has no corresponding ground truth row and must never be matched.
DETECTOR_TO_EVAL_CLASS = {"car": "car", "person": "pedestrian"}


def load_ground_truth(artifacts_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(artifacts_dir / "projected_ground_truth.parquet")


def index_by_frame(gt: pd.DataFrame) -> dict[str, list[dict]]:
    """sample_data_token -> list of {instance_token, evaluation_class, box}."""
    by_frame: dict[str, list[dict]] = {}
    for row in gt.to_dict("records"):
        by_frame.setdefault(row["sample_data_token"], []).append({
            "instance_token": row["instance_token"],
            "evaluation_class": row["evaluation_class"],
            "box": (row["x1"], row["y1"], row["x2"], row["y2"]),
        })
    return by_frame


def index_by_frame_and_instance(gt: pd.DataFrame) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """(sample_data_token, instance_token) -> box. Used to look up a specific
    real object's true projected box at a specific frame while it is hidden."""
    return {
        (row["sample_data_token"], row["instance_token"]): (row["x1"], row["y1"], row["x2"], row["y2"])
        for row in gt.to_dict("records")
    }

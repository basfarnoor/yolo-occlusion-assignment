"""Causal CAM_FRONT tracking stage, deliberately isolated from ground truth."""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from typing import Any, Callable

import cv2
import pandas as pd
import yaml
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter
from oatm.tracking.kalman import KalmanBoxTracker

from lidar_eval.common import DATA_ROOT, PROJECT_ROOT, finite_box

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from oatm_relational import RelationalOATMTracker  # noqa: E402

TrackerFactory = Callable[[], Any]
DETECTOR_CLASSES = {"car", "person"}


def tracker_factories() -> dict[str, TrackerFactory]:
    """Build promoted trackers from the project's frozen relational configuration."""
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "relational.yaml").read_text())
    shared = config["shared"]
    return {
        "bytetrack_b5": lambda: ByteTrackAdapter(**shared, track_buffer=5),
        "bytetrack_b12": lambda: ByteTrackAdapter(**shared, track_buffer=12),
        "relational_complete": lambda: RelationalOATMTracker(
            **shared,
            **config["relational"],
            camera_motion_config=config["camera_motion"],
        ),
    }


def validate_tracking_inputs(frames: pd.DataFrame, detections: pd.DataFrame) -> None:
    required_frames = {
        "scene_token",
        "sample_data_token",
        "timestamp_us",
        "frame_index",
        "is_keyframe",
        "image_path",
    }
    required_detections = {
        "scene_token",
        "sample_data_token",
        "frame_index",
        "detected_class",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    if missing := required_frames - set(frames.columns):
        raise ValueError(f"frame index is missing columns: {sorted(missing)}")
    if missing := required_detections - set(detections.columns):
        raise ValueError(f"detections are missing columns: {sorted(missing)}")
    if frames.sample_data_token.duplicated().any():
        raise ValueError("frame_index must contain one row per CAM_FRONT frame")
    ordered = frames.sort_values(["scene_token", "frame_index"])
    for scene_token, group in ordered.groupby("scene_token", sort=False):
        indices = group.frame_index.astype(int).tolist()
        if indices != list(range(len(indices))):
            raise ValueError(f"scene {scene_token} frame indices are not contiguous and zero-based")
        timestamps = group.timestamp_us.astype(int).tolist()
        if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
            raise ValueError(f"scene {scene_token} timestamps are not strictly increasing")
    frame_keys = frames.set_index("sample_data_token")[["scene_token", "frame_index"]]
    unknown = set(detections.sample_data_token) - set(frame_keys.index)
    if unknown:
        raise ValueError(f"detections refer to {len(unknown)} unknown CAM_FRONT frames")
    if not detections.empty:
        joined = detections.join(frame_keys, on="sample_data_token", rsuffix="_expected")
        inconsistent = (joined.scene_token != joined.scene_token_expected) | (
            joined.frame_index.astype(int) != joined.frame_index_expected.astype(int)
        )
        if inconsistent.any():
            raise ValueError("detection scene/frame metadata are inconsistent with frame_index")
        if not all(finite_box(row) for row in detections.to_dict("records")):
            raise ValueError("detections contain invalid boxes")


def _detections_by_frame(detections: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scoped = detections[detections.detected_class.isin(DETECTOR_CLASSES)]
    for row in scoped.sort_values(["sample_data_token", "detection_id"]).to_dict("records"):
        indexed[row["sample_data_token"]].append(
            {
                "class": row["detected_class"],
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]),
                "y1": float(row["y1"]),
                "x2": float(row["x2"]),
                "y2": float(row["y2"]),
            }
        )
    return dict(indexed)


def run_camera_trackers(
    frames: pd.DataFrame,
    detections: pd.DataFrame,
    methods: list[str],
    run_id: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run trackers scene-by-scene using only causal camera-derived inputs.

    This function intentionally has no ground-truth argument.  The only
    optional pixels are the current CAM_FRONT image for the camera-motion
    ablation; the promoted configuration currently disables that ablation.
    """
    validate_tracking_inputs(frames, detections)
    factories = tracker_factories()
    unknown = set(methods) - set(factories)
    if unknown:
        raise ValueError(f"unknown methods: {sorted(unknown)}; available: {sorted(factories)}")
    observations = _detections_by_frame(detections)
    ordered = frames.sort_values(["scene_token", "frame_index"])
    all_outputs: list[dict[str, Any]] = []
    runtimes: dict[str, float] = {}

    for method_name in methods:
        started = time.perf_counter()
        print(f"tracking: {method_name}", flush=True)
        for scene_token, scene_frames in ordered.groupby("scene_token", sort=False):
            KalmanBoxTracker.reset_id_counter()
            tracker = factories[method_name]()
            for frame in scene_frames.to_dict("records"):
                kwargs: dict[str, Any] = {
                    "timestamp": float(frame["timestamp_us"]) / 1_000_000.0,
                    "scene_token": scene_token,
                    "sample_data_token": frame["sample_data_token"],
                    "method_name": method_name,
                    "run_id": run_id,
                }
                if isinstance(tracker, RelationalOATMTracker) and tracker.enable_camera_compensation:
                    image_path = DATA_ROOT / frame["image_path"]
                    image = cv2.imread(str(image_path))
                    if image is None:
                        raise FileNotFoundError(f"could not read CAM_FRONT image {image_path}")
                    kwargs["frame"] = image
                rows = tracker.update(observations.get(frame["sample_data_token"], []), **kwargs)
                for row in rows:
                    dumped = row.model_dump()
                    if dumped["sample_data_token"] != frame["sample_data_token"]:
                        raise RuntimeError("tracker emitted output for a future or different frame")
                    all_outputs.append(dumped)
        runtimes[method_name] = time.perf_counter() - started
        print(f"tracking: {method_name} finished in {runtimes[method_name]:.2f}s", flush=True)

    outputs = pd.DataFrame(all_outputs)
    if outputs.empty:
        raise RuntimeError("trackers produced no outputs")
    duplicates = outputs.duplicated(["method_name", "scene_token", "frame_index", "track_id"])
    if duplicates.any():
        raise RuntimeError(f"trackers emitted {int(duplicates.sum())} duplicate identity/frame rows")
    if not all(finite_box(row) for row in outputs.to_dict("records")):
        raise RuntimeError("tracker outputs contain invalid boxes")
    return outputs, runtimes

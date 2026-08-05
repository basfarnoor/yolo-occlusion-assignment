#!/usr/bin/env python3
"""Held-manifest natural-event comparison on freshly generated detections."""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "OATM_UPDATED" / "src"))

from oatm.evaluation.event_metrics import EventMetricsInputs, compute_event_metrics  # noqa: E402
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402
from oatm_updated import SelectiveOATMTracker  # noqa: E402

from oatm_relational import RelationalOATMTracker  # noqa: E402

CLASS_MAP = {"car": "car", "pedestrian": "person"}
RECOVERY_HORIZON = 12


def factories(config: dict) -> dict:
    shared = config["shared"]
    selective = yaml.safe_load((REPO / "OATM_UPDATED" / "configs" / "selective.yaml").read_text())
    return {
        "bytetrack_b5": lambda: ByteTrackAdapter(**shared, track_buffer=5),
        "bytetrack_b12": lambda: ByteTrackAdapter(**shared, track_buffer=12),
        "selective_oatm": lambda: SelectiveOATMTracker(**shared, **selective["selective_oatm"]),
        "relational_camera": lambda: RelationalOATMTracker(
            **shared,
            **dict(config["relational"], enable_camera_compensation=True),
            camera_motion_config=config["camera_motion"],
        ),
        "relational_complete": lambda: RelationalOATMTracker(
            **shared, **config["relational"], camera_motion_config=config["camera_motion"]
        ),
    }


def index_rows(rows: list[dict]) -> dict[int, list[dict]]:
    indexed: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        indexed[int(row["frame_index"])].append(row)
    return dict(indexed)


def run_method(method_name: str, factory, frames_by_scene: dict, detections_by_frame: dict, run_id: str):
    outputs = {}
    started = time.perf_counter()
    for scene_token, frames in frames_by_scene.items():
        KalmanBoxTracker.reset_id_counter()
        tracker = factory()
        scene_rows = []
        for frame in frames:
            detections = detections_by_frame.get(frame["sample_data_token"], [])
            kwargs = {
                "timestamp": frame["timestamp_us"] / 1_000_000.0,
                "scene_token": scene_token,
                "sample_data_token": frame["sample_data_token"],
                "method_name": method_name,
                "run_id": run_id,
            }
            if isinstance(tracker, RelationalOATMTracker) and tracker.enable_camera_compensation:
                kwargs["frame"] = cv2.imread(str(REPO / "data" / "nuscenes" / frame["image_path"]))
            scene_rows.extend(row.model_dump() for row in tracker.update(detections, **kwargs))
        outputs[scene_token] = index_rows(scene_rows)
    return outputs, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", help="Subset of method names to run")
    parser.add_argument("--output-stem", default="natural", help="Prefix for stored result files")
    args = parser.parse_args()

    started = time.perf_counter()
    run_id = uuid.uuid4().hex[:12]
    config = yaml.safe_load((ROOT / "configs" / "relational.yaml").read_text())
    available_factories = factories(config)
    selected_names = args.methods or list(available_factories)
    unknown = set(selected_names) - set(available_factories)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}")
    artifacts = ROOT / "artifacts"
    frame_index = pd.read_parquet(artifacts / "frame_index.parquet")
    detections = pd.read_parquet(artifacts / "detections.parquet")
    ground_truth = pd.read_parquet(artifacts / "projected_ground_truth.parquet")
    frames_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in frame_index.sort_values(["scene_token", "frame_index"]).to_dict("records"):
        frames_by_scene[row["scene_token"]].append(row)
    detections_by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in detections.to_dict("records"):
        detections_by_frame[row["sample_data_token"]].append(
            {
                "class": row["detected_class"],
                "confidence": row["confidence"],
                "x1": row["x1"],
                "y1": row["y1"],
                "x2": row["x2"],
                "y2": row["y2"],
            }
        )
    frame_token = {
        (row["scene_token"], int(row["frame_index"])): row["sample_data_token"]
        for row in frame_index.to_dict("records")
    }
    max_frame = frame_index.groupby("scene_token").frame_index.max().to_dict()
    gt_by_instance = {
        (row["sample_data_token"], row["instance_token"]): (row["x1"], row["y1"], row["x2"], row["y2"])
        for row in ground_truth.to_dict("records")
    }
    events = pd.read_csv(REPO / "OATM" / "results" / "natural_event_manifest.csv")
    events = events[events.review_status == "accepted"].to_dict("records")

    method_outputs = {}
    runtimes = {}
    for method_name in selected_names:
        factory = available_factories[method_name]
        print(f"Running {method_name}...")
        method_outputs[method_name], runtimes[method_name] = run_method(
            method_name, factory, dict(frames_by_scene), dict(detections_by_frame), run_id
        )

    metric_rows = []
    for event in events:
        scene = event["scene_token"]
        instance = event["instance_token"]
        pre = int(event["pre_frame_index"])
        hidden = list(range(int(event["start_frame_index"]), int(event["end_frame_index"]) + 1))
        detector_class = CLASS_MAP[event["evaluation_class"]]
        reference = gt_by_instance.get((frame_token[(scene, pre)], instance))
        gt_by_frame = {}
        final = min(max(hidden) + RECOVERY_HORIZON, int(max_frame[scene]))
        for frame_index_value in range(pre, final + 1):
            token = frame_token.get((scene, frame_index_value))
            candidate = gt_by_instance.get((token, instance))
            if candidate is not None:
                gt_by_frame[frame_index_value] = candidate
        for method_name in selected_names:
            if reference is None:
                row = {"event_id": event["event_id"], "method_name": method_name, "target_linked": False}
            else:
                result = compute_event_metrics(
                    EventMetricsInputs(
                        event_id=event["event_id"],
                        method_name=method_name,
                        reference_box=reference,
                        reference_class=detector_class,
                        pre_frame_index=pre,
                        hidden_frame_indices=hidden,
                        recovery_search_frame_indices=list(range(max(hidden) + 1, final + 1)),
                        outputs_by_frame=method_outputs[method_name][scene],
                        gt_box_by_frame=gt_by_frame,
                    )
                )
                row = vars(result)
            row["split"] = event["split"]
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ROOT / "results" / f"{args.output_stem}_event_metrics.csv", index=False)
    linked = metrics[metrics.target_linked == True]  # noqa: E712
    summary_rows = []
    for method_name, group in linked.groupby("method_name"):
        summary_rows.append(
            {
                "method": method_name,
                "linked_events": len(group),
                "mean_hidden_coverage": group.hidden_frame_coverage.mean(),
                "fully_bridged_rate": group.fully_bridged.mean(),
                "same_id_recoveries": int((group.recovery_status == "same_id").sum()),
                "new_id_recoveries": int((group.recovery_status == "new_id").sum()),
                "not_recovered": int((group.recovery_status == "not_recovered").sum()),
                "mean_center_error_px": group.mean_center_error_px.mean(),
                "runtime_seconds": runtimes[method_name],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(ROOT / "results" / f"{args.output_stem}_summary.csv", index=False)
    report = f"""# Relational OATM Natural-Event Pilot

Run ID: `{run_id}`. Fresh detector artifacts were used. The reviewed OATM
manifest contains {len(events)} accepted events; conclusions apply only to
events that each method could link at the pre-occlusion reference frame.

{summary.to_markdown(index=False, floatfmt=".3f")}

This is a nuScenes-mini pilot, not a statistically powered superiority claim.
LiDAR-supported projection is privileged evaluation evidence only.
"""
    (ROOT / "results" / f"{args.output_stem}_report.md").write_text(report)
    metadata = {
        "run_id": run_id,
        "accepted_events": len(events),
        "selected_methods": selected_names,
        "output_stem": args.output_stem,
        "runtimes": runtimes,
        "runtime_seconds": time.perf_counter() - started,
        "config": config,
    }
    (ROOT / "results" / f"{args.output_stem}_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(report)


if __name__ == "__main__":
    main()

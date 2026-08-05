#!/usr/bin/env python3
"""Read-only nuScenes-mini indexing and privileged projection preparation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from oatm.dataset.nuscenes_index import build_frame_index  # noqa: E402
from oatm.dataset.projection import map_evaluation_class, project_annotation  # noqa: E402
from oatm.records import ProjectedGroundTruthRecord  # noqa: E402

DATA_ROOT = REPO / "data" / "nuscenes"
VERSION = "v1.0-mini"
IMAGE_SIZE = (1600, 900)


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_ground_truth() -> tuple[list[ProjectedGroundTruthRecord], list[dict]]:
    metadata = DATA_ROOT / VERSION
    sample_data = {row["token"]: row for row in load_json(metadata / "sample_data.json")}
    samples = {row["token"]: row for row in load_json(metadata / "sample.json")}
    ego_poses = {row["token"]: row for row in load_json(metadata / "ego_pose.json")}
    calibrated = {row["token"]: row for row in load_json(metadata / "calibrated_sensor.json")}
    sensors = load_json(metadata / "sensor.json")
    instances = {row["token"]: row for row in load_json(metadata / "instance.json")}
    categories = {row["token"]: row for row in load_json(metadata / "category.json")}
    annotations = load_json(metadata / "sample_annotation.json")
    camera_sensor = next(row["token"] for row in sensors if row["channel"] == "CAM_FRONT")
    by_sample: dict[str, list[dict]] = defaultdict(list)
    for annotation in annotations:
        by_sample[annotation["sample_token"]].append(annotation)
    keyframes = [
        row
        for row in sample_data.values()
        if row["is_key_frame"] and calibrated[row["calibrated_sensor_token"]]["sensor_token"] == camera_sensor
    ]
    accepted = []
    rejected = []
    for sample_record in keyframes:
        sample = samples[sample_record["sample_token"]]
        pose = ego_poses[sample_record["ego_pose_token"]]
        calibration = calibrated[sample_record["calibrated_sensor_token"]]
        for annotation in by_sample.get(sample["token"], []):
            instance = instances[annotation["instance_token"]]
            category = categories[instance["category_token"]]["name"]
            projected = project_annotation(annotation, pose, calibration, IMAGE_SIZE)
            if projected.projection_status != "accepted":
                rejected.append(
                    {
                        "scene_token": sample["scene_token"],
                        "sample_data_token": sample_record["token"],
                        "instance_token": annotation["instance_token"],
                        "reason": projected.projection_status,
                    }
                )
                continue
            accepted.append(
                ProjectedGroundTruthRecord(
                    scene_token=sample["scene_token"],
                    sample_data_token=sample_record["token"],
                    instance_token=annotation["instance_token"],
                    annotation_token=annotation["token"],
                    original_category=category,
                    evaluation_class=map_evaluation_class(category) or "other",
                    visibility_token=annotation["visibility_token"],
                    x1=projected.x1,
                    y1=projected.y1,
                    x2=projected.x2,
                    y2=projected.y2,
                    center_depth_m=projected.center_depth_m,
                    num_lidar_pts=annotation.get("num_lidar_pts", 0),
                    num_radar_pts=annotation.get("num_radar_pts", 0),
                    truncation_fraction=projected.truncation_fraction,
                    projection_status="accepted",
                )
            )
    return accepted, rejected


def main() -> None:
    started = time.perf_counter()
    artifacts = ROOT / "artifacts"
    results = ROOT / "results"
    artifacts.mkdir(exist_ok=True)
    records, audit = build_frame_index(DATA_ROOT, VERSION)
    gate = {
        "scenes": audit.n_scenes == 10,
        "keyframes": audit.n_keyframes == 404,
        "camera_records": audit.n_cam_front_records == 2342,
        "no_missing_images": all(not scene.missing_image_files for scene in audit.scene_results),
        "monotonic": all(scene.strictly_increasing_timestamps for scene in audit.scene_results),
    }
    if not all(gate.values()):
        raise RuntimeError(f"nuScenes-mini audit failed: {gate}")
    pd.DataFrame([record.model_dump() for record in records]).to_parquet(
        artifacts / "frame_index.parquet", index=False
    )
    projected, rejected = project_ground_truth()
    pd.DataFrame([record.model_dump() for record in projected]).to_parquet(
        artifacts / "projected_ground_truth.parquet", index=False
    )
    (artifacts / "projection_rejections.json").write_text(json.dumps(rejected))
    manifests = [
        ROOT.parent / "OATM" / "results" / name
        for name in ("natural_event_manifest.csv", "controlled_event_manifest.csv")
    ]
    metadata = {
        "dataset_version": VERSION,
        "gate": gate,
        "audit": asdict(audit),
        "projected_rows": len(projected),
        "projection_rejections": len(rejected),
        "source_manifest_hashes": {path.name: sha256(path) for path in manifests},
        "python": platform.python_version(),
        "runtime_seconds": time.perf_counter() - started,
    }
    (artifacts / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    report = f"""# nuScenes Mini Preparation

The dataset was read only. Privileged projected annotations are evaluation
evidence and are never supplied to an online tracker.

- Quality gate: **PASSED**
- Scenes: {audit.n_scenes}
- CAM_FRONT frames: {audit.n_cam_front_records}
- Keyframes: {audit.n_keyframes}
- Accepted projected annotations: {len(projected)}
- Rejected projections: {len(rejected)}
- Source manifest hashes: `{metadata["source_manifest_hashes"]}`
"""
    (results / "nuscenes_preparation.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()

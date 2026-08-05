#!/usr/bin/env python3
"""Run one frozen detector over the prepared relational frame index."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
from oatm.records import DetectorObservationRecord
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DATA_ROOT = REPO / "data" / "nuscenes"
MODEL_NAME = "yolo26n.pt"
CONFIDENCE_FLOOR = 0.05


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    artifacts = ROOT / "artifacts"
    results = ROOT / "results"
    weights = REPO / MODEL_NAME
    if not weights.is_file():
        raise FileNotFoundError(f"Missing {weights}; obtain the frozen pretrained weight before this stage.")
    frames = pd.read_parquet(artifacts / "frame_index.parquet").sort_values(["scene_token", "frame_index"])
    model = YOLO(str(weights))
    weights_hash = file_hash(weights)
    records = []
    started = time.perf_counter()
    for sequence_index, frame in enumerate(frames.to_dict("records")):
        image_path = DATA_ROOT / frame["image_path"]
        inference_started = time.perf_counter()
        result = model.predict(
            str(image_path), imgsz=640, conf=CONFIDENCE_FLOOR, device="cpu", verbose=False
        )[0]
        inference_ms = (time.perf_counter() - inference_started) * 1000
        if result.boxes is not None:
            for detection_id, predicted in enumerate(result.boxes):
                x1, y1, x2, y2 = [float(value) for value in predicted.xyxy[0].tolist()]
                class_id = int(predicted.cls.item())
                records.append(
                    DetectorObservationRecord(
                        scene_token=frame["scene_token"],
                        sample_data_token=frame["sample_data_token"],
                        frame_index=frame["frame_index"],
                        detection_id=detection_id,
                        model_name=MODEL_NAME,
                        model_weights_hash=weights_hash,
                        detected_class=result.names.get(class_id, str(class_id)),
                        confidence=float(predicted.conf.item()),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        inference_time_ms=inference_ms,
                        cache_key=f"{weights_hash}:{frame['sample_data_token']}:640:{CONFIDENCE_FLOOR}",
                    )
                )
        if (sequence_index + 1) % 250 == 0:
            print(f"{sequence_index + 1}/{len(frames)} frames")
    pd.DataFrame([record.model_dump() for record in records]).to_parquet(
        artifacts / "detections.parquet", index=False
    )
    metadata = {
        "model": MODEL_NAME,
        "weights_hash": weights_hash,
        "confidence_floor": CONFIDENCE_FLOOR,
        "device": "cpu",
        "frames": len(frames),
        "detections": len(records),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
    }
    (artifacts / "detector_metadata.json").write_text(json.dumps(metadata, indent=2))
    (results / "detector_audit.md").write_text(
        "# Detector Audit\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in metadata.items()) + "\n"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

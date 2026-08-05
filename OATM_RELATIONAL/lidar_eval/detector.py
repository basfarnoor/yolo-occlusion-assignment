"""Validate or regenerate the frozen CAM_FRONT detector cache.

This is the only GPU-accelerated stage.  It reads camera images and writes raw
camera detections.  It does not open projected ground truth or any LiDAR data.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from oatm.records import DetectorObservationRecord
from ultralytics import YOLO

from lidar_eval.common import (
    ARTIFACTS_ROOT,
    DATA_ROOT,
    PROJECT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    finite_box,
    load_config,
    sha256,
)

DETECTION_COLUMNS = list(DetectorObservationRecord.model_fields)


def resolve_model_path(config: dict[str, Any]) -> Path:
    configured = Path(config["detector"]["model_path"])
    return configured if configured.is_absolute() else (PROJECT_ROOT / configured).resolve()


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if requested not in {"cpu", "0", "cuda", "cuda:0"}:
        raise ValueError("device must be one of: auto, cpu, 0, cuda, cuda:0")
    if requested != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but torch cannot access CUDA")
    return requested


def validate_detector_cache(
    frames: pd.DataFrame,
    detections_path: Path,
    metadata_path: Path,
    model_path: Path,
    image_size: int,
    confidence_floor: float,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    """Validate cache provenance and every stored row without requiring detections per frame."""
    reasons: list[str] = []
    metadata: dict[str, Any] | None = None
    if not detections_path.is_file():
        reasons.append(f"missing {detections_path}")
    if not metadata_path.is_file():
        reasons.append(f"missing {metadata_path}")
    if not model_path.is_file():
        reasons.append(f"missing frozen detector weights {model_path}")
    if reasons:
        return False, reasons, metadata

    try:
        import json

        metadata = json.loads(metadata_path.read_text())
        detections = pd.read_parquet(detections_path)
    except Exception as error:  # corrupt cache should trigger a controlled rebuild
        return False, [f"cache could not be read: {error}"], metadata

    expected_hash = sha256(model_path)
    if metadata.get("weights_hash") != expected_hash:
        reasons.append("detector weight hash differs from cache metadata")
    if int(metadata.get("frames", -1)) != len(frames):
        reasons.append("cached frame count differs from prepared CAM_FRONT frame index")
    if float(metadata.get("confidence_floor", -1.0)) != float(confidence_floor):
        reasons.append("confidence floor differs from cache metadata")
    if "image_size" in metadata and int(metadata["image_size"]) != image_size:
        reasons.append("detector image size differs from cache metadata")
    if int(metadata.get("detections", -1)) != len(detections):
        reasons.append("cached detection row count differs from cache metadata")

    missing_columns = set(DETECTION_COLUMNS) - set(detections.columns)
    if missing_columns:
        reasons.append(f"cache is missing columns: {sorted(missing_columns)}")
        return False, reasons, metadata
    frame_lookup = frames.set_index("sample_data_token")[["scene_token", "frame_index"]]
    if detections.sample_data_token.duplicated().all() and detections.empty:
        reasons.append("detector cache has no rows")
    unknown_tokens = set(detections.sample_data_token) - set(frame_lookup.index)
    if unknown_tokens:
        reasons.append(f"cache contains {len(unknown_tokens)} unknown frame tokens")
    if not unknown_tokens and not detections.empty:
        joined = detections.join(frame_lookup, on="sample_data_token", rsuffix="_expected")
        bad_scene = joined.scene_token != joined.scene_token_expected
        bad_index = joined.frame_index.astype(int) != joined.frame_index_expected.astype(int)
        if bool((bad_scene | bad_index).any()):
            reasons.append("cached detections disagree with frame scene/index metadata")
        if not all(finite_box(row) for row in detections.to_dict("records")):
            reasons.append("cache contains non-finite or non-positive boxes")
        if bool(((detections.confidence < 0.0) | (detections.confidence > 1.0)).any()):
            reasons.append("cache contains confidence outside [0, 1]")
        if set(detections.model_weights_hash) != {expected_hash}:
            reasons.append("row-level detector weight hashes are inconsistent")
        expected_suffix = f":{image_size}:{confidence_floor}"
        if not detections.cache_key.astype(str).str.endswith(expected_suffix).all():
            reasons.append("row-level cache keys do not match image size/confidence")
    return not reasons, reasons, metadata


def generate_detector_cache(
    frames: pd.DataFrame,
    model_path: Path,
    detections_path: Path,
    metadata_path: Path,
    image_size: int,
    confidence_floor: float,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    """Run the frozen detector in bounded batches and atomically replace the cache."""
    device = resolve_device(device)
    weights_hash = sha256(model_path)
    model = YOLO(str(model_path))
    ordered = frames.sort_values(["scene_token", "frame_index"]).reset_index(drop=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    for start in range(0, len(ordered), batch_size):
        batch = ordered.iloc[start : start + batch_size].to_dict("records")
        image_paths = [str(DATA_ROOT / row["image_path"]) for row in batch]
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"missing CAM_FRONT image: {missing[0]}")
        inference_started = time.perf_counter()
        predictions = model.predict(
            image_paths,
            imgsz=image_size,
            conf=confidence_floor,
            device=device,
            batch=batch_size,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - inference_started) * 1000.0
        if len(predictions) != len(batch):
            raise RuntimeError("detector result count does not match submitted CAM_FRONT batch")
        per_image_ms = elapsed_ms / len(batch)
        for frame, result in zip(batch, predictions):
            if result.boxes is None:
                continue
            for detection_id, predicted in enumerate(result.boxes):
                x1, y1, x2, y2 = (float(value) for value in predicted.xyxy[0].tolist())
                class_id = int(predicted.cls.item())
                record = DetectorObservationRecord(
                    scene_token=frame["scene_token"],
                    sample_data_token=frame["sample_data_token"],
                    frame_index=int(frame["frame_index"]),
                    detection_id=detection_id,
                    model_name=model_path.name,
                    model_weights_hash=weights_hash,
                    detected_class=result.names.get(class_id, str(class_id)),
                    confidence=float(predicted.conf.item()),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    inference_time_ms=per_image_ms,
                    cache_key=(
                        f"{weights_hash}:{frame['sample_data_token']}:{image_size}:{confidence_floor}"
                    ),
                )
                records.append(record.model_dump())
        completed = min(start + batch_size, len(ordered))
        if completed == len(ordered) or completed % 256 == 0:
            print(f"detector: {completed}/{len(ordered)} CAM_FRONT frames", flush=True)

    detections = pd.DataFrame(records, columns=DETECTION_COLUMNS)
    runtime = time.perf_counter() - started
    metadata = {
        "schema_version": 2,
        "model": model_path.name,
        "weights_hash": weights_hash,
        "image_size": image_size,
        "confidence_floor": confidence_floor,
        "device": device,
        "batch_size": batch_size,
        "frames": len(ordered),
        "detections": len(detections),
        "runtime_seconds": runtime,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_write_parquet(detections, detections_path)
    atomic_write_json(metadata_path, metadata)
    return metadata


def ensure_detector_cache(
    config: dict[str, Any],
    frames: pd.DataFrame,
    force: bool = False,
    device_override: str | None = None,
) -> dict[str, Any]:
    detector_config = config["detector"]
    model_path = resolve_model_path(config)
    detections_path = ARTIFACTS_ROOT / "detections.parquet"
    metadata_path = ARTIFACTS_ROOT / "detector_metadata.json"
    valid, reasons, metadata = validate_detector_cache(
        frames,
        detections_path,
        metadata_path,
        model_path,
        int(detector_config["image_size"]),
        float(detector_config["confidence_floor"]),
    )
    if valid and not force:
        print("detector: validated and reusing existing camera-only cache", flush=True)
        return {"action": "reused", "validation_reasons": [], "metadata": metadata}
    if reasons:
        print("detector cache rebuild reasons:", *[f"\n- {reason}" for reason in reasons], flush=True)
    generated = generate_detector_cache(
        frames=frames,
        model_path=model_path,
        detections_path=detections_path,
        metadata_path=metadata_path,
        image_size=int(detector_config["image_size"]),
        confidence_floor=float(detector_config["confidence_floor"]),
        batch_size=int(detector_config["batch_size"]),
        device=device_override or str(detector_config["device"]),
    )
    valid, validation_reasons, _ = validate_detector_cache(
        frames,
        detections_path,
        metadata_path,
        model_path,
        int(detector_config["image_size"]),
        float(detector_config["confidence_floor"]),
    )
    if not valid:
        raise RuntimeError(f"new detector cache failed validation: {validation_reasons}")
    return {"action": "regenerated", "validation_reasons": reasons, "metadata": generated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--force", action="store_true", help="Regenerate even if the cache is valid")
    parser.add_argument("--device", help="Override config device: auto, cpu, 0, cuda, or cuda:0")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    frames_path = ARTIFACTS_ROOT / "frame_index.parquet"
    if not frames_path.is_file():
        raise FileNotFoundError("run scripts/prepare_nuscenes.py before detector generation")
    frames = pd.read_parquet(frames_path)
    audit = ensure_detector_cache(config, frames, force=args.force, device_override=args.device)
    print(audit)


if __name__ == "__main__":
    main()

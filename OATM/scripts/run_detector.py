"""Phase 4 (Task 5): runs the frozen YOLO detector once over every CAM_FRONT
frame in the frame index (all 10 scenes, 2,342 frames), prediction only,
caches every result, and writes DetectorObservationRecord rows. Keeps
detections down to a low, documented confidence floor so weak boxes remain
available to every later baseline -- matching for Assignments 3-4's finding
that useful low-confidence evidence exists below a naive single threshold.

Reproduction: `.venv/Scripts/python scripts/run_detector.py` from OATM/.
"""
from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import pandas as pd
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.detection.cache import DetectionCache, cache_key, sha256_of_file  # noqa: E402
from oatm.records import DetectorObservationRecord  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "yolo26n.pt"
CONFIDENCE_FLOOR = 0.05  # documented low floor -- keeps weak detections for every baseline
DEFAULT_IMGSZ = 640
FALLBACK_IMGSZ = 480
SLOW_THRESHOLD_S = 2.5
DEVICE = "cpu"


def package_versions() -> dict:
    return {"python": platform.python_version(), "ultralytics": ultralytics.__version__}


def run_one(model: YOLO, image_path: Path, imgsz: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    preds = model.predict(source=str(image_path), imgsz=imgsz, conf=CONFIDENCE_FLOOR,
                            device=DEVICE, verbose=False)
    inference_ms = (time.perf_counter() - t0) * 1000
    result = preds[0]
    dets = []
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            cls_name = result.names.get(cls_id, str(cls_id))
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            dets.append({"class": cls_name, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return dets, inference_ms


def main() -> int:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)

    frame_index = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")
    frames = frame_index.sort_values(["scene_token", "frame_index"]).to_dict("records")
    print(f"Loaded {len(frames)} frames.")

    weights_path = repo_root / MODEL_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{weights_path} not found -- this project reuses the same pretrained weights "
            f"as the earlier assignments; place {MODEL_NAME} at the repo root."
        )
    model = YOLO(str(weights_path))
    weights_hash = sha256_of_file(weights_path)
    pkg_versions = package_versions()

    bench_frames = frames[: min(5, len(frames))]
    bench_times = []
    for f in bench_frames:
        _, ms = run_one(model, config.data_root / f["image_path"], DEFAULT_IMGSZ)
        bench_times.append(ms)
    avg_bench_s = (sum(bench_times) / len(bench_times)) / 1000
    imgsz = DEFAULT_IMGSZ
    imgsz_note = (f"Benchmark average {avg_bench_s:.2f}s/frame at imgsz={DEFAULT_IMGSZ} -- "
                  f"within budget, keeping imgsz={DEFAULT_IMGSZ}.")
    if avg_bench_s > SLOW_THRESHOLD_S:
        imgsz = FALLBACK_IMGSZ
        imgsz_note = (f"Benchmark average {avg_bench_s:.2f}s/frame at imgsz={DEFAULT_IMGSZ} exceeded "
                      f"the {SLOW_THRESHOLD_S}s budget -- reduced to imgsz={FALLBACK_IMGSZ}.")
    print(imgsz_note)

    cache_path = config.artifacts_dir / ".detection_cache.json"
    cache = DetectionCache(cache_path)

    records: list[DetectorObservationRecord] = []
    n_hits, n_misses = 0, 0
    t_start = time.time()

    for f in frames:
        img_path = config.data_root / f["image_path"]
        img_hash = sha256_of_file(img_path)
        key = cache_key(img_hash, MODEL_NAME, weights_hash, imgsz, CONFIDENCE_FLOOR, pkg_versions)

        cached = cache.get(key)
        if cached is not None:
            dets, inference_ms = cached["detections"], cached["inference_time_ms"]
            n_hits += 1
        else:
            dets, inference_ms = run_one(model, img_path, imgsz)
            cache.set(key, dets, inference_ms)
            n_misses += 1

        for detection_id, d in enumerate(dets):
            records.append(DetectorObservationRecord(
                scene_token=f["scene_token"], sample_data_token=f["sample_data_token"],
                frame_index=f["frame_index"], detection_id=detection_id,
                model_name=MODEL_NAME, model_weights_hash=weights_hash,
                detected_class=d["class"], confidence=round(d["confidence"], 5),
                x1=round(d["x1"], 2), y1=round(d["y1"], 2), x2=round(d["x2"], 2), y2=round(d["y2"], 2),
                inference_time_ms=round(inference_ms, 3), cache_key=key,
            ))

        if (n_hits + n_misses) % 500 == 0:
            print(f"  ...{n_hits + n_misses}/{len(frames)} frames processed "
                  f"({n_hits} cache hits, {n_misses} misses)")

    cache.save()
    elapsed_s = time.time() - t_start

    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(config.artifacts_dir / "detections.parquet", index=False)

    lines = ["# Detector Audit\n\n"]
    lines.append(f"Model: `{MODEL_NAME}` (pretrained, prediction only -- no training/fine-tuning). "
                 f"Device: `{DEVICE}`.\n")
    lines.append(f"Confidence floor: **{CONFIDENCE_FLOOR}**. {imgsz_note}\n\n")
    lines.append(f"- Frames processed: {len(frames)} (all 10 mini scenes)\n")
    lines.append(f"- Cache hits: {n_hits}, misses (real inference): {n_misses}\n")
    lines.append(f"- Total raw detections: {len(records)}\n")
    lines.append(f"- Wall-clock time this run: {elapsed_s:.1f}s\n")
    lines.append(f"- Package versions: {pkg_versions}\n\n")

    conf_by_class = df.groupby("detected_class")["confidence"].agg(["count", "mean", "min", "max"])
    lines.append("## Confidence by class (all detections, all classes)\n\n")
    lines.append("| Class | n | mean conf | min | max |\n|---|---:|---:|---:|---:|\n")
    for cls, row in conf_by_class.sort_values("count", ascending=False).iterrows():
        lines.append(
            f"| {cls} | {int(row['count'])} | {row['mean']:.3f} | {row['min']:.3f} | {row['max']:.3f} |\n"
        )

    det_path = config.artifacts_dir / "detections.parquet"
    lines.append(
        f"\nLocal-only artifacts (git-ignored, regenerable with `python scripts/run_detector.py`):\n\n"
        f"- `OATM/artifacts/detections.parquet` -- {len(records)} rows, "
        f"schema: `oatm.records.DetectorObservationRecord`, {det_path.stat().st_size / 1024:.0f} KB.\n"
        f"- `OATM/artifacts/.detection_cache.json` -- {len(cache)} cached entries "
        f"(image+model+weights+imgsz+floor+package-version keyed).\n"
    )

    with open(config.results_dir / "detector_audit.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote detections.parquet ({len(records)} rows). Cache: {n_hits} hits, {n_misses} misses.")
    print(f"Elapsed: {elapsed_s:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

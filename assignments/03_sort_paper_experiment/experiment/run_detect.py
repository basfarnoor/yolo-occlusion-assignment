"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Task 3: run YOLO once per unique configuration over every clip frame,
caching results so later tracking experiments never re-run the detector
unnecessarily. Prediction only -- no training or fine-tuning.
"""
from __future__ import annotations

import csv
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from detector_cache import DetectionCache, cache_key, sha256_of_file  # noqa: E402

import ultralytics
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results"
CACHE_PATH = OUT_ROOT / ".detection_cache.json"
MANIFEST_PATH = OUT_ROOT / "clip_manifest.csv"
DETECTIONS_CSV = OUT_ROOT / "detections.csv"
MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.05
DEFAULT_IMGSZ = 640
FALLBACK_IMGSZ = 480
SLOW_THRESHOLD_S = 2.5
DEVICE = "cpu"


def load_frames() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def package_versions() -> dict:
    return {
        "ultralytics": ultralytics.__version__,
        "python": platform.python_version(),
    }


def get_weights_hash(weights_path: Path) -> str:
    return sha256_of_file(weights_path) if weights_path.is_file() else "unresolved-will-download"


def run_one(model: YOLO, image_path: Path, imgsz: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    preds = model.predict(source=str(image_path), imgsz=imgsz, conf=CONF_THRESHOLD,
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


def main() -> None:
    frames = load_frames()
    print(f"Loaded {len(frames)} frames across clips.")

    weights_path = PROJECT_ROOT / MODEL_NAME
    weights_existed = weights_path.is_file()
    if not weights_existed:
        print(f"'{MODEL_NAME}' not found locally -- Ultralytics will download the small "
              f"pretrained nano weights once, then reuse the local copy.")
    model = YOLO(MODEL_NAME)
    weights_hash = get_weights_hash(weights_path)
    pkg_versions = package_versions()

    # Quick 5-frame benchmark at the default image size.
    bench_frames = frames[: min(5, len(frames))]
    bench_times = []
    for f in bench_frames:
        _, ms = run_one(model, PROJECT_ROOT / f["experiment_image_path"], DEFAULT_IMGSZ)
        bench_times.append(ms)
    avg_bench_s = (sum(bench_times) / len(bench_times)) / 1000
    imgsz = DEFAULT_IMGSZ
    imgsz_note = f"Benchmark average {avg_bench_s:.2f}s/frame at imgsz={DEFAULT_IMGSZ} -- within budget, keeping imgsz={DEFAULT_IMGSZ}."
    if avg_bench_s > SLOW_THRESHOLD_S:
        imgsz = FALLBACK_IMGSZ
        imgsz_note = (f"Benchmark average {avg_bench_s:.2f}s/frame at imgsz={DEFAULT_IMGSZ} exceeded "
                       f"the {SLOW_THRESHOLD_S}s budget -- reduced to imgsz={FALLBACK_IMGSZ}.")
    print(imgsz_note)

    cache = DetectionCache(CACHE_PATH)
    detection_rows = []
    n_cache_hits = 0
    n_cache_misses = 0

    for f in frames:
        img_path = PROJECT_ROOT / f["experiment_image_path"]
        img_hash = sha256_of_file(img_path)
        key = cache_key(img_hash, MODEL_NAME, weights_hash, imgsz, CONF_THRESHOLD, pkg_versions)

        cached = cache.get(key)
        if cached is not None:
            dets, inference_ms = cached["detections"], cached["inference_time_ms"]
            n_cache_hits += 1
        else:
            dets, inference_ms = run_one(model, img_path, imgsz)
            cache.set(key, dets, inference_ms)
            n_cache_misses += 1

        for d in dets:
            detection_rows.append({
                "clip": f["clip_name"],
                "frame_number": f["frame_number"],
                "timestamp": f["timestamp"],
                "class": d["class"],
                "confidence": round(d["confidence"], 5),
                "x1": round(d["x1"], 2), "y1": round(d["y1"], 2),
                "x2": round(d["x2"], 2), "y2": round(d["y2"], 2),
                "inference_time_ms": round(inference_ms, 2),
            })

    cache.save()

    with open(DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "frame_number", "timestamp", "class", "confidence",
            "x1", "y1", "x2", "y2", "inference_time_ms",
        ])
        writer.writeheader()
        writer.writerows(detection_rows)

    print(f"Wrote {DETECTIONS_CSV} ({len(detection_rows)} detections across {len(frames)} frames).")
    print(f"Cache: {n_cache_hits} hits, {n_cache_misses} misses, {len(cache)} total entries.")
    print(f"Image size used: {imgsz}")


if __name__ == "__main__":
    main()

"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 5: run YOLO once per unique configuration over every clip frame, caching
results so later experiments never re-run the detector unnecessarily.
Prediction only -- no training or fine-tuning. Uses a lower confidence floor
than a plain detector run would, because ByteTrack's whole premise depends on
low-score boxes still being available to the second association stage.

Before reusing Assignment 3's detection cache, this script checks whether it
even exists in this checkout (results/clips/ and the cache are git-ignored
local-only artifacts) and whether every cache key field matches exactly, per
reuse_audit.md's decision to reuse the caching *mechanism* while still
validating every key field rather than assuming a hash match implies reuse.
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

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
REPO_ROOT = ASSIGNMENT_ROOT.parent.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
CACHE_PATH = OUT_ROOT / ".detection_cache.json"
ASSIGNMENT3_CACHE_PATH = REPO_ROOT / "assignments" / "03_sort_paper_experiment" / "results" / ".detection_cache.json"
MANIFEST_PATH = OUT_ROOT / "clip_manifest.csv"
DETECTIONS_CSV = OUT_ROOT / "detections.csv"
MODEL_NAME = "yolo26n.pt"
CONF_THRESHOLD = 0.05  # detection floor -- required <= 0.05 so weak boxes survive
DEFAULT_IMGSZ = 640
FALLBACK_IMGSZ = 480
SLOW_THRESHOLD_S = 2.5
DEVICE = "cpu"


def load_frames() -> list[dict]:
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def package_versions() -> dict:
    return {"ultralytics": ultralytics.__version__, "python": platform.python_version()}


def get_weights_hash(weights_path: Path) -> str:
    return sha256_of_file(weights_path) if weights_path.is_file() else "unresolved-will-download"


def run_one(model: YOLO, image_path: Path, imgsz: int) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    preds = model.predict(source=str(image_path), imgsz=imgsz, conf=CONF_THRESHOLD, device=DEVICE, verbose=False)
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

    weights_path = REPO_ROOT / MODEL_NAME
    model = YOLO(str(weights_path) if weights_path.is_file() else MODEL_NAME)
    weights_hash = get_weights_hash(weights_path)
    pkg_versions = package_versions()

    bench_frames = frames[: min(5, len(frames))]
    bench_times = []
    for f in bench_frames:
        _, ms = run_one(model, ASSIGNMENT_ROOT / f["experiment_image_path"], DEFAULT_IMGSZ)
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

    # Check whether Assignment 3's cache exists and could seed this run.
    reuse_note = "Assignment 3's detection cache was not found in this checkout -- nothing to reuse; running fresh."
    seed_cache = {}
    if ASSIGNMENT3_CACHE_PATH.is_file():
        a3_cache = DetectionCache(ASSIGNMENT3_CACHE_PATH)
        seed_cache = dict(a3_cache._data)
        reuse_note = (f"Found Assignment 3's detection cache ({len(seed_cache)} entries) -- "
                       "will reuse any entry whose full key (image hash, model, weights hash, "
                       "image size, confidence floor, package versions) matches exactly.")
    print(reuse_note)

    cache = DetectionCache(CACHE_PATH)
    cache._data.update({k: v for k, v in seed_cache.items() if k not in cache._data})

    detection_rows = []
    n_cache_hits_this_run = 0
    n_cache_misses = 0
    n_seeded_from_assignment3 = 0

    for f in frames:
        img_path = ASSIGNMENT_ROOT / f["experiment_image_path"]
        img_hash = sha256_of_file(img_path)
        key = cache_key(img_hash, MODEL_NAME, weights_hash, imgsz, CONF_THRESHOLD, pkg_versions)

        was_seeded = key in seed_cache
        cached = cache.get(key)
        if cached is not None:
            dets, inference_ms = cached["detections"], cached["inference_time_ms"]
            n_cache_hits_this_run += 1
            if was_seeded:
                n_seeded_from_assignment3 += 1
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
                "cache_key": key,
            })

    cache.save()

    with open(DETECTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "frame_number", "timestamp", "class", "confidence",
            "x1", "y1", "x2", "y2", "inference_time_ms", "cache_key",
        ])
        writer.writeheader()
        writer.writerows(detection_rows)

    audit_lines = [
        "# Detector Audit\n\n",
        f"Model: `{MODEL_NAME}` (pretrained nano, prediction only -- no training/fine-tuning).\n",
        f"Device: `{DEVICE}`. Confidence floor (detection_floor): **{CONF_THRESHOLD}**.\n",
        f"{imgsz_note}\n\n",
        "## Cache reuse check (Task 5 / reuse_audit.md)\n\n",
        f"{reuse_note}\n\n",
        f"- Frames total: {len(frames)}\n",
        f"- Cache hits this run: {n_cache_hits_this_run} (of which seeded from Assignment 3: {n_seeded_from_assignment3})\n",
        f"- Cache misses (YOLO actually invoked): {n_cache_misses}\n",
        f"- Total raw detections written: {len(detection_rows)}\n\n",
        "Every detection row carries its cache key, the raw YOLO box, class, confidence, "
        "frame identity, and inference time -- the raw box is never replaced with a "
        "tracker-corrected box anywhere downstream.\n",
    ]
    with open(OUT_ROOT / "detector_audit.md", "w", encoding="utf-8") as f:
        f.writelines(audit_lines)

    print(f"Wrote {DETECTIONS_CSV} ({len(detection_rows)} detections across {len(frames)} frames).")
    print(f"Cache: {n_cache_hits_this_run} hits ({n_seeded_from_assignment3} seeded), "
          f"{n_cache_misses} misses, {len(cache)} total entries.")
    print(f"Image size used: {imgsz}")


if __name__ == "__main__":
    main()

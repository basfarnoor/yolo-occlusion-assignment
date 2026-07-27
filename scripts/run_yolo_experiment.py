"""Run a pretrained, frozen YOLO nano model over the organized occlusion samples.

Prediction only -- no training or fine-tuning. CPU only. Processes one image
at a time (never loads the whole dataset into memory). See
assignments/1_yolo_occlusion.md Task 4 for the full specification.
"""
from __future__ import annotations

import csv
import platform
import re
import sys
import time
from pathlib import Path

import ultralytics
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_ROOT = PROJECT_ROOT / "occluded_samples"
RESULTS_ROOT = PROJECT_ROOT / "results"
ANNOTATED_ROOT = RESULTS_ROOT / "annotated"

MODEL_NAME = "yolo26n.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.05
DEVICE = "cpu"

STAGE_FILENAME_RE = re.compile(r"^(\d+)_([a-z_]+)\.(jpg|jpeg|png)$", re.IGNORECASE)


def find_stage_images() -> list[dict]:
    """Discover every organized stage image, in sample/stage order."""
    items = []
    for sample_dir in sorted(SAMPLES_ROOT.iterdir()):
        if not sample_dir.is_dir():
            continue
        for img_path in sorted(sample_dir.iterdir()):
            m = STAGE_FILENAME_RE.match(img_path.name)
            if not m:
                continue
            stage_number, stage_name, _ext = m.groups()
            items.append({
                "sample_number": sample_dir.name,
                "stage_number": int(stage_number),
                "stage_name": stage_name,
                "image_path": img_path,
            })
    return items


def main() -> None:
    if not SAMPLES_ROOT.is_dir():
        print(f"ERROR: {SAMPLES_ROOT} does not exist. Run scripts/organize_occluded_samples.py first.")
        sys.exit(1)

    stage_images = find_stage_images()
    if not stage_images:
        print("No organized images found to process.")
        sys.exit(1)

    print(f"Found {len(stage_images)} images to process across "
          f"{len(sorted({s['sample_number'] for s in stage_images}))} samples.")

    weights_path = PROJECT_ROOT / MODEL_NAME
    if not weights_path.exists():
        print(f"'{MODEL_NAME}' was not found locally. Ultralytics will download the small "
              f"pretrained nano weights (a few MB) from its official release the first time "
              f"this runs, then reuse the local copy afterward.")

    print(f"Loading model {MODEL_NAME} ...")
    model = YOLO(MODEL_NAME)

    ANNOTATED_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    detection_rows = []
    image_rows = []
    n_failed = 0
    total_start = time.perf_counter()

    for item in stage_images:
        sample_number = item["sample_number"]
        stage_number = item["stage_number"]
        stage_name = item["stage_name"]
        img_path: Path = item["image_path"]

        out_dir = ANNOTATED_ROOT / sample_number
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / img_path.name

        t0 = time.perf_counter()
        try:
            preds = model.predict(
                source=str(img_path),
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )
            result = preds[0]
            result.save(filename=str(out_path))
            inference_ms = (time.perf_counter() - t0) * 1000

            n_boxes = 0
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    cls_name = result.names.get(cls_id, str(cls_id))
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    detection_rows.append({
                        "sample_number": sample_number,
                        "stage_number": stage_number,
                        "stage_name": stage_name,
                        "image_filename": img_path.name,
                        "detected_class": cls_name,
                        "confidence": round(conf, 5),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "box_width": round(x2 - x1, 2),
                        "box_height": round(y2 - y1, 2),
                        "inference_time_ms": round(inference_ms, 2),
                    })
                    n_boxes += 1

            image_rows.append({
                "sample_number": sample_number,
                "stage_number": stage_number,
                "stage_name": stage_name,
                "image_filename": img_path.name,
                "num_detections": n_boxes,
                "status": "ok",
                "inference_time_ms": round(inference_ms, 2),
                "annotated_path": str(out_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "error": "",
            })
            print(f"  {sample_number}/{img_path.name}: {n_boxes} detection(s), "
                  f"{inference_ms:.1f} ms")
        except Exception as exc:  # noqa: BLE001 - report and continue
            inference_ms = (time.perf_counter() - t0) * 1000
            n_failed += 1
            image_rows.append({
                "sample_number": sample_number,
                "stage_number": stage_number,
                "stage_name": stage_name,
                "image_filename": img_path.name,
                "num_detections": 0,
                "status": "failed",
                "inference_time_ms": round(inference_ms, 2),
                "annotated_path": "",
                "error": str(exc),
            })
            print(f"  {sample_number}/{img_path.name}: FAILED - {exc}")

    total_runtime_s = time.perf_counter() - total_start
    n_processed = len(stage_images)
    avg_ms = (total_runtime_s * 1000 / n_processed) if n_processed else 0.0

    # all_detections.csv -- one row per detected box
    det_path = RESULTS_ROOT / "all_detections.csv"
    with open(det_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_number", "stage_number", "stage_name", "image_filename",
                      "detected_class", "confidence", "x1", "y1", "x2", "y2",
                      "box_width", "box_height", "inference_time_ms"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detection_rows)

    # all_images.csv -- one row per processed image, so zero-detection images are never lost
    img_csv_path = RESULTS_ROOT / "all_images.csv"
    with open(img_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_number", "stage_number", "stage_name", "image_filename",
                      "num_detections", "status", "inference_time_ms", "annotated_path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(image_rows)

    # run_summary.md
    summary_path = RESULTS_ROOT / "run_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# YOLO Occlusion-Sensitivity Run Summary\n\n")
        f.write(f"- Model: `{MODEL_NAME}` (pretrained, prediction only -- no training/fine-tuning)\n")
        f.write(f"- Ultralytics package version: `{ultralytics.__version__}`\n")
        f.write(f"- Python version: `{platform.python_version()}`\n")
        f.write(f"- Device: `{DEVICE}`\n")
        f.write(f"- Image size: `{IMG_SIZE}`\n")
        f.write(f"- Confidence threshold: `{CONF_THRESHOLD}`\n")
        f.write(f"- Images processed: **{n_processed}**\n")
        f.write(f"- Images failed: **{n_failed}**\n")
        f.write(f"- Total runtime: **{total_runtime_s:.2f} s**\n")
        f.write(f"- Average runtime per image: **{avg_ms:.1f} ms**\n")
        f.write(f"- Annotated results: `{ANNOTATED_ROOT.relative_to(PROJECT_ROOT)}/<sample>/`\n\n")
        f.write("This is a raw-detection count, not a measure of correctness. A box appearing "
                "in the output means the model produced a prediction above the confidence "
                "threshold -- it does not by itself mean the prediction is right. Task 5 "
                "manually reviews one target object per sample to judge correctness.\n")

    print(f"\nDone. {n_processed} images processed, {n_failed} failed.")
    print(f"Wrote {det_path}, {img_csv_path}, {summary_path}")


if __name__ == "__main__":
    main()

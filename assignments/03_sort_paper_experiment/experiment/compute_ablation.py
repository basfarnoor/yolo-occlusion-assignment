"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 9: an ablation isolating whether motion prediction itself -- not just
"having a track ID" -- causes any improvement. Compares:
  - static_memory       (Assignment 2 baseline, box frozen)
  - sort_with_motion    (Task 4's Kalman filter, real learned velocity)
  - sort_zero_velocity  (identical Kalman filter, velocity forced to 0)

No YOLO rerun: reuses the same cached detections and the same eligible
track selection as run_experiment.py (deterministic, seed 42).
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from baselines import run_three_baselines  # noqa: E402
from geometry import center_error, iou  # noqa: E402
from track_selection import build_natural_tracks, select_eligible_tracks  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXP_ROOT = Path(__file__).resolve().parent
OUT_ROOT = PROJECT_ROOT / "results"
ALLOWED_CLASSES = ("car", "truck", "bus", "person")


def load_config():
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_detections():
    detections_by_frame = defaultdict(list)
    clip_frame_numbers = defaultdict(set)
    with open(OUT_ROOT / "detections.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip, frame_no = row["clip"], int(row["frame_number"])
            clip_frame_numbers[clip].add(frame_no)
            detections_by_frame[(clip, frame_no)].append({
                "class": row["class"], "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })
    return dict(detections_by_frame), {c: sorted(v) for c, v in clip_frame_numbers.items()}


def main() -> None:
    cfg = load_config()
    occ_cfg = cfg["occlusion_experiment"]
    detections_by_frame, clip_frame_numbers = load_detections()

    natural_tracks = build_natural_tracks(detections_by_frame, clip_frame_numbers)
    selected, _log = select_eligible_tracks(
        natural_tracks,
        min_track_length=occ_cfg["min_track_length"],
        min_track_length_floor=occ_cfg["min_track_length_floor"],
        max_eligible_tracks=occ_cfg["max_eligible_tracks"],
        min_confidence=occ_cfg["min_confidence"],
        min_displacement_px=occ_cfg["min_displacement_px"],
        allowed_classes=ALLOWED_CLASSES,
        seed=occ_cfg["random_seed"],
    )

    rows = []
    for track in selected:
        n = len(track.frame_numbers)
        for gap_len in occ_cfg["gap_lengths"]:
            if n < gap_len + 2:
                continue
            gap_start_idx = (n - gap_len) // 2
            if gap_start_idx <= 0 or gap_start_idx + gap_len >= n:
                continue

            withheld = track.boxes[gap_start_idx: gap_start_idx + gap_len]
            results_motion = run_three_baselines(track.boxes, track.class_name, gap_start_idx, gap_len,
                                                    force_zero_velocity=False)
            results_zero_v = run_three_baselines(track.boxes, track.class_name, gap_start_idx, gap_len,
                                                    force_zero_velocity=True)

            for offset in range(gap_len):
                pseudo_gt = withheld[offset]
                variants = {
                    "static_memory": results_motion[offset].static_memory_box,
                    "sort_with_motion": results_motion[offset].sort_box,
                    "sort_zero_velocity": results_zero_v[offset].sort_box,
                }
                for variant, box in variants.items():
                    rows.append({
                        "clip": track.clip, "track_id": track.track_id,
                        "gap_length": gap_len, "frame_offset_in_gap": offset,
                        "variant": variant,
                        "center_error_px": center_error(box, pseudo_gt),
                        "iou": iou(box, pseudo_gt),
                    })

    ablation_csv = OUT_ROOT / "ablation.csv"
    with open(ablation_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "track_id", "gap_length", "frame_offset_in_gap",
                                                  "variant", "center_error_px", "iou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {ablation_csv} ({len(rows)} rows)")

    by_variant = defaultdict(list)
    for r in rows:
        by_variant[r["variant"]].append(r)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    variants = ["static_memory", "sort_zero_velocity", "sort_with_motion"]
    colors = ["#FFA630", "#8B5E83", "#1C7293"]
    labels = ["Static memory", "SORT, velocity forced to 0", "SORT, real motion"]

    ce_means = [statistics.mean(r["center_error_px"] for r in by_variant[v]) for v in variants]
    ce_ns = [len(by_variant[v]) for v in variants]
    bars = axes[0].bar(labels, ce_means, color=colors)
    for bar, n in zip(bars, ce_ns):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center", va="bottom")
    axes[0].set_ylabel("Mean center error (pixels)")
    axes[0].set_title("Ablation: center error")
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

    iou_means = [statistics.mean(r["iou"] for r in by_variant[v]) for v in variants]
    bars = axes[1].bar(labels, iou_means, color=colors)
    for bar, n in zip(bars, ce_ns):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}", ha="center", va="bottom")
    axes[1].set_ylabel("Mean IoU with withheld box")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Ablation: IoU")
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    fig.suptitle("Does motion prediction itself help, or just having a track ID?")
    plt.tight_layout()
    ablation_png = OUT_ROOT / "ablation.png"
    plt.savefig(ablation_png, dpi=120)
    plt.close(fig)
    print(f"Wrote {ablation_png}")

    print("\nAblation summary:")
    for v, label_ in zip(variants, labels):
        ce = statistics.mean(r["center_error_px"] for r in by_variant[v])
        iou_m = statistics.mean(r["iou"] for r in by_variant[v])
        print(f"  {label_:30s} center_error={ce:.2f}px  iou={iou_m:.3f}  n={len(by_variant[v])}")


if __name__ == "__main__":
    main()

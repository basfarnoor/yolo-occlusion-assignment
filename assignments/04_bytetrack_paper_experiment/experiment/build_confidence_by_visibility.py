"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 5: shows the distribution of detection confidence by nuScenes visibility
category on DEVELOPMENT scenes only, before any threshold is chosen. This is
what config.yaml's high_score_threshold / new_track_threshold / detection_floor
are set from -- evaluation-split results must stay unopened until afterward.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from geometry import iou  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
CHARTS_ROOT = OUT_ROOT / "charts"

DEV_CLIPS = {"clip_sample_001", "clip_sample_006"}
MATCH_IOU_THRESHOLD = 0.3
VISIBILITY_ORDER = ["v0-40", "v40-60", "v60-80", "v80-100"]
# Only categories a COCO-trained YOLO could plausibly detect -- excludes
# traffic cones, barriers, debris, etc. that would otherwise spuriously match
# a nearby YOLO box purely by IoU.
TRACKABLE_CATEGORY_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.motorcycle",
                               "vehicle.bicycle", "human.pedestrian")


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    detections = load_csv(OUT_ROOT / "detections.csv")
    gt_rows = [r for r in load_csv(OUT_ROOT / "projected_ground_truth.csv") if r["rejected"] in ("False", "false")]

    dets_by_frame = defaultdict(list)
    for d in detections:
        dets_by_frame[(d["clip"], d["frame_number"])].append(d)

    gt_by_frame = defaultdict(list)
    for g in gt_rows:
        if g["clip_name"] in DEV_CLIPS and g["category"].startswith(TRACKABLE_CATEGORY_PREFIXES):
            gt_by_frame[(g["clip_name"], g["frame_number"])].append(g)

    matched_confidence_by_visibility = defaultdict(list)
    n_gt_considered = 0
    n_gt_matched = 0

    for key, gts in gt_by_frame.items():
        dets = dets_by_frame.get(key, [])
        det_boxes = [(float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"])) for d in dets]
        for g in gts:
            n_gt_considered += 1
            gt_box = (float(g["x1"]), float(g["y1"]), float(g["x2"]), float(g["y2"]))
            best_iou, best_conf = 0.0, None
            for d, dbox in zip(dets, det_boxes):
                i = iou(gt_box, dbox)
                if i > best_iou:
                    best_iou, best_conf = i, float(d["confidence"])
            if best_iou >= MATCH_IOU_THRESHOLD:
                n_gt_matched += 1
                matched_confidence_by_visibility[g["visibility_level"]].append(best_conf)

    # Chart: one box/violin-style summary per visibility category.
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [matched_confidence_by_visibility.get(v, []) for v in VISIBILITY_ORDER]
    labels = [f"{v}\n(n={len(d)})" for v, d in zip(VISIBILITY_ORDER, data)]
    non_empty = [(lbl, d) for lbl, d in zip(labels, data) if d]
    if non_empty:
        ax.boxplot([d for _, d in non_empty], tick_labels=[lbl for lbl, _ in non_empty], showmeans=True)
    ax.set_ylabel("Matched YOLO detection confidence")
    ax.set_xlabel("nuScenes visibility category (development scenes only)")
    ax.set_title("Detection confidence by visibility -- development scenes\n"
                 "(higher visibility should generally mean higher confidence)")
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="detection_floor = 0.05")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    chart_path = CHARTS_ROOT / "detection_confidence_by_visibility.png"
    fig.savefig(chart_path, dpi=130)
    plt.close(fig)

    print(f"Development-scene ground-truth boxes considered: {n_gt_considered}")
    print(f"Matched to a detection (IoU >= {MATCH_IOU_THRESHOLD}): {n_gt_matched}")
    for v in VISIBILITY_ORDER:
        vals = matched_confidence_by_visibility.get(v, [])
        if vals:
            print(f"  {v}: n={len(vals)}, mean_conf={sum(vals)/len(vals):.3f}, "
                  f"min={min(vals):.3f}, max={max(vals):.3f}")
        else:
            print(f"  {v}: n=0")
    print(f"Wrote {chart_path}")


if __name__ == "__main__":
    main()

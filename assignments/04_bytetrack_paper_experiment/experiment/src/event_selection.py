"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 9: finds candidate "natural" weak-evidence events -- keyframes where the
same nuScenes instance_token's best-matched raw YOLO confidence crosses from
the high-score band into the low-score band (or disappears entirely), while
the object persists in independent ground truth before and after.

nuScenes only annotates keyframes (roughly every 6th frame in these 36-frame
clips), so events here are identified at keyframe granularity, not
frame-by-frame -- this sparsity is a genuine dataset limitation, recorded in
the final report rather than hidden.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from geometry import iou  # noqa: E402

TRACKABLE_CATEGORY_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.motorcycle",
                               "vehicle.bicycle", "human.pedestrian")
MATCH_IOU_THRESHOLD = 0.3
VISIBILITY_RANK = {"v0-40": 0, "v40-60": 1, "v60-80": 2, "v80-100": 3}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_detection_to_gt(gt_box: tuple[float, float, float, float], detections: list[dict]) -> tuple[float, float] | None:
    """Returns (confidence, iou) of the best-overlapping detection, or None if
    nothing clears MATCH_IOU_THRESHOLD."""
    best_iou, best_conf = 0.0, None
    for d in detections:
        dbox = (float(d["x1"]), float(d["y1"]), float(d["x2"]), float(d["y2"]))
        i = iou(gt_box, dbox)
        if i > best_iou:
            best_iou, best_conf = i, float(d["confidence"])
    if best_iou >= MATCH_IOU_THRESHOLD:
        return best_conf, best_iou
    return None


def build_instance_sequences(gt_rows: list[dict], detections_by_frame: dict) -> dict[tuple[str, str], list[dict]]:
    """Returns {(clip_name, instance_token): [ {frame_number, confidence_or_None, visibility_level, category, gt_box}, ... ]}
    sorted by frame_number, restricted to trackable categories."""
    sequences: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in gt_rows:
        if g["rejected"] in ("True", "true"):
            continue
        if not g["category"].startswith(TRACKABLE_CATEGORY_PREFIXES):
            continue
        key = (g["clip_name"], g["instance_token"])
        gt_box = (float(g["x1"]), float(g["y1"]), float(g["x2"]), float(g["y2"]))
        dets = detections_by_frame.get((g["clip_name"], g["frame_number"]), [])
        match = match_detection_to_gt(gt_box, dets)
        sequences[key].append({
            "frame_number": int(g["frame_number"]),
            "confidence": match[0] if match else None,
            "matched_iou": match[1] if match else None,
            "visibility_level": g["visibility_level"],
            "category": g["category"],
            "gt_box": gt_box,
        })
    for key in sequences:
        sequences[key].sort(key=lambda r: r["frame_number"])
    return sequences


def find_confidence_drop_events(sequences: dict[tuple[str, str], list[dict]], high_score_threshold: float
                                 ) -> list[dict]:
    """A candidate event: instance present (any confidence, i.e. a GT box
    exists) at keyframe i-1 with confidence >= high_score_threshold, then at
    keyframe i confidence < high_score_threshold OR no detection matched at
    all, then the instance is still present (any match state) in GT at
    keyframe i+1 or later."""
    events = []
    for (clip_name, instance_token), seq in sequences.items():
        if len(seq) < 3:
            continue  # need at least "before", "event", "after"
        for i in range(1, len(seq) - 1):
            before, event, after = seq[i - 1], seq[i], seq[i + 1]
            before_high = before["confidence"] is not None and before["confidence"] >= high_score_threshold
            event_weak = event["confidence"] is None or event["confidence"] < high_score_threshold
            if before_high and event_weak:
                events.append({
                    "clip_name": clip_name,
                    "instance_token": instance_token,
                    "category": event["category"],
                    "event_frame": event["frame_number"],
                    "before_frame": before["frame_number"],
                    "after_frame": after["frame_number"],
                    "before_confidence": before["confidence"],
                    "event_confidence": event["confidence"],
                    "after_confidence": after["confidence"],
                    "before_visibility": before["visibility_level"],
                    "event_visibility": event["visibility_level"],
                    "after_visibility": after["visibility_level"],
                    "plausible_cause": (
                        "visibility_drop"
                        if VISIBILITY_RANK.get(event["visibility_level"], 3) < VISIBILITY_RANK.get(before["visibility_level"], 3)
                        else ("no_detection_match" if event["confidence"] is None else "confidence_drop_same_visibility")
                    ),
                    "frames_available_before": i,
                    "frames_available_after": len(seq) - 1 - i,
                })
    return events

"""Task 11: method-level metrics computed over a full, UNMODIFIED continuous
per-scene run (never restricted to an event window) -- these answer "did
this method get worse at ordinary tracking/detection", not "how does it
handle occlusion" (that's event_metrics.py). Visible-object precision/recall
uses only rows the method itself claims as currently observed
(OBSERVED_STRONG/OBSERVED_WEAK); a PREDICTED_HIDDEN row is a memory guess,
not a claim of current visibility, and must never count toward either.
"""
from __future__ import annotations

from collections import defaultdict

from oatm.evaluation.ground_truth import DETECTOR_TO_EVAL_CLASS
from oatm.tracking.geometry import iou

VISIBLE_STATES = ("OBSERVED_STRONG", "OBSERVED_WEAK")


def compute_precision_recall(
    outputs: list[dict], gt_by_frame: dict[str, list[dict]], iou_threshold: float = 0.5,
    keyframe_sample_data_tokens: set[str] | None = None,
) -> dict:
    """Greedy one-to-one IoU matching per frame per class. Returns overall and
    per-class {tp, fp, fn, precision, recall}.

    nuScenes only has 3D annotations at keyframes (2Hz) -- the other, more
    numerous "sweep" frames (12Hz) have NO ground truth at all, not just
    unlabeled ones. Scoring predictions on those frames against empty ground
    truth would count every real detection there as a false positive,
    collapsing precision to a meaningless number. `keyframe_sample_data_tokens`,
    when given, restricts scoring to frames where ground truth genuinely
    exists."""
    predicted_by_frame_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in outputs:
        if row["state"] not in VISIBLE_STATES:
            continue
        if keyframe_sample_data_tokens is not None:
            if row["sample_data_token"] not in keyframe_sample_data_tokens:
                continue
        eval_class = DETECTOR_TO_EVAL_CLASS.get(row["class_name"])
        if eval_class is None:
            continue
        predicted_by_frame_class[(row["sample_data_token"], eval_class)].append(row)

    gt_by_frame_class: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sdt, rows in gt_by_frame.items():
        for r in rows:
            gt_by_frame_class[(sdt, r["evaluation_class"])].append(r)

    counts = {"car": {"tp": 0, "fp": 0, "fn": 0}, "pedestrian": {"tp": 0, "fp": 0, "fn": 0}}
    all_keys = set(predicted_by_frame_class) | set(gt_by_frame_class)
    for key in all_keys:
        _, eval_class = key
        if eval_class not in counts:
            continue
        preds = predicted_by_frame_class.get(key, [])
        gts = gt_by_frame_class.get(key, [])

        pairs = []
        for pi, p in enumerate(preds):
            p_box = (p["x1"], p["y1"], p["x2"], p["y2"])
            for gi, g in enumerate(gts):
                score = iou(p_box, g["box"])
                if score >= iou_threshold:
                    pairs.append((score, pi, gi))
        pairs.sort(reverse=True, key=lambda t: t[0])

        matched_p, matched_g = set(), set()
        for _, pi, gi in pairs:
            if pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi)
            matched_g.add(gi)

        counts[eval_class]["tp"] += len(matched_p)
        counts[eval_class]["fp"] += len(preds) - len(matched_p)
        counts[eval_class]["fn"] += len(gts) - len(matched_g)

    result = {}
    total_tp = total_fp = total_fn = 0
    for eval_class, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        result[eval_class] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else None,
            "recall": tp / (tp + fn) if (tp + fn) else None,
        }
    result["overall"] = {
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "precision": total_tp / (total_tp + total_fp) if (total_tp + total_fp) else None,
        "recall": total_tp / (total_tp + total_fn) if (total_tp + total_fn) else None,
    }
    return result


def compute_ghost_rate(
    outputs: list[dict], gt_by_frame: dict[str, list[dict]], iou_threshold: float = 0.3,
) -> dict:
    """A track is a "ghost" if not one single row across its entire life ever
    overlapped a real, same-class ground-truth box -- i.e. it was never once
    supported by anything real. Excludes yolo_only, whose track_id carries no
    real cross-frame identity (see reuse_audit.md) -- grouping its rows by
    track_id would fabricate fake "tracks" the same way Task 6 already found
    and rejected doing for its own summary stats."""
    by_track: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in outputs:
        by_track[(row["scene_token"], row["track_id"])].append(row)

    n_tracks = 0
    n_ghost = 0
    ghost_durations = []
    for rows in by_track.values():
        n_tracks += 1
        ever_supported = False
        for row in rows:
            eval_class = DETECTOR_TO_EVAL_CLASS.get(row["class_name"])
            if eval_class is None:
                continue
            box = (row["x1"], row["y1"], row["x2"], row["y2"])
            gt_rows = [
                g for g in gt_by_frame.get(row["sample_data_token"], [])
                if g["evaluation_class"] == eval_class
            ]
            if any(iou(box, g["box"]) >= iou_threshold for g in gt_rows):
                ever_supported = True
                break
        if not ever_supported:
            n_ghost += 1
            ghost_durations.append(len(rows))

    return {
        "n_tracks": n_tracks,
        "n_ghost_tracks": n_ghost,
        "ghost_rate": (n_ghost / n_tracks) if n_tracks else None,
        "mean_ghost_duration_frames": (
            (sum(ghost_durations) / len(ghost_durations)) if ghost_durations else None
        ),
    }

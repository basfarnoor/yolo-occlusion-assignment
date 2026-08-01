"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 13: side-by-side comparison videos (raw YOLO evidence | high-confidence
SORT | ByteTrack | evaluation reference) for three showcase scenarios --
a successful low-score recovery, a false/ambiguous weak detection, and a
complete detection absence -- plus one slow explanatory video. Every box
states whether it is current visual evidence, a motion prediction, or an
offline evaluation reference (never conflating the three).
"""
from __future__ import annotations

import copy
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from controlled_trials import COMPLETE_ABSENCE, build_modified_detections  # noqa: E402
from kalman_box_tracker import KalmanBoxTracker  # noqa: E402
from run_methods import new_bytetrack_tracker, new_sort_tracker  # noqa: E402
from target_selection import build_natural_targets  # noqa: E402
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE  # noqa: E402
from visualization import BLUE, CYAN, GREEN, MAGENTA, ORANGE, PURPLE, YELLOW, compose_grid, panel_from_boxes

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
VIDEOS_ROOT = OUT_ROOT / "videos"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_common(cfg: dict):
    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")
    gt_rows = load_csv(OUT_ROOT / "projected_ground_truth.csv")

    image_path_by_frame = {}
    timestamps_by_clip = defaultdict(dict)
    clip_frame_numbers = defaultdict(list)
    for row in manifest_rows:
        clip, frame_no = row["clip_name"], int(row["frame_number"])
        image_path_by_frame[(clip, frame_no)] = ASSIGNMENT_ROOT / row["experiment_image_path"]
        timestamps_by_clip[clip][frame_no] = float(row["timestamp"]) / 1_000_000.0
        clip_frame_numbers[clip].append(frame_no)
    for clip in clip_frame_numbers:
        clip_frame_numbers[clip].sort()

    detections_by_frame = defaultdict(list)
    for d in detections:
        detections_by_frame[(d["clip"], int(d["frame_number"]))].append({
            "class": d["class"], "confidence": float(d["confidence"]),
            "x1": float(d["x1"]), "y1": float(d["y1"]), "x2": float(d["x2"]), "y2": float(d["y2"]),
        })

    gt_by_key = {}
    for g in gt_rows:
        if g["rejected"] in ("True", "true"):
            continue
        gt_by_key[(g["clip_name"], g["frame_number"], g["instance_token"])] = g

    return dict(detections_by_frame), dict(clip_frame_numbers), dict(timestamps_by_clip), image_path_by_frame, gt_by_key


def evidence_style(evidence_source: str) -> tuple:
    if evidence_source == EVIDENCE_HIGH_SCORE:
        return BLUE, "solid"
    if evidence_source == EVIDENCE_LOW_SCORE:
        return ORANGE, "solid"
    return PURPLE, "dashed"


def bytetrack_evidence_style(evidence_source: str) -> tuple:
    if evidence_source == EVIDENCE_HIGH_SCORE:
        return CYAN, "solid"
    if evidence_source == EVIDENCE_LOW_SCORE:
        return ORANGE, "solid"
    return PURPLE, "dashed"


def render_scenario(clip: str, frame_range: list[int], detections_by_frame: dict, timestamps: dict,
                     image_path_by_frame: dict, reference_by_frame: dict, cfg: dict, out_path: Path,
                     high_score_threshold: float) -> None:
    sort_tracker = new_sort_tracker(cfg)
    byte_tracker = new_bytetrack_tracker(cfg)

    all_frames = sorted(set(range(1, max(frame_range) + 1)) & set(timestamps.keys()))
    # Warm up both trackers from the start of the clip so identities are
    # established before the showcased window (a real online tracker would
    # have seen these frames too).
    frames_out = []
    for frame_no in all_frames:
        dets = detections_by_frame.get((clip, frame_no), [])
        sort_outputs = sort_tracker.update(dets, timestamp=timestamps[frame_no])
        byte_outputs = byte_tracker.update(dets, timestamp=timestamps[frame_no])
        if frame_no not in frame_range:
            continue

        img_path = image_path_by_frame[(clip, frame_no)]

        yolo_boxes = [((d["x1"], d["y1"], d["x2"], d["y2"]),
                        GREEN if d["confidence"] >= high_score_threshold else YELLOW,
                        f"{d['class']} {d['confidence']:.2f}", "solid") for d in dets]

        sort_boxes = []
        for o in sort_outputs:
            color, style = evidence_style(o.evidence_source)
            sort_boxes.append((o.box, color, f"id{o.track_id} {o.evidence_source.replace('_', ' ')}", style))

        byte_boxes = []
        for o in byte_outputs:
            color, style = bytetrack_evidence_style(o.evidence_source)
            byte_boxes.append((o.box, color, f"id{o.track_id} {o.evidence_source.replace('_', ' ')}", style))

        ref = reference_by_frame.get(frame_no)
        ref_boxes = [(ref[0], MAGENTA, ref[1], "dotted")] if ref else []

        panels = [
            panel_from_boxes(img_path, yolo_boxes, f"RAW YOLO EVIDENCE  f{frame_no}"),
            panel_from_boxes(img_path, sort_boxes, "HIGH-CONFIDENCE SORT"),
            panel_from_boxes(img_path, byte_boxes, "BYTETRACK"),
            panel_from_boxes(img_path, ref_boxes, "EVALUATION REFERENCE"),
        ]
        frame_arr = np.array(compose_grid(panels, cols=2))[:, :, ::-1]
        frames_out.append(frame_arr)

    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    h, w = frames_out[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
    for f in frames_out:
        writer.write(f)
    writer.release()
    print(f"Wrote {out_path} ({len(frames_out)} frames)")


def build_recovery_scenario(cfg, detections_by_frame, clip_frame_numbers, timestamps_by_clip,
                              image_path_by_frame, gt_by_key):
    """Instance 8c3247533... (pedestrian, clip_sample_011): ByteTrack recovers
    and preserves identity through the natural confidence dip; SORT does not."""
    clip = "clip_sample_011"
    instance_token = "8c3247533921448abe371a59d4696252"
    frame_range = list(range(1, 14))  # covers before=1, event=7, after=13

    reference_by_frame = {}
    for frame_no in frame_range:
        gt = gt_by_key.get((clip, str(frame_no), instance_token))
        if gt:
            box = (float(gt["x1"]), float(gt["y1"]), float(gt["x2"]), float(gt["y2"]))
            reference_by_frame[frame_no] = (box, "projected nuScenes GT")

    render_scenario(clip, frame_range, detections_by_frame, timestamps_by_clip[clip], image_path_by_frame,
                     reference_by_frame, cfg, VIDEOS_ROOT / "comparison_recovery_clip_sample_011_pedestrian.mp4",
                     cfg["tracker"]["high_score_threshold"])


def build_controlled_scenario(cfg, detections_by_frame, clip_frame_numbers, timestamps_by_clip,
                                image_path_by_frame, event_id: str, out_name: str, targets_by_key: dict):
    events = load_csv(OUT_ROOT / "controlled_event_manifest.csv")
    event = next(e for e in events if e["event_id"] == event_id)
    clip = event["clip"]
    track = targets_by_key[(clip, int(event["natural_track_id"]))]
    window_frames = {int(f) for f in event["window_frame_numbers"].split(";")}

    modified = build_modified_detections(
        detections_by_frame, clip, track.frame_numbers, track.raw_boxes, window_frames,
        COMPLETE_ABSENCE, demoted_confidence=0.0)

    lo = max(1, min(window_frames) - 4)
    hi = min(max(clip_frame_numbers[clip]), max(window_frames) + 4)
    frame_range = list(range(lo, hi + 1))

    raw_box_by_frame = dict(zip(track.frame_numbers, track.raw_boxes))
    reference_by_frame = {f: (raw_box_by_frame[f], "pseudo-GT (withheld raw YOLO box)")
                           for f in frame_range if f in raw_box_by_frame}

    render_scenario(clip, frame_range, modified, timestamps_by_clip[clip], image_path_by_frame,
                     reference_by_frame, cfg, VIDEOS_ROOT / out_name, cfg["tracker"]["high_score_threshold"])


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    detections_by_frame, clip_frame_numbers, timestamps_by_clip, image_path_by_frame, gt_by_key = load_common(cfg)

    print("1/3: recovery scenario...")
    build_recovery_scenario(cfg, detections_by_frame, clip_frame_numbers, timestamps_by_clip,
                              image_path_by_frame, gt_by_key)

    print("Rebuilding natural targets for controlled scenarios...")
    # Reset the global track-ID counter so this replay assigns the SAME
    # track IDs build_controlled_experiments.py originally used (it also ran
    # build_natural_targets as the first tracker-creating call in a fresh
    # process) -- otherwise IDs drift from whatever ran earlier in this
    # process (e.g. the recovery scenario above) and natural_track_id lookups
    # from controlled_event_manifest.csv would silently point at the wrong track.
    KalmanBoxTracker.reset_id_counter()
    targets = build_natural_targets(detections_by_frame, clip_frame_numbers, cfg)
    targets_by_key = {(t.clip, t.track_id): t for t in targets}

    print("2/3: false/ambiguous association scenario...")
    build_controlled_scenario(cfg, detections_by_frame, clip_frame_numbers, timestamps_by_clip,
                                image_path_by_frame, "clip_sample_001_track10_complete_absence_w7",
                                "comparison_false_association_clip_sample_001_track10.mp4", targets_by_key)

    print("3/3: complete-absence scenario...")
    build_controlled_scenario(cfg, detections_by_frame, clip_frame_numbers, timestamps_by_clip,
                                image_path_by_frame, "clip_sample_001_track3_complete_absence_w7",
                                "comparison_complete_absence_clip_sample_001_track3.mp4", targets_by_key)


if __name__ == "__main__":
    main()

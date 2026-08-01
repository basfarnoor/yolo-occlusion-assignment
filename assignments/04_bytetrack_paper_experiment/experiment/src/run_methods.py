"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 8: the three fair comparison methods, all run on identical ordered
frames and identical raw per-frame detections.

  Method A -- YOLO only: accepted high-score detections reported frame by
              frame, with no tracker and no temporal memory at all. This is a
              detection baseline, not a tracker.
  Method B -- High-confidence SORT: sort_tracker.SortTracker, one association
              round, real track lifecycle and buffer.
  Method C -- ByteTrack: bytetrack_tracker.ByteTrackTracker, same motion model
              and first association as Method B, plus the second,
              low-confidence association round.

A single tracker instance is used for exactly one clip -- clips are never
mixed into the same tracker, so a track ID can never span two different
nuScenes scenes (required test: "frames from different scenes never share a
track").
"""
from __future__ import annotations

from bytetrack_tracker import ByteTrackTracker
from sort_tracker import SortTracker
from track import EVIDENCE_HIGH_SCORE, TrackOutput


def run_yolo_only(detections: list[dict], high_score_threshold: float) -> list[TrackOutput]:
    """No tracker at all: one TrackOutput per accepted high-score detection,
    each with a fresh, meaningless "track_id" (detection index) and no
    temporal memory. Included only as the detector-only baseline."""
    outputs = []
    for i, d in enumerate(detections):
        if d.get("confidence", 0.0) < high_score_threshold:
            continue
        outputs.append(TrackOutput(
            track_id=-1,  # YOLO-only has no track identity at all
            box=(d["x1"], d["y1"], d["x2"], d["y2"]),
            class_name=d["class"],
            evidence_source=EVIDENCE_HIGH_SCORE,
            hits=1, hit_streak=1, time_since_update=0, age=1, confirmed=True,
            velocity=(0.0, 0.0), confidence=d["confidence"],
        ))
    return outputs


def new_sort_tracker(cfg: dict) -> SortTracker:
    t = cfg["tracker"]
    return SortTracker(
        high_score_threshold=t["high_score_threshold"],
        track_buffer=cfg["sort_baseline"]["max_age"],
        new_track_threshold=t["new_track_threshold"],
        iou_threshold=cfg["sort_baseline"]["iou_threshold"],
    )


def new_bytetrack_tracker(cfg: dict) -> ByteTrackTracker:
    t = cfg["tracker"]
    return ByteTrackTracker(
        detection_floor=cfg["detector"]["detection_floor"],
        high_score_threshold=t["high_score_threshold"],
        new_track_threshold=t["new_track_threshold"],
        first_association_iou_threshold=t["first_association_iou_threshold"],
        second_association_iou_threshold=t["second_association_iou_threshold"],
        track_buffer=t["track_buffer"],
    )


def run_tracker_over_clip(tracker, frames: list[dict]) -> dict[int, list[TrackOutput]]:
    """frames: ordered list of {"frame_number", "timestamp", "detections": [...]}
    for ONE clip only. Returns {frame_number: [TrackOutput, ...]}. Feeding
    frames from more than one clip/scene into a single tracker instance would
    let a track ID span two unrelated scenes -- callers must never do this;
    see tests/test_scene_isolation.py."""
    outputs_by_frame = {}
    for f in frames:
        outputs_by_frame[f["frame_number"]] = tracker.update(f["detections"], timestamp=f["timestamp"])
    return outputs_by_frame


def run_three_methods_over_clip(frames: list[dict], cfg: dict) -> dict[str, dict[int, list[TrackOutput]]]:
    """Runs Methods A, B, and C over the SAME `frames` list -- the same
    object, not a copy -- so there is no way for one method to silently see a
    different detection set than another."""
    high_score_threshold = cfg["tracker"]["high_score_threshold"]

    method_a = {f["frame_number"]: run_yolo_only(f["detections"], high_score_threshold) for f in frames}

    sort_tracker = new_sort_tracker(cfg)
    method_b = run_tracker_over_clip(sort_tracker, frames)

    bytetrack_tracker = new_bytetrack_tracker(cfg)
    method_c = run_tracker_over_clip(bytetrack_tracker, frames)

    return {"yolo_only": method_a, "high_confidence_sort": method_b, "bytetrack": method_c}

"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

The tracker itself: on every frame, predict all existing tracks, match them
against the frame's detections, correct matched tracks, start new tracks for
unmatched detections, and remove tracks that have gone unmatched for too
long. This is a small educational reimplementation of the paper's ideas,
not a copy of the authors' sort.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from assignment import associate_detections_to_trackers
from kalman_box_tracker import KalmanBoxTracker


@dataclass
class TrackOutput:
    track_id: int
    box: tuple[float, float, float, float]
    class_name: str
    matched_this_frame: bool
    hits: int
    hit_streak: int
    time_since_update: int
    age: int
    confirmed: bool
    velocity: tuple[float, float]
    confidence: float


class SortTracker:
    """SORT: predict -> associate -> correct -> manage track birth/death."""

    def __init__(self, max_age: int = 3, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: list[dict], timestamp: float = 0.0) -> list[TrackOutput]:
        """detections: list of {"class","confidence","x1","y1","x2","y2"}."""
        self.frame_count += 1

        # 1. Predict every existing track forward.
        predicted_boxes = [t.predict() for t in self.trackers]
        predicted_classes = [t.class_name for t in self.trackers]

        # 2. Associate this frame's detections with the predicted tracks.
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, predicted_boxes, predicted_classes, self.iou_threshold)

        # 3. Correct matched tracks with their real detection.
        matched_tracker_indices = set()
        for det_idx, trk_idx in matches:
            det = detections[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)
            matched_tracker_indices.add(trk_idx)

        # 4. Start new tracks for detections nothing matched.
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_tracker = KalmanBoxTracker(
                (det["x1"], det["y1"], det["x2"], det["y2"]), det["class"], timestamp)
            new_tracker.hits = 1
            new_tracker.hit_streak = 1
            new_tracker.time_since_update = 0
            new_tracker.last_confidence = det.get("confidence", 0.0)
            self.trackers.append(new_tracker)

        # 5. Remove tracks that have been missing for too long.
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 6. Build this frame's output for every surviving track.
        outputs = []
        for i, t in enumerate(self.trackers):
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            outputs.append(TrackOutput(
                track_id=t.id,
                box=t.current_box(),
                class_name=t.class_name,
                matched_this_frame=(t.time_since_update == 0),
                hits=t.hits,
                hit_streak=t.hit_streak,
                time_since_update=t.time_since_update,
                age=t.age,
                confirmed=confirmed,
                velocity=t.velocity(),
                confidence=t.last_confidence,
            ))
        return outputs

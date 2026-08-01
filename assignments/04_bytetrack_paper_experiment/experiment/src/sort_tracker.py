"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Method B, "high-confidence SORT": a single-association-round tracker, reused
and repaired from the Assignment 3 SORT experiment (Bewley et al., ICIP
2016) -- see reuse_audit.md. This is the fair baseline ByteTrack (Method C,
bytetrack_tracker.py) is compared against: identical motion model, identical
first-association logic, identical lifecycle -- the *only* difference between
the two methods is that this tracker never gets a second, low-confidence
association round.

This class receives the exact same raw, unfiltered detection list every
frame as ByteTrackTracker (Task 8's "identical raw detections" requirement) --
it simply discards anything below `high_score_threshold` itself, internally,
before association, and never gives low-score detections a second chance.
That is the *entire* difference from ByteTrackTracker.
"""
from __future__ import annotations

from assignment import associate_detections_to_trackers
from kalman_box_tracker import KalmanBoxTracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_PREDICTION, TrackOutput


class SortTracker:
    """predict -> associate (high-confidence only) -> correct -> manage track birth/death."""

    def __init__(self, high_score_threshold: float = 0.5, track_buffer: int = 5,
                 new_track_threshold: float = 0.6, iou_threshold: float = 0.3, min_hits: int = 1):
        self.high_score_threshold = high_score_threshold
        self.track_buffer = track_buffer
        self.new_track_threshold = new_track_threshold
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None

    def update(self, detections: list[dict], timestamp: float = 0.0) -> list[TrackOutput]:
        """detections: the SAME raw, unfiltered per-frame detection list given
        to ByteTrackTracker. Low-score detections are discarded here, inside
        this method, not by the caller."""
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp

        high_dets = [d for d in detections if d.get("confidence", 0.0) >= self.high_score_threshold]

        predicted_boxes = [t.predict(dt) for t in self.trackers]
        predicted_classes = [t.class_name for t in self.trackers]

        matches, unmatched_dets, _ = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.iou_threshold)

        for det_idx, trk_idx in matches:
            det = high_dets[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)

        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            if det.get("confidence", 0.0) < self.new_track_threshold:
                continue  # too weak to trust as a brand-new object
            new_tracker = KalmanBoxTracker(
                (det["x1"], det["y1"], det["x2"], det["y2"]), det["class"], timestamp)
            new_tracker.hits = 1
            new_tracker.hit_streak = 1
            new_tracker.time_since_update = 0
            new_tracker.last_confidence = det.get("confidence", 0.0)
            self.trackers.append(new_tracker)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.track_buffer]

        outputs = []
        for t in self.trackers:
            confirmed = t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            matched_this_frame = t.time_since_update == 0
            outputs.append(TrackOutput(
                track_id=t.id,
                box=t.current_box(),
                class_name=t.class_name,
                evidence_source=EVIDENCE_HIGH_SCORE if matched_this_frame else EVIDENCE_PREDICTION,
                hits=t.hits,
                hit_streak=t.hit_streak,
                time_since_update=t.time_since_update,
                age=t.age,
                confirmed=confirmed,
                velocity=t.velocity(),
                confidence=t.last_confidence,
                raw_detection_box=t.last_raw_box if matched_this_frame else None,
            ))
        return outputs

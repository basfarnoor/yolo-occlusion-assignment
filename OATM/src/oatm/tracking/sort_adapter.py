"""Method C: high-confidence SORT -- single association round, real track
lifecycle. Reused from Assignment 4's `sort_tracker.py` (see reuse_audit.md);
only the output type changes, to OATM's canonical `TrackerOutputRecord`.

This is also the fair baseline ByteTrack (Method D, bytetrack_adapter.py) is
compared against: identical motion model and first-association logic --
the only difference is ByteTrack's second, low-confidence association round.
"""
from __future__ import annotations

import numpy as np

from oatm.records import TrackerOutputRecord
from oatm.tracking.association import associate_detections_to_trackers
from oatm.tracking.kalman import KalmanBoxTracker


class SortAdapter:
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

    def update(self, detections: list[dict], timestamp: float = 0.0,
               scene_token: str = "", sample_data_token: str = "",
               method_name: str = "sort", run_id: str = "") -> list[TrackerOutputRecord]:
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
                continue
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
            matched_this_frame = t.time_since_update == 0
            box = t.current_box()
            outputs.append(TrackerOutputRecord(
                scene_token=scene_token, sample_data_token=sample_data_token,
                frame_index=self.frame_count - 1, method_name=method_name, run_id=run_id,
                track_id=t.id, class_name=t.class_name,
                state="OBSERVED_STRONG" if matched_this_frame else "PREDICTED_HIDDEN",
                evidence_source="strong_detection" if matched_this_frame else "motion_prediction",
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                raw_detection_x1=t.last_raw_box[0] if t.last_raw_box else None,
                raw_detection_y1=t.last_raw_box[1] if t.last_raw_box else None,
                raw_detection_x2=t.last_raw_box[2] if t.last_raw_box else None,
                raw_detection_y2=t.last_raw_box[3] if t.last_raw_box else None,
                detector_confidence=t.last_confidence if matched_this_frame else None,
                existence_confidence=1.0, identity_confidence=1.0,  # not modeled by this baseline
                localization_uncertainty=float(np.trace(t.P)),  # real Kalman covariance trace
                memory_age_frames=t.time_since_update,
                memory_age_seconds=t.time_since_update * dt,
                termination_reason=None,
            ))
        return outputs

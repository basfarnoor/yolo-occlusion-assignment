"""Method D: ByteTrack -- the two-stage BYTE association tracker. Reused from
Assignment 4's `bytetrack_tracker.py` (see reuse_audit.md), including the
`raw_detection_box` repair (the tracker's smoothed Kalman state must never be
mistaken for a real detection). Only the output type changes, to OATM's
canonical `TrackerOutputRecord`.
"""
from __future__ import annotations

import numpy as np

from oatm.records import TrackerOutputRecord
from oatm.tracking.association import associate_detections_to_trackers
from oatm.tracking.kalman import KalmanBoxTracker

EVIDENCE_STRONG = "strong_detection"
EVIDENCE_WEAK = "weak_detection"
EVIDENCE_PREDICTION = "motion_prediction"


class ByteTrackAdapter:
    def __init__(self, detection_floor: float = 0.05, high_score_threshold: float = 0.5,
                 new_track_threshold: float = 0.6, first_association_iou_threshold: float = 0.3,
                 second_association_iou_threshold: float = 0.5, track_buffer: int = 5,
                 min_hits: int = 1):
        self.detection_floor = detection_floor
        self.high_score_threshold = high_score_threshold
        self.new_track_threshold = new_track_threshold
        self.first_iou = first_association_iou_threshold
        self.second_iou = second_association_iou_threshold
        self.track_buffer = track_buffer
        self.min_hits = min_hits

        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None

    def update(self, detections: list[dict], timestamp: float = 0.0,
               scene_token: str = "", sample_data_token: str = "",
               method_name: str = "bytetrack", run_id: str = "") -> list[TrackerOutputRecord]:
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp

        detections = [d for d in detections if d.get("confidence", 0.0) >= self.detection_floor]
        high_dets = [d for d in detections if d["confidence"] >= self.high_score_threshold]
        low_dets = [d for d in detections if d["confidence"] < self.high_score_threshold]

        predicted_boxes = [t.predict(dt) for t in self.trackers]
        predicted_classes = [t.class_name for t in self.trackers]

        evidence_this_frame: dict[int, str] = {}

        matches1, unmatched_high, unmatched_trk_idx = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.first_iou)

        for det_idx, trk_idx in matches1:
            det = high_dets[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[trk_idx] = EVIDENCE_STRONG

        remaining_boxes = [predicted_boxes[i] for i in unmatched_trk_idx]
        remaining_classes = [predicted_classes[i] for i in unmatched_trk_idx]
        matches2, unmatched_low, _ = associate_detections_to_trackers(
            low_dets, remaining_boxes, remaining_classes, self.second_iou)

        for det_idx, local_trk_idx in matches2:
            trk_idx = unmatched_trk_idx[local_trk_idx]
            det = low_dets[det_idx]
            self.trackers[trk_idx].update((det["x1"], det["y1"], det["x2"], det["y2"]))
            self.trackers[trk_idx].last_timestamp = timestamp
            self.trackers[trk_idx].last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[trk_idx] = EVIDENCE_WEAK

        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            if det.get("confidence", 0.0) < self.new_track_threshold:
                continue
            new_tracker = KalmanBoxTracker(
                (det["x1"], det["y1"], det["x2"], det["y2"]), det["class"], timestamp)
            new_tracker.hits = 1
            new_tracker.hit_streak = 1
            new_tracker.time_since_update = 0
            new_tracker.last_confidence = det.get("confidence", 0.0)
            evidence_this_frame[len(self.trackers)] = EVIDENCE_STRONG
            self.trackers.append(new_tracker)

        keep_indices = [i for i, t in enumerate(self.trackers) if t.time_since_update <= self.track_buffer]
        kept_trackers = [self.trackers[i] for i in keep_indices]
        kept_evidence = {new_i: evidence_this_frame.get(old_i, EVIDENCE_PREDICTION)
                          for new_i, old_i in enumerate(keep_indices)}
        self.trackers = kept_trackers

        outputs = []
        for i, t in enumerate(self.trackers):
            evidence = kept_evidence[i]
            box = t.current_box()
            state = {
                EVIDENCE_STRONG: "OBSERVED_STRONG",
                EVIDENCE_WEAK: "OBSERVED_WEAK",
                EVIDENCE_PREDICTION: "PREDICTED_HIDDEN",
            }[evidence]
            outputs.append(TrackerOutputRecord(
                scene_token=scene_token, sample_data_token=sample_data_token,
                frame_index=self.frame_count - 1, method_name=method_name, run_id=run_id,
                track_id=t.id, state=state, evidence_source=evidence,
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                detector_confidence=t.last_confidence if evidence != EVIDENCE_PREDICTION else None,
                existence_confidence=1.0, identity_confidence=1.0,  # not modeled by this baseline
                localization_uncertainty=float(np.trace(t.P)),
                memory_age_frames=t.time_since_update,
                memory_age_seconds=t.time_since_update * dt,
                termination_reason=None,
            ))
        return outputs

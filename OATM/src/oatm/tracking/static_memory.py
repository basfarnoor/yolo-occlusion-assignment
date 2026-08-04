"""Method B: static last-seen memory (Assignment 2's central idea, rebuilt
with a real track lifecycle -- Assignment 2 never had birth/expiry, so it
could not be fairly compared against SORT/ByteTrack's buffer-based lifecycle).
A track's box is frozen exactly at its last matched position; it never moves
on its own."""
from __future__ import annotations

from oatm.records import TrackerOutputRecord
from oatm.tracking.association import associate_detections_to_trackers


class _StaticTrack:
    _next_id = 0

    def __init__(self, box: tuple[float, float, float, float], class_name: str):
        self.id = _StaticTrack._next_id
        _StaticTrack._next_id += 1
        self.class_name = class_name
        self.box = box
        self.raw_detection_box: tuple[float, float, float, float] | None = box
        self.time_since_update = 0
        self.age = 0
        self.last_confidence = 0.0
        self.last_update_timestamp = 0.0

    @classmethod
    def reset_id_counter(cls) -> None:
        cls._next_id = 0


class StaticMemoryTracker:
    def __init__(self, high_score_threshold: float = 0.5, track_buffer: int = 5,
                 new_track_threshold: float = 0.6, iou_threshold: float = 0.3):
        self.high_score_threshold = high_score_threshold
        self.track_buffer = track_buffer
        self.new_track_threshold = new_track_threshold
        self.iou_threshold = iou_threshold
        self.trackers: list[_StaticTrack] = []
        self.frame_count = 0

    def update(self, detections: list[dict], timestamp: float = 0.0,
               scene_token: str = "", sample_data_token: str = "",
               method_name: str = "static_memory", run_id: str = "") -> list[TrackerOutputRecord]:
        self.frame_count += 1
        high_dets = [d for d in detections if d.get("confidence", 0.0) >= self.high_score_threshold]

        # "Predict": nothing moves -- the frozen box is exactly last frame's box.
        predicted_boxes = [t.box for t in self.trackers]
        predicted_classes = [t.class_name for t in self.trackers]
        for t in self.trackers:
            t.age += 1
            t.time_since_update += 1
            t.raw_detection_box = None  # no match yet this frame

        matches, unmatched_dets, _ = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.iou_threshold)

        for det_idx, trk_idx in matches:
            det = high_dets[det_idx]
            box = (det["x1"], det["y1"], det["x2"], det["y2"])
            t = self.trackers[trk_idx]
            t.box = box
            t.raw_detection_box = box
            t.time_since_update = 0
            t.last_confidence = det.get("confidence", 0.0)
            t.last_update_timestamp = timestamp

        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            if det.get("confidence", 0.0) < self.new_track_threshold:
                continue
            box = (det["x1"], det["y1"], det["x2"], det["y2"])
            new_track = _StaticTrack(box, det["class"])
            new_track.last_confidence = det.get("confidence", 0.0)
            new_track.last_update_timestamp = timestamp
            self.trackers.append(new_track)

        self.trackers = [t for t in self.trackers if t.time_since_update <= self.track_buffer]

        outputs = []
        for t in self.trackers:
            matched_this_frame = t.time_since_update == 0
            memory_age_seconds = max(0.0, timestamp - t.last_update_timestamp)
            outputs.append(TrackerOutputRecord(
                scene_token=scene_token, sample_data_token=sample_data_token,
                frame_index=self.frame_count - 1, method_name=method_name, run_id=run_id,
                track_id=t.id,
                state="OBSERVED_STRONG" if matched_this_frame else "PREDICTED_HIDDEN",
                evidence_source="strong_detection" if matched_this_frame else "motion_prediction",
                x1=t.box[0], y1=t.box[1], x2=t.box[2], y2=t.box[3],
                detector_confidence=t.last_confidence if matched_this_frame else None,
                existence_confidence=1.0, identity_confidence=1.0,  # not modeled by this baseline
                localization_uncertainty=memory_age_seconds,  # crude proxy: longer frozen = less trustworthy
                memory_age_frames=t.time_since_update, memory_age_seconds=memory_age_seconds,
                termination_reason=None,
            ))
        return outputs

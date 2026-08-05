"""ByteTrack association plus selective, evidence-gated hidden persistence."""
from __future__ import annotations

import numpy as np
from oatm.records import TrackerOutputRecord
from oatm.tracking.association import associate_detections_to_trackers
from oatm.tracking.kalman import KalmanBoxTracker

from oatm_updated.gating import Box, decide_hidden_admission


class _SelectiveTrack:
    def __init__(self, kalman: KalmanBoxTracker) -> None:
        self.kalman = kalman
        self.hidden_frames = 0
        self.hidden_seconds = 0.0
        self.state = "OBSERVED_STRONG"
        self.evidence_source = "strong_detection"
        self.existence_confidence = 1.0
        self.termination_reason: str | None = None
        self.occlusion_support = 0.0


class SelectiveOATMTracker:
    """Preserve ByteTrack association and selectively alter missing-track life."""

    def __init__(
        self,
        detection_floor: float = 0.05,
        high_score_threshold: float = 0.5,
        new_track_threshold: float = 0.6,
        first_association_iou_threshold: float = 0.3,
        second_association_iou_threshold: float = 0.5,
        max_hidden_frames: int = 8,
        ordinary_miss_grace_frames: int = 1,
        min_track_hits_for_occlusion: int = 2,
        occluder_target_coverage_threshold: float = 0.15,
        occluder_min_area_ratio: float = 0.5,
        boundary_margin_px: float = 25.0,
        uncertainty_ceiling: float = 5000.0,
        image_width: float = 1600.0,
        image_height: float = 900.0,
    ) -> None:
        self.detection_floor = detection_floor
        self.high_score_threshold = high_score_threshold
        self.new_track_threshold = new_track_threshold
        self.first_iou = first_association_iou_threshold
        self.second_iou = second_association_iou_threshold
        self.max_hidden_frames = max_hidden_frames
        self.ordinary_miss_grace_frames = ordinary_miss_grace_frames
        self.min_track_hits = min_track_hits_for_occlusion
        self.coverage_threshold = occluder_target_coverage_threshold
        self.min_area_ratio = occluder_min_area_ratio
        self.boundary_margin_px = boundary_margin_px
        self.uncertainty_ceiling = uncertainty_ceiling
        self.image_width = image_width
        self.image_height = image_height
        self.tracks: list[_SelectiveTrack] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None
        self.termination_events: list[dict] = []

    @staticmethod
    def _box(det: dict) -> Box:
        return (det["x1"], det["y1"], det["x2"], det["y2"])

    def _terminate(self, track: _SelectiveTrack, reason: str) -> None:
        track.termination_reason = reason
        self.termination_events.append(
            {"track_id": track.kalman.id, "frame_index": self.frame_count - 1, "reason": reason}
        )

    def update(
        self,
        detections: list[dict],
        timestamp: float = 0.0,
        scene_token: str = "",
        sample_data_token: str = "",
        method_name: str = "selective_oatm",
        run_id: str = "",
    ) -> list[TrackerOutputRecord]:
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp
        detections = [d for d in detections if d.get("confidence", 0.0) >= self.detection_floor]
        high_dets = [d for d in detections if d["confidence"] >= self.high_score_threshold]
        low_dets = [d for d in detections if d["confidence"] < self.high_score_threshold]

        predicted_boxes = [t.kalman.predict(dt) for t in self.tracks]
        predicted_classes = [t.kalman.class_name for t in self.tracks]
        matched_tracks: set[int] = set()

        matches1, unmatched_high, unmatched_tracks = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.first_iou
        )
        for det_idx, track_idx in matches1:
            self._observe(self.tracks[track_idx], high_dets[det_idx], "OBSERVED_STRONG", "strong_detection")
            matched_tracks.add(track_idx)

        remaining_boxes = [predicted_boxes[i] for i in unmatched_tracks]
        remaining_classes = [predicted_classes[i] for i in unmatched_tracks]
        matches2, _, _ = associate_detections_to_trackers(
            low_dets, remaining_boxes, remaining_classes, self.second_iou
        )
        for det_idx, local_idx in matches2:
            track_idx = unmatched_tracks[local_idx]
            self._observe(self.tracks[track_idx], low_dets[det_idx], "OBSERVED_WEAK", "weak_detection")
            matched_tracks.add(track_idx)

        # A visible object remains valid occluder evidence after it is claimed
        # by its own track; otherwise stable occlusion support disappears.
        visible_other_boxes = [self._box(d) for d in high_dets + low_dets]

        survivors: list[_SelectiveTrack] = []
        for idx, track in enumerate(self.tracks):
            if idx in matched_tracks:
                survivors.append(track)
                continue
            track.hidden_frames += 1
            track.hidden_seconds += dt
            decision = decide_hidden_admission(
                predicted_boxes[idx], track.kalman.velocity(), visible_other_boxes,
                track.kalman.hits, track.hidden_frames,
                image_width=self.image_width, image_height=self.image_height,
                boundary_margin_px=self.boundary_margin_px, min_track_hits=self.min_track_hits,
                ordinary_miss_grace_frames=self.ordinary_miss_grace_frames,
                coverage_threshold=self.coverage_threshold, min_area_ratio=self.min_area_ratio,
            )
            uncertainty = float(np.trace(track.kalman.P))
            reason = None
            if decision.predicted_exit:
                reason = "predicted_exit"
            elif track.hidden_frames > self.max_hidden_frames:
                reason = "maximum_hidden_duration"
            elif uncertainty > self.uncertainty_ceiling and track.kalman.hits >= self.min_track_hits:
                reason = "uncertainty_ceiling_exceeded"
            elif not decision.admit_hidden:
                reason = decision.reason
            if reason is not None:
                self._terminate(track, reason)
                continue
            track.state = "PREDICTED_HIDDEN"
            track.evidence_source = "motion_prediction"
            track.occlusion_support = decision.support_score
            track.existence_confidence = max(0.0, 1.0 - track.hidden_frames / (self.max_hidden_frames + 1))
            survivors.append(track)
        self.tracks = survivors

        for det_idx in unmatched_high:
            det = high_dets[det_idx]
            if det["confidence"] < self.new_track_threshold:
                continue
            kalman = KalmanBoxTracker(self._box(det), det["class"], timestamp)
            kalman.hits = 1
            kalman.hit_streak = 1
            kalman.time_since_update = 0
            kalman.last_confidence = det["confidence"]
            self.tracks.append(_SelectiveTrack(kalman))

        return [
            self._output(track, timestamp, scene_token, sample_data_token, method_name, run_id)
            for track in self.tracks
        ]

    @staticmethod
    def _observe(track: _SelectiveTrack, det: dict, state: str, source: str) -> None:
        track.kalman.update((det["x1"], det["y1"], det["x2"], det["y2"]))
        track.kalman.last_confidence = det["confidence"]
        track.hidden_frames = 0
        track.hidden_seconds = 0.0
        track.state = state
        track.evidence_source = source
        track.existence_confidence = 1.0
        track.occlusion_support = 0.0

    def _output(self, track, timestamp, scene_token, sample_data_token, method_name, run_id):
        box = track.kalman.current_box()
        raw = track.kalman.last_raw_box
        return TrackerOutputRecord(
            scene_token=scene_token, sample_data_token=sample_data_token,
            frame_index=self.frame_count - 1, method_name=method_name, run_id=run_id,
            track_id=track.kalman.id, class_name=track.kalman.class_name,
            state=track.state, evidence_source=track.evidence_source,
            x1=box[0], y1=box[1], x2=box[2], y2=box[3],
            raw_detection_x1=raw[0] if raw else None, raw_detection_y1=raw[1] if raw else None,
            raw_detection_x2=raw[2] if raw else None, raw_detection_y2=raw[3] if raw else None,
            detector_confidence=(track.kalman.last_confidence if raw else None),
            existence_confidence=track.existence_confidence, identity_confidence=1.0,
            localization_uncertainty=float(np.trace(track.kalman.P)),
            memory_age_frames=track.hidden_frames, memory_age_seconds=track.hidden_seconds,
            termination_reason=None,
        )

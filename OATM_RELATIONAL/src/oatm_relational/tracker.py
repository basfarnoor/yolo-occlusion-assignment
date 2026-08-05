"""Relational OATM: ByteTrack association plus target--occluder memory."""

from __future__ import annotations

import numpy as np
from oatm.tracking.association import associate_detections_to_trackers
from oatm.tracking.geometry import box_to_state
from oatm.tracking.kalman import KalmanBoxTracker

from oatm_relational.camera_motion import CameraMotionEstimator, apply_affine_to_box, identity_estimate
from oatm_relational.reappearance import associate_reappearances
from oatm_relational.records import RelationalTrackerOutput
from oatm_relational.relations import (
    OcclusionRelation,
    RelationPhase,
    expected_clearance_frames,
    occlusion_probability,
    relation_features,
    relative_geometry,
    target_from_relative_geometry,
)

Box = tuple[float, float, float, float]


class _RelationalTrack:
    def __init__(self, kalman: KalmanBoxTracker) -> None:
        self.kalman = kalman
        self.relation: OcclusionRelation | None = None
        self.hidden_frames = 0
        self.hidden_seconds = 0.0
        self.state = "OBSERVED_STRONG"
        self.evidence_source = "strong_detection"
        self.existence_confidence = 1.0
        self.termination_reason: str | None = None


class RelationalOATMTracker:
    def __init__(
        self,
        detection_floor: float = 0.05,
        high_score_threshold: float = 0.5,
        new_track_threshold: float = 0.6,
        first_association_iou_threshold: float = 0.3,
        second_association_iou_threshold: float = 0.5,
        max_hidden_frames: int = 12,
        ordinary_miss_grace_frames: int = 1,
        min_track_hits: int = 2,
        relation_probability_threshold: float = 0.48,
        target_coverage_threshold: float = 0.12,
        clearance_coverage_threshold: float = 0.05,
        occluder_min_area_ratio: float = 0.5,
        clearance_horizon_frames: int = 12,
        reappearance_grace_frames: int = 4,
        reappearance_score_threshold: float = 0.55,
        uncertainty_ceiling: float = 10000.0,
        anchor_max_residual_ratio: float = 0.75,
        anchor_min_residual_px: float = 20.0,
        anchor_max_scale_ratio: float = 1.25,
        boundary_margin_px: float = 25.0,
        image_width: float = 1600.0,
        image_height: float = 900.0,
        enable_camera_compensation: bool = True,
        enable_clearance_termination: bool = True,
        camera_motion_config: dict | None = None,
    ) -> None:
        self.detection_floor = detection_floor
        self.high_score_threshold = high_score_threshold
        self.new_track_threshold = new_track_threshold
        self.first_iou = first_association_iou_threshold
        self.second_iou = second_association_iou_threshold
        self.max_hidden_frames = max_hidden_frames
        self.ordinary_miss_grace_frames = ordinary_miss_grace_frames
        self.min_track_hits = min_track_hits
        self.relation_probability_threshold = relation_probability_threshold
        self.target_coverage_threshold = target_coverage_threshold
        self.clearance_coverage_threshold = clearance_coverage_threshold
        self.occluder_min_area_ratio = occluder_min_area_ratio
        self.clearance_horizon_frames = clearance_horizon_frames
        self.reappearance_grace_frames = reappearance_grace_frames
        self.reappearance_score_threshold = reappearance_score_threshold
        self.uncertainty_ceiling = uncertainty_ceiling
        self.anchor_max_residual_ratio = anchor_max_residual_ratio
        self.anchor_min_residual_px = anchor_min_residual_px
        self.anchor_max_scale_ratio = anchor_max_scale_ratio
        self.boundary_margin_px = boundary_margin_px
        self.image_width = image_width
        self.image_height = image_height
        self.enable_camera_compensation = enable_camera_compensation
        self.enable_clearance_termination = enable_clearance_termination
        self.motion_estimator = CameraMotionEstimator(**(camera_motion_config or {}))
        self.previous_frame: np.ndarray | None = None
        self.tracks: list[_RelationalTrack] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None
        self.termination_events: list[dict] = []
        self.relation_events: list[dict] = []
        self.last_camera_quality = 0.0

    @staticmethod
    def _box(detection: dict) -> Box:
        return (detection["x1"], detection["y1"], detection["x2"], detection["y2"])

    def _anchor_is_consistent(self, predicted: Box, anchored: Box) -> bool:
        predicted_width = max(predicted[2] - predicted[0], 1e-6)
        predicted_height = max(predicted[3] - predicted[1], 1e-6)
        anchored_width = max(anchored[2] - anchored[0], 1e-6)
        anchored_height = max(anchored[3] - anchored[1], 1e-6)
        predicted_center = ((predicted[0] + predicted[2]) / 2, (predicted[1] + predicted[3]) / 2)
        anchored_center = ((anchored[0] + anchored[2]) / 2, (anchored[1] + anchored[3]) / 2)
        residual = float(
            np.hypot(
                anchored_center[0] - predicted_center[0],
                anchored_center[1] - predicted_center[1],
            )
        )
        diagonal = float(np.hypot(predicted_width, predicted_height))
        residual_limit = max(
            self.anchor_min_residual_px,
            self.anchor_max_residual_ratio * diagonal,
        )
        scale_limit = self.anchor_max_scale_ratio
        width_ratio = anchored_width / predicted_width
        height_ratio = anchored_height / predicted_height
        scale_consistent = all(
            1.0 / scale_limit <= ratio <= scale_limit for ratio in (width_ratio, height_ratio)
        )
        return residual <= residual_limit and scale_consistent

    def _observe(
        self, track: _RelationalTrack, detection: dict, strong: bool, reappearance: bool = False
    ) -> None:
        was_relationally_hidden = track.hidden_frames > 0 and track.relation is not None
        track.kalman.update(self._box(detection))
        track.kalman.last_confidence = detection["confidence"]
        track.hidden_frames = 0
        track.hidden_seconds = 0.0
        track.state = "OBSERVED_STRONG" if strong else "OBSERVED_WEAK"
        track.evidence_source = "strong_detection" if strong else "weak_detection"
        track.existence_confidence = 1.0
        if track.relation is not None:
            if reappearance or was_relationally_hidden:
                track.relation.phase = RelationPhase.RESOLVED
                self.relation_events.append(
                    {
                        "frame_index": self.frame_count - 1,
                        "target_track_id": track.kalman.id,
                        "occluder_track_id": track.relation.occluder_track_id,
                        "event": "resolved",
                    }
                )
            elif track.relation.phase == RelationPhase.RESOLVED:
                track.relation = None

    def _predicted_exit(self, box: Box, velocity: tuple[float, float]) -> bool:
        x1, y1, x2, y2 = box
        vx, vy = velocity
        margin = self.boundary_margin_px
        return (
            (x1 <= margin and vx < 0)
            or (x2 >= self.image_width - margin and vx > 0)
            or (y1 <= margin and vy < 0)
            or (y2 >= self.image_height - margin and vy > 0)
        )

    def _terminate(self, track: _RelationalTrack, reason: str) -> None:
        track.termination_reason = reason
        if track.relation is not None:
            track.relation.phase = RelationPhase.FAILED
        self.termination_events.append(
            {"frame_index": self.frame_count - 1, "track_id": track.kalman.id, "reason": reason}
        )

    def _compensate_predictions(
        self, predictions: list[Box], frame: np.ndarray | None
    ) -> tuple[list[Box], float]:
        estimate = identity_estimate()
        if self.enable_camera_compensation:
            estimate = self.motion_estimator.estimate(self.previous_frame, frame, predictions)
        self.previous_frame = None if frame is None else frame.copy()
        self.last_camera_quality = estimate.quality
        if estimate.used_fallback:
            return predictions, estimate.quality
        compensated = []
        linear = estimate.affine[:, :2]
        for track, box in zip(self.tracks, predictions):
            moved = apply_affine_to_box(box, estimate.affine)
            track.kalman.x[:4] = box_to_state(moved)
            track.kalman.x[4:6] = linear @ track.kalman.x[4:6]
            compensated.append(moved)
        return compensated, estimate.quality

    def update(
        self,
        detections: list[dict],
        timestamp: float = 0.0,
        scene_token: str = "",
        sample_data_token: str = "",
        method_name: str = "oatm_relational",
        run_id: str = "",
        frame: np.ndarray | None = None,
    ) -> list[RelationalTrackerOutput]:
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp
        detections = [item for item in detections if item.get("confidence", 0.0) >= self.detection_floor]
        high = [item for item in detections if item["confidence"] >= self.high_score_threshold]
        low = [item for item in detections if item["confidence"] < self.high_score_threshold]

        for track in self.tracks:
            if track.relation is not None and track.relation.phase == RelationPhase.RESOLVED:
                track.relation = None

        predictions = [track.kalman.predict(dt) for track in self.tracks]
        predictions, camera_quality = self._compensate_predictions(predictions, frame)
        classes = [track.kalman.class_name for track in self.tracks]
        matched_tracks: set[int] = set()

        protected_track_indices = [
            index
            for index, track in enumerate(self.tracks)
            if track.hidden_frames > 0 and track.relation is not None
        ]
        association_indices = [
            index for index in range(len(self.tracks)) if index not in protected_track_indices
        ]
        association_predictions = [predictions[index] for index in association_indices]
        association_classes = [classes[index] for index in association_indices]
        matches1_local, unmatched_high, unmatched_local_tracks = associate_detections_to_trackers(
            high, association_predictions, association_classes, self.first_iou
        )
        for detection_index, local_index in matches1_local:
            track_index = association_indices[local_index]
            self._observe(self.tracks[track_index], high[detection_index], True)
            matched_tracks.add(track_index)

        unmatched_tracks = [association_indices[index] for index in unmatched_local_tracks]
        remaining_boxes = [predictions[index] for index in unmatched_tracks]
        remaining_classes = [classes[index] for index in unmatched_tracks]
        matches2, unmatched_low, unmatched_local = associate_detections_to_trackers(
            low, remaining_boxes, remaining_classes, self.second_iou
        )
        for detection_index, local_index in matches2:
            track_index = unmatched_tracks[local_index]
            self._observe(self.tracks[track_index], low[detection_index], False)
            matched_tracks.add(track_index)
        remaining_track_indices = [
            unmatched_tracks[index] for index in unmatched_local
        ] + protected_track_indices

        remaining_detections = [
            dict(high[index], _source="high", _source_index=index) for index in unmatched_high
        ]
        remaining_detections += [
            dict(low[index], _source="low", _source_index=index) for index in unmatched_low
        ]
        reappearances = associate_reappearances(
            remaining_detections,
            list(range(len(remaining_detections))),
            self.tracks,
            remaining_track_indices,
            predictions,
            self.reappearance_score_threshold,
            {self.tracks[index].kalman.id for index in matched_tracks},
        )
        used_remaining: set[int] = set()
        for candidate in reappearances:
            detection = remaining_detections[candidate.detection_index]
            self._observe(
                self.tracks[candidate.track_index],
                detection,
                detection["confidence"] >= self.high_score_threshold,
                reappearance=True,
            )
            matched_tracks.add(candidate.track_index)
            used_remaining.add(candidate.detection_index)

        used_high = {
            remaining_detections[index]["_source_index"]
            for index in used_remaining
            if remaining_detections[index]["_source"] == "high"
        }
        for detection_index in unmatched_high:
            if detection_index in used_high or high[detection_index]["confidence"] < self.new_track_threshold:
                continue
            detection = high[detection_index]
            kalman = KalmanBoxTracker(self._box(detection), detection["class"], timestamp)
            kalman.hits = 1
            kalman.hit_streak = 1
            kalman.time_since_update = 0
            kalman.last_confidence = detection["confidence"]
            self.tracks.append(_RelationalTrack(kalman))
            predictions.append(self._box(detection))
            matched_tracks.add(len(self.tracks) - 1)

        visible_indices = [
            index
            for index, track in enumerate(self.tracks)
            if index in matched_tracks and track.state in {"OBSERVED_STRONG", "OBSERVED_WEAK"}
        ]
        survivors: list[_RelationalTrack] = []
        for index, track in enumerate(self.tracks):
            if index in matched_tracks:
                survivors.append(track)
                continue
            track.hidden_frames += 1
            track.hidden_seconds += dt
            predicted = predictions[index]
            reason = None
            if self._predicted_exit(predicted, track.kalman.velocity()):
                reason = "predicted_exit"
            elif track.kalman.hits < self.min_track_hits:
                reason = "immature_track"
            else:
                candidates = []
                for occluder_index in visible_indices:
                    occluder = self.tracks[occluder_index]
                    if occluder.kalman.id == track.kalman.id:
                        continue
                    if track.relation is not None and track.relation.occluder_track_id != occluder.kalman.id:
                        continue
                    occluder_box = occluder.kalman.current_box()
                    features = relation_features(
                        predicted,
                        occluder_box,
                        track.kalman.velocity(),
                        occluder.kalman.velocity(),
                        track.kalman.hits,
                        camera_quality,
                    )
                    probability = occlusion_probability(features)
                    area_ratio = max(
                        0.0, (occluder_box[2] - occluder_box[0]) * (occluder_box[3] - occluder_box[1])
                    ) / max((predicted[2] - predicted[0]) * (predicted[3] - predicted[1]), 1e-6)
                    if (
                        features.coverage >= self.target_coverage_threshold
                        and area_ratio >= self.occluder_min_area_ratio
                        and probability >= self.relation_probability_threshold
                    ):
                        anchored = predicted
                        if track.relation is not None:
                            anchored = target_from_relative_geometry(
                                track.relation.relative_geometry, occluder_box
                            )
                            if not self._anchor_is_consistent(predicted, anchored):
                                continue
                        clearance = expected_clearance_frames(
                            predicted,
                            occluder_box,
                            track.kalman.velocity(),
                            occluder.kalman.velocity(),
                            self.clearance_coverage_threshold,
                            self.clearance_horizon_frames,
                        )
                        candidates.append((probability, features.coverage, clearance, occluder, anchored))
                if candidates:
                    probability, coverage, clearance, occluder, anchored = max(
                        candidates, key=lambda item: item[0]
                    )
                    if track.relation is None:
                        track.relation = OcclusionRelation(
                            track.kalman.id,
                            occluder.kalman.id,
                            self.frame_count - 1,
                            self.frame_count - 1,
                            RelationPhase.FORMING,
                            probability,
                            coverage,
                            clearance,
                            relative_geometry(predicted, occluder_box),
                        )
                        self.relation_events.append(
                            {
                                "frame_index": self.frame_count - 1,
                                "target_track_id": track.kalman.id,
                                "occluder_track_id": occluder.kalman.id,
                                "event": "formed",
                            }
                        )
                    else:
                        track.relation.update_support(self.frame_count - 1, probability, coverage, clearance)
                    track.kalman.x[:4] = box_to_state(anchored)
                    predictions[index] = anchored
                    predicted = anchored
                elif track.relation is not None:
                    track.relation.phase = RelationPhase.CLEARING
                    track.relation.clearing_age_frames += 1

                if track.relation is None and track.hidden_frames > self.ordinary_miss_grace_frames:
                    reason = "insufficient_occlusion_evidence"
                elif (
                    self.enable_clearance_termination
                    and track.relation is not None
                    and track.relation.phase == RelationPhase.CLEARING
                    and track.relation.clearing_age_frames > self.reappearance_grace_frames
                ):
                    reason = "failed_expected_reappearance"
                elif track.hidden_frames > self.max_hidden_frames:
                    reason = "maximum_hidden_duration"
                elif float(np.trace(track.kalman.P)) > self.uncertainty_ceiling:
                    reason = "uncertainty_ceiling_exceeded"

            if reason is not None:
                self._terminate(track, reason)
                continue
            track.state = "PREDICTED_HIDDEN"
            track.evidence_source = "motion_prediction"
            relation_probability_value = track.relation.probability if track.relation else 0.2
            duration_decay = max(0.0, 1.0 - track.hidden_frames / (self.max_hidden_frames + 1))
            track.existence_confidence = relation_probability_value * duration_decay
            survivors.append(track)
        self.tracks = survivors

        return [
            self._output(track, scene_token, sample_data_token, method_name, run_id) for track in self.tracks
        ]

    def _output(
        self,
        track: _RelationalTrack,
        scene_token: str,
        sample_data_token: str,
        method_name: str,
        run_id: str,
    ) -> RelationalTrackerOutput:
        box = track.kalman.current_box()
        raw = track.kalman.last_raw_box
        relation = track.relation
        return RelationalTrackerOutput(
            scene_token=scene_token,
            sample_data_token=sample_data_token,
            frame_index=self.frame_count - 1,
            method_name=method_name,
            run_id=run_id,
            track_id=track.kalman.id,
            class_name=track.kalman.class_name,
            state=track.state,
            evidence_source=track.evidence_source,
            x1=box[0],
            y1=box[1],
            x2=box[2],
            y2=box[3],
            raw_detection_x1=raw[0] if raw else None,
            raw_detection_y1=raw[1] if raw else None,
            raw_detection_x2=raw[2] if raw else None,
            raw_detection_y2=raw[3] if raw else None,
            detector_confidence=track.kalman.last_confidence if raw else None,
            existence_confidence=track.existence_confidence,
            identity_confidence=1.0,
            localization_uncertainty=float(np.trace(track.kalman.P)),
            memory_age_frames=track.hidden_frames,
            memory_age_seconds=track.hidden_seconds,
            termination_reason=None,
            occluder_track_id=relation.occluder_track_id if relation else None,
            occlusion_probability=relation.probability if relation else 0.0,
            expected_clearance_frames=relation.expected_clearance_frames if relation else None,
            relation_phase=relation.phase.value if relation else None,
            camera_motion_quality=self.last_camera_quality,
        )

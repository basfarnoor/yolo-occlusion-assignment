"""Task 12: OATM MVP (Method F) plus an optional third association stage --
appearance-based reconnection -- for the ablation comparing motion-only,
appearance-only, and dual (motion+appearance) reconnection. Everything else
(two-stage IoU association, motion, the state machine, adaptive termination)
is identical to `oatm.tracking.oatm_adapter.OATMTracker`; this file is a
separate, self-contained adapter (matching this project's existing one-
adapter-per-method pattern) rather than a subclass, since the reconnection
stage has to interleave in the middle of the per-frame loop, not wrap it.

`appearance_mode`:
  - "motion_only": the reconnection stage is skipped entirely -- behaves
    exactly like Task 11's `OATMTracker` (this IS the ablation's baseline arm).
  - "appearance_only": reconnection matches purely on cosine similarity,
    ignoring predicted location entirely.
  - "dual": reconnection requires appearance similarity AND a location-
    consistency gate.

Detections must carry a precomputed `"embedding"` key (np.ndarray) for
appearance_mode != "motion_only" -- this tracker never touches raw images
itself, matching the rest of this project's "detector/embedder runs once,
tracker only ever sees precomputed per-frame observations" boundary.
"""
from __future__ import annotations

import numpy as np

from oatm.memory.appearance import AppearanceAnchor, is_eligible_for_anchor_update
from oatm.memory.confidence import ExistenceConfidenceTracker
from oatm.occlusion.evidence import EvidenceInputs, classify_event
from oatm.occlusion.state_machine import (
    EVENT_INSUFFICIENT_EVIDENCE,
    EVENT_STRONG_DETECTION,
    EVENT_WEAK_DETECTION,
    OBSERVED_STRONG,
    OBSERVED_WEAK,
    PREDICTED_HIDDEN,
    TrackStateMachine,
)
from oatm.occlusion.termination import TerminationInputs, evaluate_termination
from oatm.records import TrackerOutputRecord
from oatm.tracking.association import associate_detections_to_trackers
from oatm.tracking.geometry import iou
from oatm.tracking.kalman import KalmanBoxTracker
from oatm.tracking.reconnection import HiddenTrackCandidate, resolve_reconnection

IMAGE_WIDTH = 1600.0
IMAGE_HEIGHT = 900.0
VALID_APPEARANCE_MODES = ("motion_only", "appearance_only", "dual")


class _OATMAppearanceTrack:
    def __init__(self, kalman: KalmanBoxTracker):
        self.kalman = kalman
        self.state_machine = TrackStateMachine()
        self.state_machine.birth()
        self.confidence_tracker = ExistenceConfidenceTracker()
        self.recent_confidences: list[float] = [kalman.last_confidence]
        self.frames_since_last_evidence = 0
        self.termination_reason: str | None = None
        self.appearance_anchor = AppearanceAnchor()

    @property
    def confidence_trend_declining(self) -> bool:
        if len(self.recent_confidences) < 2:
            return False
        return self.recent_confidences[-1] < self.recent_confidences[-2]


class OATMAppearanceTracker:
    def __init__(self, detection_floor: float = 0.05, high_score_threshold: float = 0.5,
                 new_track_threshold: float = 0.6, first_association_iou_threshold: float = 0.3,
                 second_association_iou_threshold: float = 0.5,
                 existence_floor: float = 0.05, uncertainty_ceiling: float = 500.0,
                 confidence_beta: float = 0.15, confidence_alpha: float = 0.01,
                 boundary_margin_px: float = 25.0, max_grace_frames_without_evidence: int = 1,
                 occluder_iou_threshold: float = 0.02,
                 appearance_mode: str = "motion_only",
                 appearance_similarity_threshold: float = 0.7,
                 reconnection_location_iou_threshold: float = 0.05,
                 min_anchor_box_area: float = 400.0, anchor_boundary_margin_px: float = 5.0):
        if appearance_mode not in VALID_APPEARANCE_MODES:
            raise ValueError(
                f"appearance_mode must be one of {VALID_APPEARANCE_MODES}, got {appearance_mode!r}"
            )
        self.detection_floor = detection_floor
        self.high_score_threshold = high_score_threshold
        self.new_track_threshold = new_track_threshold
        self.first_iou = first_association_iou_threshold
        self.second_iou = second_association_iou_threshold
        self.existence_floor = existence_floor
        self.uncertainty_ceiling = uncertainty_ceiling
        self.confidence_beta = confidence_beta
        self.confidence_alpha = confidence_alpha
        self.boundary_margin_px = boundary_margin_px
        self.max_grace_frames_without_evidence = max_grace_frames_without_evidence
        self.occluder_iou_threshold = occluder_iou_threshold
        self.appearance_mode = appearance_mode
        self.appearance_similarity_threshold = appearance_similarity_threshold
        self.reconnection_location_iou_threshold = reconnection_location_iou_threshold
        self.min_anchor_box_area = min_anchor_box_area
        self.anchor_boundary_margin_px = anchor_boundary_margin_px

        self.tracks: list[_OATMAppearanceTrack] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None

    def _has_occluder_overlap(
        self, predicted_box, unclaimed_boxes: list[tuple[float, float, float, float]]
    ) -> bool:
        return any(iou(predicted_box, b) >= self.occluder_iou_threshold for b in unclaimed_boxes)

    def _maybe_update_anchor(self, track: _OATMAppearanceTrack, det: dict, state: str) -> None:
        embedding = det.get("embedding")
        if embedding is None:
            return
        box = (det["x1"], det["y1"], det["x2"], det["y2"])
        eligible = is_eligible_for_anchor_update(
            state, box, IMAGE_WIDTH, IMAGE_HEIGHT, self.min_anchor_box_area, self.anchor_boundary_margin_px,
        )
        track.appearance_anchor.update(embedding, eligible)

    def update(self, detections: list[dict], timestamp: float = 0.0,
               scene_token: str = "", sample_data_token: str = "",
               method_name: str = "oatm_appearance", run_id: str = "") -> list[TrackerOutputRecord]:
        self.frame_count += 1
        dt = 1.0 if self.last_timestamp is None else max(timestamp - self.last_timestamp, 1e-6)
        self.last_timestamp = timestamp

        detections = [d for d in detections if d.get("confidence", 0.0) >= self.detection_floor]
        high_dets = [d for d in detections if d["confidence"] >= self.high_score_threshold]
        low_dets = [d for d in detections if d["confidence"] < self.high_score_threshold]

        predicted_boxes = [t.kalman.predict(dt) for t in self.tracks]
        predicted_classes = [t.kalman.class_name for t in self.tracks]

        matches1, unmatched_high, unmatched_trk_idx = associate_detections_to_trackers(
            high_dets, predicted_boxes, predicted_classes, self.first_iou)
        evidence_received_this_frame: list[_OATMAppearanceTrack] = []
        for det_idx, trk_idx in matches1:
            det = high_dets[det_idx]
            track = self.tracks[trk_idx]
            track.kalman.update((det["x1"], det["y1"], det["x2"], det["y2"]))
            track.kalman.last_confidence = det["confidence"]
            track.state_machine.transition(EVENT_STRONG_DETECTION)
            self._maybe_update_anchor(track, det, OBSERVED_STRONG)
            evidence_received_this_frame.append(track)

        remaining_boxes = [predicted_boxes[i] for i in unmatched_trk_idx]
        remaining_classes = [predicted_classes[i] for i in unmatched_trk_idx]
        matches2, unmatched_low, unmatched_remaining_local = associate_detections_to_trackers(
            low_dets, remaining_boxes, remaining_classes, self.second_iou)
        for det_idx, local_idx in matches2:
            trk_idx = unmatched_trk_idx[local_idx]
            det = low_dets[det_idx]
            track = self.tracks[trk_idx]
            track.kalman.update((det["x1"], det["y1"], det["x2"], det["y2"]))
            track.kalman.last_confidence = det["confidence"]
            track.state_machine.transition(EVENT_WEAK_DETECTION)
            # OBSERVED_WEAK is never eligible for an anchor update -- called
            # anyway (as a no-op) so the ineligible-write path stays exercised.
            self._maybe_update_anchor(track, det, OBSERVED_WEAK)
            evidence_received_this_frame.append(track)

        claimed_low_indices = {det_idx for det_idx, _ in matches2}
        unclaimed_boxes = [
            (d["x1"], d["y1"], d["x2"], d["y2"])
            for i, d in enumerate(low_dets) if i not in claimed_low_indices
        ] + [
            (d["x1"], d["y1"], d["x2"], d["y2"])
            for i, d in enumerate(high_dets) if i in unmatched_high
        ]

        still_unmatched_trk_idx = [unmatched_trk_idx[i] for i in unmatched_remaining_local]
        for trk_idx in still_unmatched_trk_idx:
            track = self.tracks[trk_idx]
            box = predicted_boxes[trk_idx]
            track.frames_since_last_evidence += 1

            inputs = EvidenceInputs(
                matched_detection_confidence=None, high_score_threshold=self.high_score_threshold,
                predicted_box=box, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
                predicted_velocity=track.kalman.velocity(),
                has_occluder_overlap=self._has_occluder_overlap(box, unclaimed_boxes),
                confidence_trend_declining=track.confidence_trend_declining,
                frames_since_last_evidence=track.frames_since_last_evidence,
                boundary_margin_px=self.boundary_margin_px,
                max_grace_frames_without_evidence=self.max_grace_frames_without_evidence,
            )
            event = classify_event(inputs)
            new_state = track.state_machine.transition(event)

            if new_state == PREDICTED_HIDDEN:
                uncertainty = float(np.trace(track.kalman.P))
                existence_conf = track.confidence_tracker.decay(dt, uncertainty)
                if track.kalman.hits >= 2:
                    decision = evaluate_termination(TerminationInputs(
                        predicted_exit=False, impossible_occluder_relationship=False,
                        localization_uncertainty=uncertainty,
                        uncertainty_ceiling=self.uncertainty_ceiling,
                        existence_confidence=existence_conf, existence_floor=self.existence_floor,
                        failed_expected_reappearance=False,
                    ))
                    if decision.should_terminate:
                        track.state_machine.transition(EVENT_INSUFFICIENT_EVIDENCE)
                        track.termination_reason = decision.reason

        # Third stage (Task 12): try to reconnect any track that is STILL
        # PREDICTED_HIDDEN this frame to an otherwise-unclaimed high-score
        # detection, using its frozen appearance anchor -- before the birth
        # loop runs, so a successful reconnection claims that detection
        # instead of a duplicate new track being born from it.
        reconnect_claimed_high_idx: set[int] = set()
        if self.appearance_mode != "motion_only":
            hidden_candidates = []
            hidden_track_refs: list[_OATMAppearanceTrack] = []
            for trk_idx in still_unmatched_trk_idx:
                track = self.tracks[trk_idx]
                if track.state_machine.state != PREDICTED_HIDDEN:
                    continue  # this frame it went LOST/EXITED instead -- terminal, cannot reconnect
                hidden_candidates.append(HiddenTrackCandidate(
                    class_name=track.kalman.class_name,
                    predicted_box=predicted_boxes[trk_idx],
                    appearance_embedding=track.appearance_anchor.embedding,
                ))
                hidden_track_refs.append(track)

            reconnect_pool = [high_dets[i] for i in unmatched_high]
            reconnect_matches = resolve_reconnection(
                hidden_candidates, reconnect_pool, mode=self.appearance_mode,
                appearance_similarity_threshold=self.appearance_similarity_threshold,
                location_iou_threshold=self.reconnection_location_iou_threshold,
            )
            for pool_idx, hidden_local_idx in reconnect_matches:
                det = reconnect_pool[pool_idx]
                track = hidden_track_refs[hidden_local_idx]
                track.kalman.update((det["x1"], det["y1"], det["x2"], det["y2"]))
                track.kalman.last_confidence = det["confidence"]
                track.state_machine.transition(EVENT_STRONG_DETECTION)
                self._maybe_update_anchor(track, det, OBSERVED_STRONG)
                evidence_received_this_frame.append(track)
                reconnect_claimed_high_idx.add(unmatched_high[pool_idx])

        for det_idx in unmatched_high:
            if det_idx in reconnect_claimed_high_idx:
                continue
            det = high_dets[det_idx]
            if det["confidence"] < self.new_track_threshold:
                continue
            new_kalman = KalmanBoxTracker(
                (det["x1"], det["y1"], det["x2"], det["y2"]), det["class"], timestamp
            )
            new_kalman.hits = 1
            new_kalman.hit_streak = 1
            new_kalman.time_since_update = 0
            new_kalman.last_confidence = det["confidence"]
            new_track = _OATMAppearanceTrack(new_kalman)
            self._maybe_update_anchor(new_track, det, OBSERVED_STRONG)
            self.tracks.append(new_track)
            evidence_received_this_frame.append(new_track)

        for track in evidence_received_this_frame:
            track.recent_confidences.append(track.kalman.last_confidence)
            track.recent_confidences = track.recent_confidences[-3:]
            track.frames_since_last_evidence = 0
            track.confidence_tracker.reset()

        keep = [t for t in self.tracks if not t.state_machine.is_terminal]
        self.tracks = keep

        outputs = []
        for track in self.tracks:
            state = track.state_machine.state
            box = track.kalman.current_box()
            uncertainty = float(np.trace(track.kalman.P))
            outputs.append(TrackerOutputRecord(
                scene_token=scene_token, sample_data_token=sample_data_token,
                frame_index=self.frame_count - 1, method_name=method_name, run_id=run_id,
                track_id=track.kalman.id, class_name=track.kalman.class_name,
                state=state, evidence_source=track.state_machine.evidence_source,
                x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                raw_detection_x1=track.kalman.last_raw_box[0] if track.kalman.last_raw_box else None,
                raw_detection_y1=track.kalman.last_raw_box[1] if track.kalman.last_raw_box else None,
                raw_detection_x2=track.kalman.last_raw_box[2] if track.kalman.last_raw_box else None,
                raw_detection_y2=track.kalman.last_raw_box[3] if track.kalman.last_raw_box else None,
                detector_confidence=(
                    track.kalman.last_confidence if state in (OBSERVED_STRONG, OBSERVED_WEAK) else None
                ),
                existence_confidence=track.confidence_tracker.existence_confidence,
                identity_confidence=1.0,  # not modeled beyond gated association in this ablation
                localization_uncertainty=uncertainty,
                memory_age_frames=track.frames_since_last_evidence,
                memory_age_seconds=track.frames_since_last_evidence * dt,
                termination_reason=track.termination_reason,
            ))
        return outputs

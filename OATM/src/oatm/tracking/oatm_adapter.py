"""Method F: the OATM MVP tracker. Combines everything built so far into one
coherent per-frame loop:

  - Two-stage BYTE-style association (Task 6 / bytetrack_adapter.py's logic).
  - Timestamp-aware constant-velocity Kalman motion (Task 8).
  - The exact five-state machine, driven by a camera-only evidence gate
    (Task 9) -- occlusion vs. exit vs. ordinary miss are distinguished
    causally, using only this frame's OTHER detections and this track's own
    history, never a future frame or a privileged label.
  - Adaptive existence-confidence decay and priority-ordered, single-reason
    anti-ghost termination (Task 10), layered on top of the state machine:
    a track the evidence gate still calls "plausibly hidden" can still be
    cut off if its existence confidence has decayed past the frozen floor or
    its uncertainty has exceeded the frozen ceiling.

This is the MVP: no appearance memory, no ego-motion compensation (both
wait for their own later, approved phases).
"""
from __future__ import annotations

import numpy as np

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

IMAGE_WIDTH = 1600.0
IMAGE_HEIGHT = 900.0


class _OATMTrack:
    def __init__(self, kalman: KalmanBoxTracker):
        self.kalman = kalman
        self.state_machine = TrackStateMachine()
        self.state_machine.birth()
        self.confidence_tracker = ExistenceConfidenceTracker()
        self.recent_confidences: list[float] = [kalman.last_confidence]
        self.frames_since_last_evidence = 0
        self.termination_reason: str | None = None

    @property
    def confidence_trend_declining(self) -> bool:
        if len(self.recent_confidences) < 2:
            return False
        return self.recent_confidences[-1] < self.recent_confidences[-2]


class OATMTracker:
    def __init__(self, detection_floor: float = 0.05, high_score_threshold: float = 0.5,
                 new_track_threshold: float = 0.6, first_association_iou_threshold: float = 0.3,
                 second_association_iou_threshold: float = 0.5,
                 existence_floor: float = 0.05, uncertainty_ceiling: float = 500.0,
                 confidence_beta: float = 0.15, confidence_alpha: float = 0.01,
                 boundary_margin_px: float = 25.0, max_grace_frames_without_evidence: int = 1,
                 occluder_iou_threshold: float = 0.02):
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

        self.tracks: list[_OATMTrack] = []
        self.frame_count = 0
        self.last_timestamp: float | None = None

    def _has_occluder_overlap(
        self, predicted_box, unclaimed_boxes: list[tuple[float, float, float, float]]
    ) -> bool:
        return any(iou(predicted_box, b) >= self.occluder_iou_threshold for b in unclaimed_boxes)

    def update(self, detections: list[dict], timestamp: float = 0.0,
               scene_token: str = "", sample_data_token: str = "",
               method_name: str = "oatm_mvp", run_id: str = "") -> list[TrackerOutputRecord]:
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
        # Existing tracks matched THIS frame -- transitioned immediately, so
        # there is never any ambiguity later between "just birthed" and
        # "pre-existing track matched again" (the bug this replaced: both
        # cases used to collide in one dict keyed by post-birth indices).
        evidence_received_this_frame: list[_OATMTrack] = []
        for det_idx, trk_idx in matches1:
            det = high_dets[det_idx]
            track = self.tracks[trk_idx]
            track.kalman.update((det["x1"], det["y1"], det["x2"], det["y2"]))
            track.kalman.last_confidence = det["confidence"]
            track.state_machine.transition(EVENT_STRONG_DETECTION)
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
            evidence_received_this_frame.append(track)

        # Detections nothing claimed -- available as camera-derived, causal
        # "possible occluder" evidence for tracks that went unmatched (never
        # using privileged depth/visibility, only this frame's own boxes).
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
                # A track's Kalman filter starts with a deliberately huge
                # velocity covariance (trace ~30000, far past uncertainty_
                # ceiling) that only collapses once a SECOND real detection
                # runs the correction step (kalman.hits >= 2) -- see Task 10's
                # own warm-up workaround in termination_study.py. Without this
                # guard, any track occluded on the very frame after birth --
                # a common case, not an edge case -- would be killed by the
                # uncertainty ceiling instantly, regardless of existence_floor,
                # since that check outranks existence_floor in the priority
                # order. The grace period in classify_event() still bounds
                # this exemption: once frames_since_last_evidence exceeds
                # max_grace_frames_without_evidence, insufficient_evidence
                # fires on its own, so this is not an unlimited loophole.
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

        for det_idx in unmatched_high:
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
            new_track = _OATMTrack(new_kalman)  # birth() inside __init__ already sets OBSERVED_STRONG
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
                identity_confidence=1.0,  # not modeled beyond IoU association in this MVP
                localization_uncertainty=uncertainty,
                memory_age_frames=track.frames_since_last_evidence,
                memory_age_seconds=track.frames_since_last_evidence * dt,
                termination_reason=track.termination_reason,
            ))
        return outputs

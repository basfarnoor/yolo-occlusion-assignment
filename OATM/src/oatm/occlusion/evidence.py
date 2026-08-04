"""Phase 7 (Task 9): a small, rule-based, CAMERA-ONLY classifier that turns
raw per-frame signals into exactly one state-machine event. Low confidence
alone is weak support, not proof of occlusion -- notice `classify_event`
never treats a low-confidence *detection* as occlusion evidence by itself;
weak detections go straight to OBSERVED_WEAK. Occlusion evidence only ever
applies when there is NO current detection at all.

No privileged nuScenes label (visibility, depth, instance identity) is used
here -- every signal is something the online system could compute for
itself from current and earlier camera frames and its own tracker state.
A real camera-derived depth-ordering proxy and full trajectory-consistency
check are deferred to a later phase (ego-motion/appearance); this MVP gate
uses foreground-overlap, confidence trend, image-boundary/outward-motion,
and elapsed time only, exactly as scoped for Task 9.
"""
from __future__ import annotations

from dataclasses import dataclass

from oatm.occlusion.state_machine import (
    EVENT_INSUFFICIENT_EVIDENCE,
    EVENT_OCCLUSION_EVIDENCE,
    EVENT_PREDICTED_EXIT,
    EVENT_STRONG_DETECTION,
    EVENT_WEAK_DETECTION,
)


@dataclass
class EvidenceInputs:
    matched_detection_confidence: float | None  # None if nothing matched this track this frame
    high_score_threshold: float
    predicted_box: tuple[float, float, float, float]
    image_width: float
    image_height: float
    predicted_velocity: tuple[float, float]  # px/second, from the Kalman state
    has_occluder_overlap: bool  # a DIFFERENT current detection overlaps the predicted box
    confidence_trend_declining: bool  # recent confidence was falling before it disappeared
    frames_since_last_evidence: int
    boundary_margin_px: float = 25.0
    max_grace_frames_without_evidence: int = 1


def _is_predicted_exit(inputs: EvidenceInputs) -> bool:
    x1, y1, x2, y2 = inputs.predicted_box
    vx, vy = inputs.predicted_velocity
    m = inputs.boundary_margin_px
    outward_left = x1 <= m and vx < 0
    outward_right = x2 >= inputs.image_width - m and vx > 0
    outward_top = y1 <= m and vy < 0
    outward_bottom = y2 >= inputs.image_height - m and vy > 0
    return outward_left or outward_right or outward_top or outward_bottom


def classify_event(inputs: EvidenceInputs) -> str:
    """Returns exactly one of the five state-machine events. Causal: uses
    only this frame's matched detection (if any) and this track's own
    history/prediction -- never a future frame."""
    if inputs.matched_detection_confidence is not None:
        if inputs.matched_detection_confidence >= inputs.high_score_threshold:
            return EVENT_STRONG_DETECTION
        return EVENT_WEAK_DETECTION

    if _is_predicted_exit(inputs):
        return EVENT_PREDICTED_EXIT

    if inputs.has_occluder_overlap or inputs.confidence_trend_declining:
        return EVENT_OCCLUSION_EVIDENCE

    if inputs.frames_since_last_evidence <= inputs.max_grace_frames_without_evidence:
        # A brief, explicit benefit of the doubt -- NOT unlimited: once this
        # grace period is spent, an ordinary miss with no supporting evidence
        # goes to LOST, not PREDICTED_HIDDEN. This is what stops the method
        # from keeping every missing detection as hidden forever.
        return EVENT_OCCLUSION_EVIDENCE

    return EVENT_INSUFFICIENT_EVIDENCE

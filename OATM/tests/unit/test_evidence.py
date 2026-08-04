"""Required Task 9 fixtures: true occlusion, field-of-view exit, ordinary
detector miss, poor visibility, false initial track, and compatible/
incompatible reappearance -- checked against the rule-based, camera-only
evidence classifier."""
from oatm.occlusion.evidence import EvidenceInputs, classify_event
from oatm.occlusion.state_machine import (
    EVENT_INSUFFICIENT_EVIDENCE,
    EVENT_OCCLUSION_EVIDENCE,
    EVENT_PREDICTED_EXIT,
    EVENT_STRONG_DETECTION,
    EVENT_WEAK_DETECTION,
)
from oatm.tracking.association import associate_detections_to_trackers

IMG_W, IMG_H = 1600.0, 900.0


def _inputs(**overrides):
    base = dict(
        matched_detection_confidence=None, high_score_threshold=0.5,
        predicted_box=(700.0, 400.0, 740.0, 440.0), image_width=IMG_W, image_height=IMG_H,
        predicted_velocity=(0.0, 0.0), has_occluder_overlap=False,
        confidence_trend_declining=False, frames_since_last_evidence=1,
        boundary_margin_px=25.0, max_grace_frames_without_evidence=1,
    )
    base.update(overrides)
    return EvidenceInputs(**base)


def test_true_occlusion_with_a_visible_occluder_is_occlusion_evidence():
    """Fixture: a real, currently-detected object overlaps the track's
    predicted location -- the textbook true-occlusion case."""
    inputs = _inputs(has_occluder_overlap=True, frames_since_last_evidence=4)
    assert classify_event(inputs) == EVENT_OCCLUSION_EVIDENCE


def test_field_of_view_exit_with_outward_motion_at_the_boundary():
    """Fixture: the predicted box sits at the image edge, moving further
    outward -- must be classified as an exit, never occlusion, even if an
    occluder happens to be nearby (exit is checked first)."""
    inputs = _inputs(
        predicted_box=(5.0, 400.0, 45.0, 440.0),  # x1=5, within the 25px boundary margin
        predicted_velocity=(-30.0, 0.0),  # moving further left/outward
        has_occluder_overlap=True,  # even with an occluder present
    )
    assert classify_event(inputs) == EVENT_PREDICTED_EXIT


def test_object_near_boundary_but_moving_inward_is_not_an_exit():
    """Being near the edge alone must not trigger exit -- only outward
    motion at the boundary does."""
    inputs = _inputs(
        predicted_box=(5.0, 400.0, 45.0, 440.0),
        predicted_velocity=(30.0, 0.0),  # moving back INTO frame
        has_occluder_overlap=True, frames_since_last_evidence=2,
    )
    assert classify_event(inputs) != EVENT_PREDICTED_EXIT


def test_ordinary_detector_miss_with_no_occluder_becomes_insufficient_evidence():
    """Fixture: object vanished mid-frame, nothing overlaps its predicted
    box, confidence was not trending down, and the grace period has passed --
    this is an ordinary detector miss, not occlusion, and must not be kept
    hidden forever."""
    inputs = _inputs(has_occluder_overlap=False, confidence_trend_declining=False,
                       frames_since_last_evidence=5, max_grace_frames_without_evidence=1)
    assert classify_event(inputs) == EVENT_INSUFFICIENT_EVIDENCE


def test_poor_visibility_low_confidence_detection_is_still_just_a_weak_detection():
    """Fixture: the object IS detected, just barely above the floor -- low
    confidence alone is weak support, never occlusion. A present (if weak)
    detection is a detection, not a hidden state."""
    inputs = _inputs(matched_detection_confidence=0.06, high_score_threshold=0.5)
    assert classify_event(inputs) == EVENT_WEAK_DETECTION


def test_high_confidence_detection_is_a_strong_detection_event():
    inputs = _inputs(matched_detection_confidence=0.9, high_score_threshold=0.5)
    assert classify_event(inputs) == EVENT_STRONG_DETECTION


def test_false_initial_track_goes_to_insufficient_evidence_quickly():
    """Fixture: a track was just born (possibly from a spurious detection)
    and immediately goes missing with no supporting evidence -- once the
    (short) grace period is spent, it must not be propped up as hidden."""
    inputs = _inputs(has_occluder_overlap=False, confidence_trend_declining=False,
                       frames_since_last_evidence=2, max_grace_frames_without_evidence=1)
    assert classify_event(inputs) == EVENT_INSUFFICIENT_EVIDENCE


def test_compatible_reappearance_is_simply_reported_as_a_detection_event():
    """Fixture: a detection reappears near a hidden track's predicted
    location. The state machine doesn't need special-casing for this --
    ANY current matched detection, regardless of prior state, is reported
    as a normal strong/weak-detection event; the transition table (tested
    separately) is what allows PREDICTED_HIDDEN -> OBSERVED_STRONG."""
    inputs = _inputs(matched_detection_confidence=0.85, frames_since_last_evidence=3)
    assert classify_event(inputs) == EVENT_STRONG_DETECTION


def test_incompatible_reappearance_is_never_matched_by_the_association_layer():
    """Fixture: a detection appears near a track's predicted box, but it is
    the WRONG CLASS -- the association layer (tested in Task 6) must reject
    it, so no confidence ever reaches the evidence classifier for it, and
    the track correctly continues down the missing-evidence path instead of
    being handed an incompatible match."""
    predicted_boxes = [(700.0, 400.0, 740.0, 440.0)]
    predicted_classes = ["car"]
    detections = [{"class": "person", "x1": 700.0, "y1": 400.0, "x2": 740.0, "y2": 440.0}]

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, predicted_boxes, predicted_classes, iou_threshold=0.3)

    assert matches == [], "an incompatible class must never be treated as a valid reappearance"
    assert unmatched_trks == [0]

    # With no compatible match, the track's evidence classification proceeds
    # on the "no detection this frame" branch, exactly like an ordinary miss.
    inputs = _inputs(matched_detection_confidence=None, frames_since_last_evidence=1)
    assert classify_event(inputs) in (EVENT_OCCLUSION_EVIDENCE, EVENT_INSUFFICIENT_EVIDENCE)

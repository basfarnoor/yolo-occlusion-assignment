import pytest
from oatm.tracking.kalman import KalmanBoxTracker

from oatm_relational import RelationalOATMTracker


def det(x1, cls="car", confidence=0.9, width=40.0, y1=100.0, height=40.0):
    return {
        "x1": float(x1),
        "y1": y1,
        "x2": float(x1 + width),
        "y2": y1 + height,
        "class": cls,
        "confidence": confidence,
    }


def mature(tracker):
    first = tracker.update([det(100)], timestamp=0.0)
    target_id = first[0].track_id
    tracker.update([det(105)], timestamp=1.0)
    return target_id


def test_relation_supports_long_occlusion_and_exposes_diagnostics():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    for frame_index in range(2, 9):
        rows = tracker.update(
            [det(100 + frame_index * 5, cls="truck", width=90, y1=90, height=70)],
            timestamp=float(frame_index),
        )
        target = next(row for row in rows if row.track_id == target_id)
        assert target.state == "PREDICTED_HIDDEN"
        assert target.occluder_track_id is not None
        assert target.occlusion_probability > 0.0


def test_unsupported_miss_expires_after_grace():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    assert any(row.track_id == target_id for row in tracker.update([], timestamp=2.0))
    assert not any(row.track_id == target_id for row in tracker.update([], timestamp=3.0))
    assert tracker.termination_events[-1]["reason"] == "insufficient_occlusion_evidence"


def test_strong_mature_track_receives_conditional_extended_grace():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(
        ordinary_miss_grace_frames=1,
        mature_miss_grace_frames=3,
        mature_miss_min_hits=2,
        mature_miss_min_confidence=0.7,
        uncertainty_ceiling=1e9,
    )
    target_id = mature(tracker)
    for timestamp in (2.0, 3.0, 4.0):
        assert any(row.track_id == target_id for row in tracker.update([], timestamp=timestamp))
    assert not any(row.track_id == target_id for row in tracker.update([], timestamp=5.0))


def test_weak_last_observation_does_not_receive_extended_grace():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(
        ordinary_miss_grace_frames=1,
        mature_miss_grace_frames=3,
        mature_miss_min_hits=2,
        mature_miss_min_confidence=0.7,
        uncertainty_ceiling=1e9,
    )
    target_id = mature(tracker)
    tracker.update([det(110, confidence=0.55)], timestamp=2.0)
    assert any(row.track_id == target_id for row in tracker.update([], timestamp=3.0))
    assert not any(row.track_id == target_id for row in tracker.update([], timestamp=4.0))


def test_invalid_conditional_grace_is_rejected():
    with pytest.raises(ValueError, match="must not be shorter"):
        RelationalOATMTracker(
            ordinary_miss_grace_frames=2,
            mature_miss_grace_frames=1,
        )


def test_dormant_track_is_silent_and_reactivates_on_weak_detection():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(
        ordinary_miss_grace_frames=1,
        dormant_reactivation_frames=3,
        uncertainty_ceiling=1e9,
    )
    target_id = mature(tracker)
    first_miss = tracker.update([], timestamp=2.0)
    assert any(row.track_id == target_id for row in first_miss)

    dormant_output = tracker.update([], timestamp=3.0)
    assert not any(row.track_id == target_id for row in dormant_output)
    assert any(track.kalman.id == target_id and track.state == "DORMANT" for track in tracker.tracks)

    reactivated = tracker.update([det(115, confidence=0.3)], timestamp=4.0)
    target = next(row for row in reactivated if row.track_id == target_id)
    assert target.state == "OBSERVED_WEAK"
    assert target.evidence_source == "weak_detection"


def test_dormant_track_rejects_detection_below_strict_threshold():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(
        ordinary_miss_grace_frames=1,
        dormant_reactivation_frames=1,
        dormant_reappearance_score_threshold=1.0,
        uncertainty_ceiling=1e9,
    )
    target_id = mature(tracker)
    tracker.update([], timestamp=2.0)
    tracker.update([], timestamp=3.0)

    rows = tracker.update([det(115, confidence=0.3)], timestamp=4.0)

    assert not any(row.track_id == target_id for row in rows)
    assert all(track.kalman.id != target_id for track in tracker.tracks)
    assert tracker.termination_events[-1]["reason"] == "insufficient_occlusion_evidence"


def test_dormant_track_eventually_terminates_without_emitting_predictions():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(
        ordinary_miss_grace_frames=1,
        dormant_reactivation_frames=2,
        uncertainty_ceiling=1e9,
    )
    target_id = mature(tracker)
    tracker.update([], timestamp=2.0)
    for timestamp in (3.0, 4.0):
        rows = tracker.update([], timestamp=timestamp)
        assert not any(row.track_id == target_id for row in rows)
    tracker.update([], timestamp=5.0)
    assert all(track.kalman.id != target_id for track in tracker.tracks)
    assert tracker.termination_events[-1]["reason"] == "insufficient_occlusion_evidence"


def test_invalid_dormant_reappearance_threshold_is_rejected():
    with pytest.raises(ValueError, match="dormant_reappearance_score_threshold"):
        RelationalOATMTracker(dormant_reappearance_score_threshold=1.01)


def test_partial_boundary_truncation_is_not_an_exit():
    tracker = RelationalOATMTracker(
        image_width=200.0,
        image_height=100.0,
        boundary_margin_px=25.0,
        exit_visible_fraction_threshold=0.05,
    )
    assert not tracker._predicted_exit((-10.0, 20.0, 30.0, 60.0), (-5.0, 0.0))
    assert not tracker._predicted_exit((-39.0, 20.0, 1.0, 60.0), (5.0, 0.0))


def test_nearly_outside_box_moving_outward_is_an_exit():
    tracker = RelationalOATMTracker(
        image_width=200.0,
        image_height=100.0,
        boundary_margin_px=25.0,
        exit_visible_fraction_threshold=0.05,
    )
    assert tracker._predicted_exit((-39.0, 20.0, 1.0, 60.0), (-5.0, 0.0))


def test_invalid_exit_visible_fraction_is_rejected():
    with pytest.raises(ValueError, match="exit_visible_fraction_threshold"):
        RelationalOATMTracker(exit_visible_fraction_threshold=1.1)


def test_failed_expected_reappearance_terminates_relation():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(reappearance_grace_frames=1, uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    tracker.update([det(110, cls="truck", width=90, y1=90, height=70)], timestamp=2.0)
    tracker.update([det(115, cls="truck", width=90, y1=90, height=70)], timestamp=3.0)
    tracker.update([], timestamp=4.0)
    rows = tracker.update([], timestamp=5.0)
    assert not any(row.track_id == target_id for row in rows)
    target_events = [event for event in tracker.termination_events if event["track_id"] == target_id]
    assert target_events[-1]["reason"] == "failed_expected_reappearance"


def test_relation_aware_stage_recovers_original_identity():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    for frame_index in range(2, 6):
        tracker.update([det(110, cls="truck", width=90, y1=90, height=70)], timestamp=float(frame_index))
    rows = tracker.update([det(145)], timestamp=6.0)
    recovered = next(row for row in rows if row.class_name == "car")
    assert recovered.track_id == target_id
    assert recovered.relation_phase == "RESOLVED"


def test_visible_occluder_blocks_same_class_identity_hijack():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    tracker.update([det(110, cls="truck", width=90, y1=90, height=70)], timestamp=2.0)
    rows = tracker.update(
        [
            det(115, cls="truck", width=90, y1=90, height=70),
            det(115, cls="car"),
        ],
        timestamp=3.0,
    )
    target = next(row for row in rows if row.track_id == target_id)
    distractor = next(row for row in rows if row.class_name == "car" and row.track_id != target_id)
    assert target.state == "PREDICTED_HIDDEN"
    assert distractor.state == "OBSERVED_STRONG"


def test_inconsistent_occluder_motion_clears_without_dragging_target():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    tracker.update([det(110, cls="truck", width=90, y1=90, height=70)], timestamp=2.0)
    rows = tracker.update([det(155, cls="truck", width=90, y1=90, height=70)], timestamp=3.0)
    target = next(row for row in rows if row.track_id == target_id)
    center_x = (target.x1 + target.x2) / 2
    assert target.state == "PREDICTED_HIDDEN"
    assert target.relation_phase == "CLEARING"
    assert center_x < 150.0


def test_resolved_relation_is_cleared_before_a_new_hidden_episode():
    KalmanBoxTracker.reset_id_counter()
    tracker = RelationalOATMTracker(uncertainty_ceiling=1e9)
    target_id = mature(tracker)
    for frame_index in range(2, 6):
        tracker.update(
            [det(110, cls="truck", width=90, y1=90, height=70)],
            timestamp=float(frame_index),
        )
    recovered = tracker.update([det(145)], timestamp=6.0)
    assert next(row for row in recovered if row.track_id == target_id).relation_phase == "RESOLVED"
    hidden_again = tracker.update([det(150, cls="truck", width=90, y1=90, height=70)], timestamp=7.0)
    target = next(row for row in hidden_again if row.track_id == target_id)
    assert target.relation_phase == "FORMING"


def test_new_instance_is_required_at_scene_boundary():
    KalmanBoxTracker.reset_id_counter()
    first = RelationalOATMTracker().update([det(100)], scene_token="scene-a")
    second = RelationalOATMTracker().update([det(100)], scene_token="scene-b")
    assert first[0].track_id != second[0].track_id

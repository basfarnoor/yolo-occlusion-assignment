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

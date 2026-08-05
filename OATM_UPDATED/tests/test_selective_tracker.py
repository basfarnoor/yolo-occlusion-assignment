from oatm.tracking.kalman import KalmanBoxTracker

from oatm_updated import SelectiveOATMTracker


def det(x1, y1, x2, y2, cls="car", confidence=0.9):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "class": cls, "confidence": confidence}


def mature_tracker(tracker):
    first = tracker.update([det(100, 100, 140, 140)], timestamp=0.0)
    track_id = first[0].track_id
    tracker.update([det(105, 100, 145, 140)], timestamp=1.0)
    return track_id


def test_occluder_support_extends_beyond_bytetrack_buffer():
    KalmanBoxTracker.reset_id_counter()
    tracker = SelectiveOATMTracker(max_hidden_frames=8, uncertainty_ceiling=1e9)
    track_id = mature_tracker(tracker)
    for frame in range(2, 8):
        rows = tracker.update([det(100, 90, 190, 160, cls="truck")], timestamp=float(frame))
        target = [r for r in rows if r.track_id == track_id]
        assert target and target[0].state == "PREDICTED_HIDDEN"


def test_ordinary_miss_expires_after_bounded_grace():
    KalmanBoxTracker.reset_id_counter()
    tracker = SelectiveOATMTracker(ordinary_miss_grace_frames=1, uncertainty_ceiling=1e9)
    track_id = mature_tracker(tracker)
    assert any(r.track_id == track_id for r in tracker.update([], timestamp=2.0))
    assert not any(r.track_id == track_id for r in tracker.update([], timestamp=3.0))
    assert tracker.termination_events[-1]["reason"] == "insufficient_occlusion_evidence"


def test_scene_instances_do_not_share_tracks():
    KalmanBoxTracker.reset_id_counter()
    first = SelectiveOATMTracker().update([det(100, 100, 140, 140)], scene_token="a")
    second = SelectiveOATMTracker().update([det(100, 100, 140, 140)], scene_token="b")
    assert first[0].scene_token == "a"
    assert second[0].scene_token == "b"
    assert first[0].track_id != second[0].track_id

"""Tests for src/run_methods.py -- Task 8's fair-comparison requirements.
ByteTrack paper reference: Zhang et al., ECCV 2022 (arxiv.org/abs/2110.06864)."""
import inspect

import bytetrack_tracker
import sort_tracker
from bytetrack_tracker import ByteTrackTracker
from kalman_box_tracker import KalmanBoxTracker
from run_methods import new_bytetrack_tracker, new_sort_tracker, run_tracker_over_clip
from sort_tracker import SortTracker

CFG = {
    "detector": {"detection_floor": 0.05},
    "tracker": {
        "high_score_threshold": 0.5, "new_track_threshold": 0.6,
        "first_association_iou_threshold": 0.3, "second_association_iou_threshold": 0.5,
        "track_buffer": 3,
    },
    "sort_baseline": {"max_age": 3, "iou_threshold": 0.3},
}


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_sort_and_bytetrack_receive_byte_for_byte_identical_raw_detections():
    """Required test #12."""
    KalmanBoxTracker.reset_id_counter()
    shared_detections = [_det(0, 0, 40, 40, conf=0.9), _det(100, 0, 140, 40, conf=0.2)]

    sort = new_sort_tracker(CFG)
    byte = new_bytetrack_tracker(CFG)

    # Call both with the literal same list object -- no per-method copy or
    # filtering happens outside the tracker classes themselves.
    sort_outputs = sort.update(shared_detections, timestamp=0.0)
    byte_outputs = byte.update(shared_detections, timestamp=0.0)

    assert shared_detections == [_det(0, 0, 40, 40, conf=0.9), _det(100, 0, 140, 40, conf=0.2)], (
        "neither tracker may mutate the shared input list")
    # Both birth exactly one track from the shared high-score detection.
    assert len(sort_outputs) == 1
    assert len(byte_outputs) == 1


def test_ground_truth_is_structurally_inaccessible_from_tracker_interfaces():
    """Required test #13: the online tracker's public interface must have no
    parameter, attribute, or import that could smuggle in privileged
    nuScenes ground truth."""
    for module in (sort_tracker, bytetrack_tracker):
        source = inspect.getsource(module)
        assert "projected_ground_truth" not in source
        assert "import projection" not in source
        assert "from projection" not in source

    sort_sig = inspect.signature(SortTracker.update)
    byte_sig = inspect.signature(ByteTrackTracker.update)
    for sig in (sort_sig, byte_sig):
        params = set(sig.parameters) - {"self"}
        assert params == {"detections", "timestamp"}, (
            f"tracker.update() must only accept detections and timestamp, got {params}")


def test_frames_from_different_scenes_never_share_a_track():
    """Required test #14: a single tracker instance must be used for exactly
    one clip/scene. Running two clips through two separate tracker instances
    (as run_methods.run_three_methods_over_clip always does, one call per
    clip) must never produce overlapping track IDs claiming to be the same
    object."""
    KalmanBoxTracker.reset_id_counter()

    clip_a_frames = [
        {"frame_number": 1, "timestamp": 0.0, "detections": [_det(0, 0, 40, 40, conf=0.9)]},
        {"frame_number": 2, "timestamp": 0.1, "detections": [_det(5, 0, 45, 40, conf=0.9)]},
    ]
    clip_b_frames = [
        {"frame_number": 1, "timestamp": 0.0, "detections": [_det(0, 0, 40, 40, conf=0.9)]},
        {"frame_number": 2, "timestamp": 0.1, "detections": [_det(5, 0, 45, 40, conf=0.9)]},
    ]

    tracker_a = new_bytetrack_tracker(CFG)
    outputs_a = run_tracker_over_clip(tracker_a, clip_a_frames)

    tracker_b = new_bytetrack_tracker(CFG)  # a fresh, independent tracker for clip B
    outputs_b = run_tracker_over_clip(tracker_b, clip_b_frames)

    ids_a = {o.track_id for outs in outputs_a.values() for o in outs}
    ids_b = {o.track_id for outs in outputs_b.values() for o in outs}
    assert ids_a.isdisjoint(ids_b), (
        "two different clips/scenes must never produce overlapping track IDs -- "
        "each clip must be run through its own fresh tracker instance")

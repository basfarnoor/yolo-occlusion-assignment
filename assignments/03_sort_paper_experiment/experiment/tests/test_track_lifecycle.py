"""Tests for src/sort_tracker.py. Paper: Bewley et al., ICIP 2016 (SORT)."""
from kalman_box_tracker import KalmanBoxTracker
from sort_tracker import SortTracker


def _det(x1, y1, x2, y2, cls="car"):
    return {"class": cls, "confidence": 0.9, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_track_survives_the_allowed_missing_frames():
    KalmanBoxTracker.reset_id_counter()
    tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.3)

    # Establish one track.
    outputs = tracker.update([_det(0, 0, 40, 40)])
    track_id = outputs[0].track_id

    # Miss detections for exactly max_age frames -- track must still be alive.
    for _ in range(3):
        outputs = tracker.update([])
    ids_alive = [o.track_id for o in outputs]
    assert track_id in ids_alive, "track should still exist within max_age missing frames"


def test_track_expires_after_max_age():
    KalmanBoxTracker.reset_id_counter()
    tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.3)

    outputs = tracker.update([_det(0, 0, 40, 40)])
    track_id = outputs[0].track_id

    # One frame beyond max_age of consecutive misses -- track must be gone.
    for _ in range(4):
        outputs = tracker.update([])
    ids_alive = [o.track_id for o in outputs]
    assert track_id not in ids_alive, "track should be removed once time_since_update exceeds max_age"


def test_same_input_and_seed_produce_the_same_output():
    def run_once():
        KalmanBoxTracker.reset_id_counter()
        tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.3)
        frames = [
            [_det(0, 0, 40, 40)],
            [_det(10, 0, 50, 40)],
            [_det(20, 0, 60, 40)],
            [],
            [_det(40, 0, 80, 40)],
        ]
        all_outputs = []
        for frame_dets in frames:
            outs = tracker.update(frame_dets)
            all_outputs.append([(o.track_id, o.box, o.matched_this_frame) for o in outs])
        return all_outputs

    result_a = run_once()
    result_b = run_once()
    assert result_a == result_b, "identical deterministic input must produce identical output"

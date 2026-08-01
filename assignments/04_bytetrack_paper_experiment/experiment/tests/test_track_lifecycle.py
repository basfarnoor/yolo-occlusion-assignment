"""Tests for src/sort_tracker.py (Method B baseline).
ByteTrack paper reference: Zhang et al., ECCV 2022 (arxiv.org/abs/2110.06864)."""
from kalman_box_tracker import KalmanBoxTracker
from sort_tracker import SortTracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_PREDICTION


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_track_survives_the_allowed_missing_frames():
    KalmanBoxTracker.reset_id_counter()
    tracker = SortTracker(track_buffer=3, iou_threshold=0.3)

    outputs = tracker.update([_det(0, 0, 40, 40)])
    track_id = outputs[0].track_id

    for _ in range(3):
        outputs = tracker.update([])
    ids_alive = [o.track_id for o in outputs]
    assert track_id in ids_alive, "track should still exist within track_buffer missing frames"


def test_track_expires_after_track_buffer():
    KalmanBoxTracker.reset_id_counter()
    tracker = SortTracker(track_buffer=3, iou_threshold=0.3)

    outputs = tracker.update([_det(0, 0, 40, 40)])
    track_id = outputs[0].track_id

    for _ in range(4):
        outputs = tracker.update([])
    ids_alive = [o.track_id for o in outputs]
    assert track_id not in ids_alive, "track should be removed once time_since_update exceeds track_buffer"


def test_low_score_detections_are_never_matched_or_used_to_birth():
    """SortTracker (Method B) has no second association round at all -- a
    low-score detection must be completely invisible to it."""
    KalmanBoxTracker.reset_id_counter()
    tracker = SortTracker(high_score_threshold=0.5, track_buffer=3)
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)])
    track_id = outputs[0].track_id

    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.2)])  # low-score, same location
    assert outputs[0].track_id == track_id
    assert outputs[0].evidence_source == EVIDENCE_PREDICTION, (
        "a low-score detection must not update the track -- it should stay in prediction state")


def test_same_input_and_seed_produce_the_same_output():
    def run_once():
        KalmanBoxTracker.reset_id_counter()
        tracker = SortTracker(track_buffer=3)
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
            all_outputs.append([(o.track_id, o.box, o.evidence_source) for o in outs])
        return all_outputs

    assert run_once() == run_once()

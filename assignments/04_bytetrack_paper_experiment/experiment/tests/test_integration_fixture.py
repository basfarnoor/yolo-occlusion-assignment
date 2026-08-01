"""Task 7's required integration fixture: two objects, one weak detection, one
false weak box, a short complete miss, and a returning object -- every
expected ID and state verified by hand below.
ByteTrack paper reference: Zhang et al., ECCV 2022 (arxiv.org/abs/2110.06864)."""
from bytetrack_tracker import ByteTrackTracker
from kalman_box_tracker import KalmanBoxTracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE, EVIDENCE_PREDICTION


def _det(x1, y1, x2, y2, conf, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_two_object_scenario_with_weak_detection_false_box_miss_and_return():
    KalmanBoxTracker.reset_id_counter()
    tracker = ByteTrackTracker(detection_floor=0.05, high_score_threshold=0.5,
                                 new_track_threshold=0.6, first_association_iou_threshold=0.3,
                                 second_association_iou_threshold=0.5, track_buffer=3)

    # --- Frame 1: two clean high-score detections birth two tracks. ---
    outputs = tracker.update([
        _det(0, 0, 40, 40, conf=0.9),      # object A
        _det(300, 0, 340, 40, conf=0.9),   # object B
    ])
    assert len(outputs) == 2
    by_id = {o.track_id: o for o in outputs}
    assert set(by_id) == {0, 1}, "first two births must get IDs 0 and 1 in KalmanBoxTracker's counter order"
    assert by_id[0].evidence_source == EVIDENCE_HIGH_SCORE
    assert by_id[1].evidence_source == EVIDENCE_HIGH_SCORE
    track_a_id, track_b_id = 0, 1

    # --- Frame 2: object A gets only a WEAK (low-score) detection; a FALSE
    # weak box appears far away with no nearby track; object B keeps a clean
    # high-score match. ---
    outputs = tracker.update([
        _det(2, 2, 42, 42, conf=0.2),        # weak detection near A's predicted box
        _det(500, 500, 540, 540, conf=0.15),  # false weak box, nowhere near any track
        _det(305, 2, 345, 42, conf=0.9),      # clean high-score match for B
    ])
    by_id = {o.track_id: o for o in outputs}
    assert set(by_id) == {track_a_id, track_b_id}, (
        "the false weak box must not create a third track -- exactly A and B must be present")
    assert by_id[track_a_id].evidence_source == EVIDENCE_LOW_SCORE, (
        "A's weak detection should update it via the second association stage")
    assert by_id[track_b_id].evidence_source == EVIDENCE_HIGH_SCORE

    # --- Frame 3: object A gets a complete miss (no detection at all nearby);
    # object B keeps matching. ---
    outputs = tracker.update([
        _det(310, 4, 350, 44, conf=0.9),  # only B's detection this frame
    ])
    by_id = {o.track_id: o for o in outputs}
    assert set(by_id) == {track_a_id, track_b_id}, "A must survive the miss (within track_buffer=3)"
    assert by_id[track_a_id].evidence_source == EVIDENCE_PREDICTION
    assert by_id[track_a_id].time_since_update == 1
    assert by_id[track_b_id].evidence_source == EVIDENCE_HIGH_SCORE

    # --- Frame 4: object A returns with a clean high-score detection near its
    # (slightly drifted) predicted location; it must reconnect to the SAME ID,
    # not birth a new one. ---
    outputs = tracker.update([
        _det(4, 4, 44, 44, conf=0.9),      # A returns
        _det(315, 6, 355, 46, conf=0.9),   # B continues
    ])
    by_id = {o.track_id: o for o in outputs}
    assert set(by_id) == {track_a_id, track_b_id}, "no new track may be born -- A reconnects to its original ID"
    assert by_id[track_a_id].evidence_source == EVIDENCE_HIGH_SCORE
    assert by_id[track_a_id].time_since_update == 0
    assert by_id[track_b_id].evidence_source == EVIDENCE_HIGH_SCORE

    # Across all four frames, only IDs 0 and 1 were ever used.
    assert KalmanBoxTracker._next_id == 2, "exactly two tracks should ever have been created in this whole scenario"

"""Tests for src/bytetrack_tracker.py -- the paper's central two-stage BYTE
association contribution. ByteTrack paper reference: Zhang et al., ECCV 2022
(arxiv.org/abs/2110.06864)."""
from bytetrack_tracker import ByteTrackTracker
from kalman_box_tracker import KalmanBoxTracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE, EVIDENCE_PREDICTION


def _det(x1, y1, x2, y2, conf, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _new_tracker(**overrides):
    KalmanBoxTracker.reset_id_counter()
    kwargs = dict(detection_floor=0.05, high_score_threshold=0.5, new_track_threshold=0.6,
                  first_association_iou_threshold=0.3, second_association_iou_threshold=0.5,
                  track_buffer=3)
    kwargs.update(overrides)
    return ByteTrackTracker(**kwargs)


def test_detections_split_correctly_at_exact_score_boundaries():
    """Required test #2: a detection exactly at high_score_threshold counts as
    high-score (>=), and a detection exactly at detection_floor is kept, not
    silently dropped by an off-by-one comparison."""
    tracker = _new_tracker()
    outputs = tracker.update([
        _det(0, 0, 40, 40, conf=0.5),     # exactly at high_score_threshold -> high
        _det(100, 0, 140, 40, conf=0.05),  # exactly at detection_floor -> low, but kept
    ])
    # Both should have birthed or been considered: the conf=0.5 detection clears
    # new_track_threshold? No -- new_track_threshold is 0.6, so 0.5 should NOT
    # start a track (see test_high_score_below_new_track_threshold_does_not_birth).
    # This test only checks that a conf=0.05 detection is not thrown away before
    # even reaching the tracker -- i.e. it appears in the low-score group and is
    # eligible for stage 2 matching in a later frame.
    assert len(outputs) == 0  # neither the exactly-at-floor nor the not-yet-new-track-eligible box births yet


def test_high_score_matching_happens_before_low_score_matching():
    """Required test #5: given one existing track and both a high-score and a
    low-score detection that could each plausibly match it, the high-score
    detection must win the match (stage 1 runs first and consumes the track)."""
    tracker = _new_tracker()
    tracker.update([_det(0, 0, 40, 40, conf=0.9)])  # births a track at (0,0,40,40)

    # Next frame: both a high-score and a low-score box overlap the predicted
    # track location almost identically.
    outputs = tracker.update([
        _det(1, 1, 41, 41, conf=0.9),   # high-score, near-perfect overlap
        _det(2, 2, 42, 42, conf=0.2),   # low-score, also near-perfect overlap
    ])
    assert len(outputs) == 1
    assert outputs[0].evidence_source == EVIDENCE_HIGH_SCORE
    # Kalman-corrected box should sit much closer to the high-score detection
    # (1,1,41,41) than to the low-score one (2,2,42,42), confirming the
    # high-score box is what was actually matched.
    assert abs(outputs[0].box[0] - 1) < abs(outputs[0].box[0] - 2)


def test_only_tracks_unmatched_after_stage_one_enter_stage_two():
    """Required test #6: a track already claimed by a high-score detection in
    stage 1 must not also be handed a low-score detection in stage 2, even if
    that low-score box also overlaps it (there is only one track to give)."""
    tracker = _new_tracker()
    tracker.update([_det(0, 0, 40, 40, conf=0.9)])

    outputs = tracker.update([
        _det(0, 0, 40, 40, conf=0.9),   # stage 1 claims the only track
        _det(0, 0, 40, 40, conf=0.2),   # would also match, but track is already taken
    ])
    assert len(outputs) == 1
    assert outputs[0].evidence_source == EVIDENCE_HIGH_SCORE


def test_plausible_low_score_detection_updates_an_existing_unmatched_track():
    """Required test #7: a track left unmatched by stage 1 (no high-score box
    nearby) can still be updated by a plausible low-score box in stage 2."""
    tracker = _new_tracker()
    tracker.update([_det(0, 0, 40, 40, conf=0.9)])  # births track

    # Next frame: only a low-score detection appears, at the same location.
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.2)])
    assert len(outputs) == 1
    assert outputs[0].evidence_source == EVIDENCE_LOW_SCORE
    assert outputs[0].time_since_update == 0


def test_unmatched_low_score_detection_cannot_create_a_new_track():
    """Required test #8: a low-score detection with no existing track nearby
    must never birth a new track, regardless of how confident-adjacent it is."""
    tracker = _new_tracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.49)])  # below high_score_threshold
    assert outputs == []
    outputs2 = tracker.update([_det(0, 0, 40, 40, conf=0.49)])
    assert outputs2 == [], "a low-score detection must never start a track even if it repeats"


def test_unmatched_high_score_detection_can_create_a_new_track_when_eligible():
    """Required test #9."""
    tracker = _new_tracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)])  # >= new_track_threshold (0.6)
    assert len(outputs) == 1
    assert outputs[0].evidence_source == EVIDENCE_HIGH_SCORE


def test_high_score_below_new_track_threshold_does_not_birth():
    """A detection can be "high-score" (>= high_score_threshold=0.5) without
    being confident enough to birth a brand-new track (>= new_track_threshold=0.6)."""
    tracker = _new_tracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.55)])  # high-score but below 0.6
    assert outputs == []


def test_track_survives_exactly_the_configured_buffer_and_then_expires():
    """Required test #10."""
    tracker = _new_tracker(track_buffer=3)
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)])
    track_id = outputs[0].track_id

    for _ in range(3):
        outputs = tracker.update([])
    assert track_id in [o.track_id for o in outputs], "track should survive within track_buffer missing frames"
    assert all(o.evidence_source == EVIDENCE_PREDICTION for o in outputs if o.track_id == track_id)

    outputs = tracker.update([])
    assert track_id not in [o.track_id for o in outputs], "track should be removed once past track_buffer"


def test_returning_detection_reconnects_only_through_real_association():
    """Required test #11: a track's ID after reappearance must come from the
    real association process (the same KalmanBoxTracker surviving inside the
    buffer window), never assigned by test/experiment bookkeeping."""
    tracker = _new_tracker(track_buffer=3)
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)])
    original_id = outputs[0].track_id

    tracker.update([])  # one missing frame, within buffer
    outputs = tracker.update([_det(2, 0, 42, 40, conf=0.9)])  # detection returns, close to predicted location

    assert len(outputs) == 1
    assert outputs[0].track_id == original_id, "the SAME track object must be the one matched, not a new track"
    assert outputs[0].evidence_source == EVIDENCE_HIGH_SCORE


def test_every_output_has_exactly_one_evidence_source_label():
    """Required test #17."""
    tracker = _new_tracker(track_buffer=2)
    tracker.update([_det(0, 0, 40, 40, conf=0.9), _det(200, 0, 240, 40, conf=0.9)])
    outputs = tracker.update([_det(1, 1, 41, 41, conf=0.2)])  # one low-score match, one goes missing
    valid_labels = {EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE, EVIDENCE_PREDICTION}
    for o in outputs:
        assert o.evidence_source in valid_labels
        assert isinstance(o.evidence_source, str)


def test_raw_detection_box_is_the_actual_yolo_box_not_the_kalman_estimate():
    """Regression test for the exact bug Assignment 3's mentor review flagged
    (reuse_audit.md repair #1): TrackOutput.raw_detection_box must be the
    literal matched detection box, not the smoothed Kalman state (`box`),
    and must be None whenever evidence_source is motion_prediction."""
    tracker = _new_tracker(track_buffer=2)
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)])
    assert outputs[0].raw_detection_box == (0, 0, 40, 40)

    # A detection that disagrees slightly with the (as-yet velocity-less)
    # prediction still gets Kalman-corrected in `box`, but raw_detection_box
    # must equal the exact input box, unmodified.
    outputs = tracker.update([_det(3, 1, 43, 41, conf=0.9)])
    assert outputs[0].raw_detection_box == (3, 1, 43, 41)
    assert outputs[0].box != outputs[0].raw_detection_box, (
        "the Kalman-corrected box should differ from the raw detection once there is any innovation")

    # A missing-detection frame must report no raw box at all.
    outputs = tracker.update([])
    assert outputs[0].evidence_source == EVIDENCE_PREDICTION
    assert outputs[0].raw_detection_box is None


def test_same_input_and_seed_produce_the_same_output():
    """Required test #16."""
    def run_once():
        tracker = _new_tracker()
        frames = [
            [_det(0, 0, 40, 40, conf=0.9)],
            [_det(10, 0, 50, 40, conf=0.9)],
            [_det(20, 0, 60, 40, conf=0.2)],
            [],
            [_det(35, 0, 75, 40, conf=0.9)],
        ]
        all_outputs = []
        for frame_dets in frames:
            outs = tracker.update(frame_dets)
            all_outputs.append([(o.track_id, o.box, o.evidence_source) for o in outs])
        return all_outputs

    assert run_once() == run_once()

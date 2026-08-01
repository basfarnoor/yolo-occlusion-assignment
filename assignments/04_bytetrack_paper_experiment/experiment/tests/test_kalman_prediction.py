"""Tests for src/kalman_box_tracker.py.
ByteTrack paper reference: Zhang et al., ECCV 2022 (arxiv.org/abs/2110.06864)."""
from kalman_box_tracker import KalmanBoxTracker
from geometry import box_center


def test_box_moving_steadily_keeps_moving_when_detection_is_missing():
    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")

    for step in range(1, 4):
        tracker.predict(dt=1.0)
        tracker.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))

    cx_before, _ = box_center(tracker.current_box())
    predicted_box = tracker.predict(dt=1.0)
    cx_after, _ = box_center(predicted_box)

    assert cx_after > cx_before, "prediction should continue moving in the learned direction"


def test_new_observation_corrects_an_imperfect_prediction():
    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")

    predicted_box = tracker.predict(dt=1.0)
    predicted_cx, _ = box_center(predicted_box)

    true_box = (100.0, 0.0, 140.0, 40.0)
    true_cx, _ = box_center(true_box)

    tracker.update(true_box)
    corrected_cx, _ = box_center(tracker.current_box())

    dist_before = abs(predicted_cx - true_cx)
    dist_after = abs(corrected_cx - true_cx)
    assert dist_after < dist_before, "correction should move the estimate closer to the true observation"


def test_timestamp_aware_prediction_scales_displacement_with_dt():
    """Required test #15: timestamp-aware prediction behaves correctly for
    unequal frame intervals. A dt=2.0 prediction should move roughly twice as
    far as a dt=1.0 prediction, given the same learned velocity -- this is
    the repair of Assignment 3's fixed one-step-per-call Kalman transition
    (reuse_audit.md, required repair #7)."""
    def build_tracker_with_learned_velocity() -> KalmanBoxTracker:
        KalmanBoxTracker.reset_id_counter()
        t = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")
        for step in range(1, 5):
            t.predict(dt=1.0)
            t.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))
        return t

    tracker_a = build_tracker_with_learned_velocity()
    cx_start_a, _ = box_center(tracker_a.current_box())
    box_dt1 = tracker_a.predict(dt=1.0)
    cx_dt1, _ = box_center(box_dt1)
    displacement_dt1 = cx_dt1 - cx_start_a

    tracker_b = build_tracker_with_learned_velocity()
    cx_start_b, _ = box_center(tracker_b.current_box())
    box_dt2 = tracker_b.predict(dt=2.0)
    cx_dt2, _ = box_center(box_dt2)
    displacement_dt2 = cx_dt2 - cx_start_b

    assert displacement_dt1 > 0
    assert abs(displacement_dt2 - 2.0 * displacement_dt1) < 1e-6, (
        f"dt=2.0 should displace exactly twice as far as dt=1.0 under a constant-velocity model "
        f"(got {displacement_dt2} vs 2x{displacement_dt1}={2 * displacement_dt1})")


def test_zero_or_negative_dt_does_not_crash_or_move_backward():
    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")
    for step in range(1, 4):
        tracker.predict(dt=1.0)
        tracker.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))
    cx_before, _ = box_center(tracker.current_box())
    box_after = tracker.predict(dt=0.0)
    cx_after, _ = box_center(box_after)
    assert cx_after >= cx_before - 1e-6

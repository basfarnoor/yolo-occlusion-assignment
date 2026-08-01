"""Tests for src/kalman_box_tracker.py. Paper: Bewley et al., ICIP 2016 (SORT)."""
from kalman_box_tracker import KalmanBoxTracker
from geometry import box_center


def test_box_moving_steadily_keeps_moving_when_detection_is_missing():
    """A box moving 10px/step to the right should keep drifting right through
    a run of predict()-only calls (no update()), because the filter has
    learned a nonzero velocity from the observed motion."""
    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")

    # Feed a few real observations moving steadily right, to let the filter
    # learn the velocity (a single update() can't estimate motion by itself).
    for step in range(1, 4):
        tracker.predict()
        tracker.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))

    cx_before, _ = box_center(tracker.current_box())

    # Now simulate a missing detection: predict only, no update.
    predicted_box = tracker.predict()
    cx_after, _ = box_center(predicted_box)

    assert cx_after > cx_before, "prediction should continue moving in the learned direction"


def test_new_observation_corrects_an_imperfect_prediction():
    """If the filter's prediction drifts from where an object actually is,
    a real observation should pull the estimate back toward it."""
    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")

    # No motion learned yet -- predict alone assumes zero velocity.
    predicted_box = tracker.predict()
    predicted_cx, _ = box_center(predicted_box)

    # The real object actually jumped forward.
    true_box = (100.0, 0.0, 140.0, 40.0)
    true_cx, _ = box_center(true_box)

    tracker.update(true_box)
    corrected_cx, _ = box_center(tracker.current_box())

    dist_before = abs(predicted_cx - true_cx)
    dist_after = abs(corrected_cx - true_cx)
    assert dist_after < dist_before, "correction should move the estimate closer to the true observation"

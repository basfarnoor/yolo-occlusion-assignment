"""Required Task 10 tests: existence confidence never rises without new
evidence, and it correctly uses real elapsed SECONDS, not a frame count."""
from oatm.memory.confidence import ExistenceConfidenceTracker


def test_confidence_is_monotonically_non_increasing_without_new_evidence():
    tracker = ExistenceConfidenceTracker(beta=0.15, alpha=0.01)
    values = [tracker.decay(dt_seconds=1.0, current_uncertainty=float(u)) for u in range(1, 10)]
    assert all(b <= a for a, b in zip(values, values[1:])), (
        "existence confidence must never increase while no new evidence arrives"
    )


def test_reset_returns_confidence_to_exactly_one():
    tracker = ExistenceConfidenceTracker()
    tracker.decay(dt_seconds=2.0, current_uncertainty=50.0)
    tracker.decay(dt_seconds=2.0, current_uncertainty=100.0)
    assert tracker.existence_confidence < 1.0
    tracker.reset()
    assert tracker.existence_confidence == 1.0


def test_a_longer_elapsed_time_costs_more_confidence_than_a_shorter_one():
    """Required: elapsed seconds, not frame count alone. Two trackers facing
    the identical uncertainty growth must lose different amounts of
    confidence if their real dt differs."""
    tracker_short = ExistenceConfidenceTracker(beta=0.2, alpha=0.0)
    tracker_long = ExistenceConfidenceTracker(beta=0.2, alpha=0.0)

    tracker_short.decay(dt_seconds=0.1, current_uncertainty=10.0)
    tracker_long.decay(dt_seconds=1.0, current_uncertainty=10.0)

    assert tracker_long.existence_confidence < tracker_short.existence_confidence, (
        "a full-second gap must cost more confidence than a tenth-of-a-second gap "
        "at the same hazard rate -- proves dt is read as seconds, not just 'one step'"
    )


def test_faster_growing_uncertainty_costs_more_confidence():
    tracker_stable = ExistenceConfidenceTracker(beta=0.1, alpha=0.05)
    tracker_growing = ExistenceConfidenceTracker(beta=0.1, alpha=0.05)

    tracker_stable.decay(dt_seconds=1.0, current_uncertainty=10.0)
    tracker_growing.decay(dt_seconds=1.0, current_uncertainty=10.0)

    tracker_stable.decay(dt_seconds=1.0, current_uncertainty=10.0)   # no growth
    tracker_growing.decay(dt_seconds=1.0, current_uncertainty=100.0)  # large growth

    assert tracker_growing.existence_confidence < tracker_stable.existence_confidence


def test_confidence_stays_within_zero_one_bounds_over_a_long_gap():
    tracker = ExistenceConfidenceTracker(beta=0.3, alpha=0.02)
    for u in range(1, 100):
        value = tracker.decay(dt_seconds=1.0, current_uncertainty=float(u) * 5)
        assert 0.0 <= value <= 1.0

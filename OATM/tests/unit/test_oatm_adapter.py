"""Smoke tests for the OATM MVP tracker (Method F): birth, real occlusion
bridging, real termination, and the exact state-transition bug caught and
fixed while wiring it up (a track previously OBSERVED_WEAK must correctly
move to OBSERVED_STRONG when a strong detection matches it again -- not get
silently stuck because it was mistaken for a freshly-birthed track)."""
from oatm.tracking.kalman import KalmanBoxTracker
from oatm.tracking.oatm_adapter import OATMTracker


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_birth_produces_observed_strong():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    assert len(outputs) == 1
    assert outputs[0].state == "OBSERVED_STRONG"
    assert outputs[0].evidence_source == "strong_detection"


def test_weak_then_strong_correctly_upgrades_state_not_stuck():
    """Regression test for the exact bug found while building this tracker:
    a track that was OBSERVED_WEAK last frame must become OBSERVED_STRONG
    when a strong detection matches it, not incorrectly skip the transition
    because it was confused with a newly-birthed track."""
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()

    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)
    track_id = outputs[0].track_id

    # Weak detection matches via the second association stage.
    outputs = tracker.update([_det(2, 2, 42, 42, conf=0.2)], timestamp=1.0)
    assert outputs[0].track_id == track_id
    assert outputs[0].state == "OBSERVED_WEAK"

    # Strong detection returns -- must upgrade, not stay stuck at OBSERVED_WEAK.
    outputs = tracker.update([_det(4, 4, 44, 44, conf=0.9)], timestamp=2.0)
    assert outputs[0].track_id == track_id
    assert outputs[0].state == "OBSERVED_STRONG", (
        "a track must upgrade from OBSERVED_WEAK to OBSERVED_STRONG on a strong match, "
        "not get stuck because it was mistaken for a freshly-birthed track"
    )


def test_occlusion_is_bridged_when_a_plausible_occluder_is_present():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)

    # Target vanishes; a DIFFERENT, unclaimed detection overlaps its predicted
    # location -- a plausible occluder, camera-derived, no privileged data.
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9, cls="truck")], timestamp=1.0)
    target_output = next((o for o in outputs if o.evidence_source == "motion_prediction"), None)
    assert target_output is not None
    assert target_output.state == "PREDICTED_HIDDEN"


def test_freshly_born_track_survives_immediate_occlusion():
    """Regression test for a real bug found while wiring this tracker up: a
    freshly-birthed Kalman filter starts with deliberately huge velocity
    uncertainty (it hasn't seen any real motion yet) that only collapses
    after a SECOND real detection. Without a guard, a track occluded on the
    very next frame after birth -- before ever getting that second
    detection -- was killed instantly by the uncertainty ceiling, no matter
    how existence_floor was tuned. A plausible occluder is present, so this
    must bridge to PREDICTED_HIDDEN, not die on the spot."""
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)
    track_id = outputs[0].track_id

    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9, cls="truck")], timestamp=1.0)
    target = next((o for o in outputs if o.track_id == track_id), None)
    assert target is not None, "a freshly-born track must not be instantly killed by immature uncertainty"
    assert target.state == "PREDICTED_HIDDEN"
    assert target.termination_reason is None


def test_ordinary_miss_with_no_occluder_eventually_terminates():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)
    track_id = outputs[0].track_id

    # Missing for many frames with nothing else in the scene at all.
    alive = True
    for step in range(1, 15):
        outputs = tracker.update([], timestamp=float(step))
        alive = any(o.track_id == track_id for o in outputs)
        if not alive:
            break
    assert not alive, "a track with no supporting evidence at all must eventually be terminated"


def test_every_output_has_exactly_one_evidence_source_or_none():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    tracker.update([_det(0, 0, 40, 40, conf=0.9), _det(200, 0, 240, 40, conf=0.9)], timestamp=0.0)
    outputs = tracker.update([_det(1, 1, 41, 41, conf=0.2)], timestamp=1.0)
    for o in outputs:
        assert o.evidence_source in ("strong_detection", "weak_detection", "motion_prediction")
        assert o.state in ("OBSERVED_STRONG", "OBSERVED_WEAK", "PREDICTED_HIDDEN")


def test_terminated_tracks_are_never_output_again():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMTracker()
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)
    track_id = outputs[0].track_id

    seen_after_termination = False
    terminated = False
    for step in range(1, 15):
        outputs = tracker.update([], timestamp=float(step))
        ids_now = {o.track_id for o in outputs}
        if terminated and track_id in ids_now:
            seen_after_termination = True
        if track_id not in ids_now:
            terminated = True
    assert not seen_after_termination

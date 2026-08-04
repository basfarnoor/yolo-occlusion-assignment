"""Required Task 8 tests: stationary vs. timestamp-aware constant-velocity
prediction across all seven synthetic motion regimes, with growing
uncertainty confirmed. Ground truth here is EXACT (synthetic), so "who is
closer" is a checkable fact."""
import pytest

from oatm.memory.motion_comparison import run_comparison
from oatm.memory.motion_regimes import (
    ALL_REGIMES,
    abrupt_motion_regime,
    missing_then_reappear_regime,
    slow_motion_regime,
    smooth_motion_regime,
    stationary_regime,
    turning_motion_regime,
    unequal_timestamp_gaps_regime,
)


@pytest.mark.parametrize("regime_fn", ALL_REGIMES)
def test_kalman_uncertainty_grows_monotonically_during_every_gap(regime_fn):
    result = run_comparison(regime_fn())
    assert result["kalman_uncertainty_grows_monotonically"], (
        f"{result['regime']}: uncertainty must not shrink while no new evidence arrives"
    )
    assert result["kalman_uncertainty_last"] > result["kalman_uncertainty_first"], (
        f"{result['regime']}: uncertainty should have grown by the end of a multi-step gap"
    )


def test_stationary_object_both_models_perform_perfectly():
    result = run_comparison(stationary_regime())
    assert result["mean_stationary_center_error"] < 1e-6
    assert result["mean_kalman_center_error"] < 1e-6


def test_smooth_motion_kalman_beats_stationary():
    result = run_comparison(smooth_motion_regime())
    assert result["kalman_beats_stationary_on_error"]
    assert result["mean_kalman_center_error"] < result["mean_stationary_center_error"] / 2


def test_slow_motion_the_two_models_are_close():
    """When motion is very slow, static memory should be competitive --
    not necessarily a "loss" for stationary."""
    result = run_comparison(slow_motion_regime())
    assert result["mean_stationary_center_error"] < 50.0
    assert result["mean_kalman_center_error"] < 50.0


def test_unequal_timestamp_gaps_kalman_tracks_real_world_velocity():
    """This is the exact repair Assignment 4 made over Assignment 3's fixed
    one-step-per-call Kalman transition -- confirms it actually pays off."""
    result = run_comparison(unequal_timestamp_gaps_regime())
    assert result["kalman_beats_stationary_on_error"]


def test_turning_motion_is_reported_honestly_even_if_kalman_struggles():
    """Constant-velocity models are NOT expected to handle turning well --
    this test only requires the comparison to run and report a real number,
    not that Kalman wins. A negative or mixed result here is expected and
    must not be hidden."""
    result = run_comparison(turning_motion_regime())
    assert result["mean_kalman_center_error"] >= 0.0
    assert result["mean_stationary_center_error"] >= 0.0
    # No assertion that kalman "wins" -- turning violates its assumption by design.


def test_abrupt_motion_change_degrades_kalman_prediction_during_the_gap():
    """After a sudden velocity change, Kalman's prediction (based on the OLD
    velocity) should drift increasingly far from the NEW true position -- a
    known, honestly-reported limitation, not a hidden failure."""
    result = run_comparison(abrupt_motion_regime())
    gap_errors = [s["kalman_center_error"] for s in result["per_step"] if s["in_gap"]]
    assert gap_errors[-1] > gap_errors[0], "error should grow as the stale velocity compounds"


def test_missing_then_reappear_kalman_still_beats_stationary_over_a_long_gap():
    result = run_comparison(missing_then_reappear_regime())
    assert result["kalman_beats_stationary_on_error"]
    assert result["n_gap_steps"] == 9


def test_all_regimes_produce_a_reportable_result_without_crashing():
    for regime_fn in ALL_REGIMES:
        result = run_comparison(regime_fn())
        assert result["n_gap_steps"] > 0
        assert isinstance(result["mean_kalman_center_error"], float)

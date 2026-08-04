"""Required Task 10 tests: exactly one termination reason (even with
multiple simultaneous triggers), and clear exits terminate earlier than
plausible occlusions."""
from oatm.occlusion.termination import (
    REASON_EXISTENCE_FLOOR,
    REASON_FAILED_EXPECTED_REAPPEARANCE,
    REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP,
    REASON_PREDICTED_EXIT,
    REASON_UNCERTAINTY_CEILING,
    TerminationInputs,
    evaluate_termination,
)


def test_no_termination_when_nothing_is_triggered():
    decision = evaluate_termination(TerminationInputs())
    assert not decision.should_terminate
    assert decision.reason is None


def test_exactly_one_reason_even_when_every_condition_is_simultaneously_true():
    inputs = TerminationInputs(
        predicted_exit=True, impossible_occluder_relationship=True,
        localization_uncertainty=1000.0, uncertainty_ceiling=500.0,
        existence_confidence=0.01, existence_floor=0.05,
        failed_expected_reappearance=True,
    )
    decision = evaluate_termination(inputs)
    assert decision.should_terminate
    assert decision.reason == REASON_PREDICTED_EXIT, "predicted_exit has top priority"
    assert decision.reason in (
        REASON_PREDICTED_EXIT, REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP,
        REASON_UNCERTAINTY_CEILING, REASON_EXISTENCE_FLOOR, REASON_FAILED_EXPECTED_REAPPEARANCE,
    ), "the reason must be exactly one of the five known reasons, never a combination"


def test_each_reason_fires_in_isolation():
    assert evaluate_termination(TerminationInputs(predicted_exit=True)).reason == REASON_PREDICTED_EXIT
    assert evaluate_termination(
        TerminationInputs(impossible_occluder_relationship=True)
    ).reason == REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP
    assert evaluate_termination(
        TerminationInputs(localization_uncertainty=999.0, uncertainty_ceiling=500.0)
    ).reason == REASON_UNCERTAINTY_CEILING
    assert evaluate_termination(
        TerminationInputs(existence_confidence=0.01, existence_floor=0.05)
    ).reason == REASON_EXISTENCE_FLOOR
    assert evaluate_termination(
        TerminationInputs(failed_expected_reappearance=True)
    ).reason == REASON_FAILED_EXPECTED_REAPPEARANCE


def test_clear_exit_terminates_earlier_than_plausible_occlusion():
    """A track predicted to have exited the frame must be cut loose
    immediately; a track with only mild uncertainty growth (plausible
    occlusion) must be allowed to persist longer."""
    exit_decision = evaluate_termination(TerminationInputs(predicted_exit=True))
    assert exit_decision.should_terminate

    plausible_occlusion = evaluate_termination(TerminationInputs(
        predicted_exit=False, localization_uncertainty=50.0, uncertainty_ceiling=500.0,
        existence_confidence=0.8, existence_floor=0.05,
    ))
    assert not plausible_occlusion.should_terminate, (
        "a plausible, still-early occlusion must not be terminated at the same time as a clear exit"
    )


def test_termination_config_thresholds_are_frozen_and_not_silently_changed():
    """Loads the committed termination.yaml and confirms the values match
    what was frozen -- a canary against accidental re-tuning after
    evaluation results are opened (thresholds must only change via an
    explicit, reviewed edit to this file, never programmatically)."""
    import yaml

    from oatm.config import find_repo_root

    repo_root = find_repo_root()
    with open(repo_root / "OATM" / "configs" / "termination.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert cfg["existence_confidence"]["beta"] == 0.15
    assert cfg["existence_confidence"]["alpha"] == 0.01
    assert cfg["existence_confidence"]["existence_floor"] == 0.05
    assert cfg["localization"]["uncertainty_ceiling"] == 500.0
    assert cfg["fixed_lifetime_baseline"]["max_missing_frames"] == 5

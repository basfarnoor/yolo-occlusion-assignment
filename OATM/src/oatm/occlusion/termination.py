"""Phase 9 (Task 10): anti-ghost termination. Decides whether a
`PREDICTED_HIDDEN` track must become `LOST`/`EXITED` this step, and records
EXACTLY ONE primary reason -- never more than one, even when several
conditions are simultaneously true.

Checked in a fixed priority order (documented, not incidental): a predicted
exit is checked first because it is the most specific, most confidently
causal signal (the track's own trajectory says it left); an impossible
occluder relationship next (a direct geometric contradiction); then the two
continuous, threshold-based signals (uncertainty ceiling, then existence
floor); "failed expected reappearance" last, since it is the softest signal
(an expectation not met, not a hard contradiction).
"""
from __future__ import annotations

from dataclasses import dataclass

REASON_PREDICTED_EXIT = "predicted_exit"
REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP = "impossible_occluder_relationship"
REASON_UNCERTAINTY_CEILING = "uncertainty_ceiling_exceeded"
REASON_EXISTENCE_FLOOR = "existence_confidence_below_floor"
REASON_FAILED_EXPECTED_REAPPEARANCE = "failed_expected_reappearance"

TERMINATION_PRIORITY = (
    REASON_PREDICTED_EXIT,
    REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP,
    REASON_UNCERTAINTY_CEILING,
    REASON_EXISTENCE_FLOOR,
    REASON_FAILED_EXPECTED_REAPPEARANCE,
)


@dataclass
class TerminationInputs:
    predicted_exit: bool = False
    impossible_occluder_relationship: bool = False
    localization_uncertainty: float = 0.0
    uncertainty_ceiling: float = 500.0
    existence_confidence: float = 1.0
    existence_floor: float = 0.05
    failed_expected_reappearance: bool = False


@dataclass
class TerminationDecision:
    should_terminate: bool
    reason: str | None  # exactly one of the REASON_* constants, or None


def evaluate_termination(inputs: TerminationInputs) -> TerminationDecision:
    """Returns a decision with AT MOST one reason -- the highest-priority
    condition that is true, even if several are true simultaneously."""
    if inputs.predicted_exit:
        return TerminationDecision(True, REASON_PREDICTED_EXIT)
    if inputs.impossible_occluder_relationship:
        return TerminationDecision(True, REASON_IMPOSSIBLE_OCCLUDER_RELATIONSHIP)
    if inputs.localization_uncertainty > inputs.uncertainty_ceiling:
        return TerminationDecision(True, REASON_UNCERTAINTY_CEILING)
    if inputs.existence_confidence < inputs.existence_floor:
        return TerminationDecision(True, REASON_EXISTENCE_FLOOR)
    if inputs.failed_expected_reappearance:
        return TerminationDecision(True, REASON_FAILED_EXPECTED_REAPPEARANCE)
    return TerminationDecision(False, None)

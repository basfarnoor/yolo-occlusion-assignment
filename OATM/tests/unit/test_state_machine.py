"""Required Task 9 test: an EXHAUSTIVE transition truth table -- every
allowed transition works, and every other (state, event) pair -- including
any attempt to move a terminal state -- is rejected, not silently ignored."""
import itertools

import pytest

from oatm.occlusion.state_machine import (
    ALLOWED_TRANSITIONS,
    EVENT_INSUFFICIENT_EVIDENCE,
    EVENT_OCCLUSION_EVIDENCE,
    EVENT_PREDICTED_EXIT,
    EVENT_STRONG_DETECTION,
    EVENT_WEAK_DETECTION,
    EXITED,
    LOST,
    OBSERVED_STRONG,
    OBSERVED_WEAK,
    PREDICTED_HIDDEN,
    InvalidTransitionError,
    TrackStateMachine,
)

ALL_STATES = [OBSERVED_STRONG, OBSERVED_WEAK, PREDICTED_HIDDEN, LOST, EXITED]
ALL_EVENTS = [EVENT_STRONG_DETECTION, EVENT_WEAK_DETECTION, EVENT_OCCLUSION_EVIDENCE,
              EVENT_INSUFFICIENT_EVIDENCE, EVENT_PREDICTED_EXIT]


def _machine_in_state(state: str) -> TrackStateMachine:
    m = TrackStateMachine()
    m.birth()
    if state == OBSERVED_STRONG:
        return m
    # Route to the target state via one valid transition from OBSERVED_STRONG.
    route = {
        OBSERVED_WEAK: EVENT_WEAK_DETECTION,
        PREDICTED_HIDDEN: EVENT_OCCLUSION_EVIDENCE,
        LOST: EVENT_INSUFFICIENT_EVIDENCE,
        EXITED: EVENT_PREDICTED_EXIT,
    }
    m.transition(route[state])
    return m


@pytest.mark.parametrize("state,event", list(itertools.product(ALL_STATES, ALL_EVENTS)))
def test_exhaustive_transition_table(state, event):
    machine = _machine_in_state(state)
    key = (state, event)

    if state in (LOST, EXITED):
        # Terminal: EVERY event must be rejected, no exceptions.
        with pytest.raises(InvalidTransitionError):
            machine.transition(event)
        return

    if key in ALLOWED_TRANSITIONS:
        result = machine.transition(event)
        assert result == ALLOWED_TRANSITIONS[key]
    else:
        with pytest.raises(InvalidTransitionError):
            machine.transition(event)


def test_the_transition_table_itself_defines_no_transitions_out_of_terminal_states():
    """Guards the truth table's own invariant: LOST/EXITED must never appear
    as a source state in ALLOWED_TRANSITIONS -- terminality is structural,
    not just enforced by an if-check that could be forgotten later."""
    source_states = {state for state, _event in ALLOWED_TRANSITIONS}
    assert LOST not in source_states
    assert EXITED not in source_states


def test_birth_only_ever_produces_observed_strong():
    m = TrackStateMachine()
    assert m.birth() == OBSERVED_STRONG


def test_birth_cannot_be_called_twice():
    m = TrackStateMachine()
    m.birth()
    with pytest.raises(InvalidTransitionError):
        m.birth()


def test_cannot_transition_before_birth():
    m = TrackStateMachine()
    with pytest.raises(InvalidTransitionError):
        m.transition(EVENT_STRONG_DETECTION)


def test_lost_and_exited_are_terminal_for_every_possible_event():
    for terminal_state in (LOST, EXITED):
        for event in ALL_EVENTS:
            m = _machine_in_state(terminal_state)
            with pytest.raises(InvalidTransitionError):
                m.transition(event)


def test_evidence_source_mapping_is_unambiguous():
    m = TrackStateMachine()
    m.birth()
    assert m.evidence_source == "strong_detection"
    m.transition(EVENT_WEAK_DETECTION)
    assert m.evidence_source == "weak_detection"
    m.transition(EVENT_OCCLUSION_EVIDENCE)
    assert m.evidence_source == "motion_prediction"
    m.transition(EVENT_INSUFFICIENT_EVIDENCE)
    assert m.evidence_source is None, "LOST must never claim any evidence source"

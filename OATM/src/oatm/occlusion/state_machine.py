"""Phase 7 (Task 9): the exact OATM track-state machine, from
IMPLEMENTATION_PLAN.md's state diagram.

Five states only: OBSERVED_STRONG, OBSERVED_WEAK, PREDICTED_HIDDEN, LOST,
EXITED. A track is only ever BORN into OBSERVED_STRONG (never from a weak
detection -- matches the ByteTrack baseline's own birth rule). LOST and
EXITED are terminal: once a track reaches either, this state machine object
must not be updated again. A later compatible detection starts a genuinely
NEW track (new track_id), never resurrects the old one -- "LOST cannot
silently reconnect to an old ID."

Every transition is driven by one of five mutually exclusive EVENTS, decided
causally (current frame and earlier only) by `oatm.occlusion.evidence`:

  strong_detection      -- a confident detection matched this track now.
  weak_detection         -- a low-confidence detection matched this track now.
  occlusion_evidence      -- no detection matched, but real, camera-derived
                             evidence supports "probably still hidden."
  insufficient_evidence   -- no detection matched, and no such evidence exists.
  predicted_exit          -- no detection matched, and the track's own
                             trajectory/uncertainty predicts it left the frame.
"""
from __future__ import annotations

OBSERVED_STRONG = "OBSERVED_STRONG"
OBSERVED_WEAK = "OBSERVED_WEAK"
PREDICTED_HIDDEN = "PREDICTED_HIDDEN"
LOST = "LOST"
EXITED = "EXITED"

TERMINAL_STATES = frozenset({LOST, EXITED})

EVENT_STRONG_DETECTION = "strong_detection"
EVENT_WEAK_DETECTION = "weak_detection"
EVENT_OCCLUSION_EVIDENCE = "occlusion_evidence"
EVENT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
EVENT_PREDICTED_EXIT = "predicted_exit"

# The exhaustive, exact transition table. Any (state, event) pair not listed
# here is an invalid transition and must raise, not silently do nothing.
ALLOWED_TRANSITIONS: dict[tuple[str, str], str] = {
    (OBSERVED_STRONG, EVENT_STRONG_DETECTION): OBSERVED_STRONG,
    (OBSERVED_STRONG, EVENT_WEAK_DETECTION): OBSERVED_WEAK,
    (OBSERVED_STRONG, EVENT_OCCLUSION_EVIDENCE): PREDICTED_HIDDEN,
    (OBSERVED_STRONG, EVENT_INSUFFICIENT_EVIDENCE): LOST,
    (OBSERVED_STRONG, EVENT_PREDICTED_EXIT): EXITED,

    (OBSERVED_WEAK, EVENT_STRONG_DETECTION): OBSERVED_STRONG,
    (OBSERVED_WEAK, EVENT_WEAK_DETECTION): OBSERVED_WEAK,
    (OBSERVED_WEAK, EVENT_OCCLUSION_EVIDENCE): PREDICTED_HIDDEN,
    (OBSERVED_WEAK, EVENT_INSUFFICIENT_EVIDENCE): LOST,
    (OBSERVED_WEAK, EVENT_PREDICTED_EXIT): EXITED,

    (PREDICTED_HIDDEN, EVENT_STRONG_DETECTION): OBSERVED_STRONG,
    (PREDICTED_HIDDEN, EVENT_WEAK_DETECTION): OBSERVED_WEAK,
    (PREDICTED_HIDDEN, EVENT_OCCLUSION_EVIDENCE): PREDICTED_HIDDEN,
    (PREDICTED_HIDDEN, EVENT_INSUFFICIENT_EVIDENCE): LOST,
    (PREDICTED_HIDDEN, EVENT_PREDICTED_EXIT): EXITED,
    # LOST and EXITED: no entries -- terminal, enforced in transition() below.
}

# Evidence sources a saved output row may report -- current detections and
# memory-only predictions must never share an ambiguous label.
STATE_TO_EVIDENCE_SOURCE = {
    OBSERVED_STRONG: "strong_detection",
    OBSERVED_WEAK: "weak_detection",
    PREDICTED_HIDDEN: "motion_prediction",
    LOST: None,     # no output row at all once LOST
    EXITED: None,   # no output row at all once EXITED
}


class InvalidTransitionError(Exception):
    """Raised for any (state, event) pair the exhaustive table does not
    define -- including any attempt to transition a terminal (LOST/EXITED)
    track, or to birth a track into anything but OBSERVED_STRONG."""


class TrackStateMachine:
    def __init__(self):
        self.state: str | None = None
        self.history: list[str] = []

    def birth(self) -> str:
        """A track may only ever be born from a confirmed high-score
        detection -- there is no `birth_weak()`."""
        if self.state is not None:
            raise InvalidTransitionError("birth() may only be called once, on a fresh track")
        self.state = OBSERVED_STRONG
        self.history.append(self.state)
        return self.state

    def transition(self, event: str) -> str:
        if self.state is None:
            raise InvalidTransitionError("cannot transition before birth()")
        if self.state in TERMINAL_STATES:
            raise InvalidTransitionError(
                f"{self.state} is terminal -- a new track must be created instead of "
                f"transitioning this one further (attempted event={event})"
            )
        key = (self.state, event)
        if key not in ALLOWED_TRANSITIONS:
            raise InvalidTransitionError(f"no transition defined for state={self.state}, event={event}")
        self.state = ALLOWED_TRANSITIONS[key]
        self.history.append(self.state)
        return self.state

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def evidence_source(self) -> str | None:
        return STATE_TO_EVIDENCE_SOURCE[self.state]

"""Phase 9 (Task 10): adaptive existence-confidence decay.

Implements METHODOLOGY.md's hazard model:

    h_t = beta + alpha * delta_U_t
    P_exist_t = P_exist_{t-1} * exp(-h_t * delta_t)

`delta_U_t` is the INCREMENTAL growth in localization uncertainty since the
last step (never negative here -- uncertainty only grows while a track has
no new evidence; a real update resets both uncertainty and existence
confidence). `delta_t` is real elapsed seconds, not a frame count, so a
missing 0.5s costs half as much confidence as a missing 1.0s at the same
hazard rate.

Existence confidence, identity confidence, and localization uncertainty are
kept as three separate numbers everywhere in this module -- never collapsed
into one score. This is an MVP hazard policy, not a claim that the equation
itself is novel (METHODOLOGY.md is explicit about this).
"""
from __future__ import annotations

import math


class ExistenceConfidenceTracker:
    def __init__(self, beta: float = 0.15, alpha: float = 0.01):
        """`beta`: base hazard rate (confidence lost per second even with no
        growing uncertainty). `alpha`: extra hazard per unit of uncertainty
        growth. Both must be tuned only on development/validation data and
        then frozen -- see configs/termination.yaml."""
        self.beta = beta
        self.alpha = alpha
        self.existence_confidence = 1.0
        self._last_uncertainty: float | None = None

    def reset(self) -> None:
        """Call whenever new real evidence (a detection) arrives. Confidence
        cannot increase without new evidence -- this is the ONLY way it goes
        back up, and it goes all the way back to 1.0, not partway."""
        self.existence_confidence = 1.0
        self._last_uncertainty = None

    def decay(self, dt_seconds: float, current_uncertainty: float) -> float:
        """Advances existence confidence by one step of missing evidence.
        Must be called with real elapsed seconds, not an assumed frame
        count."""
        dt_seconds = max(dt_seconds, 1e-6)
        if self._last_uncertainty is None:
            delta_u = 0.0
        else:
            delta_u = max(0.0, current_uncertainty - self._last_uncertainty)
        self._last_uncertainty = current_uncertainty

        hazard = self.beta + self.alpha * delta_u
        self.existence_confidence *= math.exp(-hazard * dt_seconds)
        return self.existence_confidence

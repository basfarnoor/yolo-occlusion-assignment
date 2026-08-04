"""Task 10: compares a fixed-lifetime termination policy against the
uncertainty-aware adaptive one, at MATCHED ghost risk -- not just at each
policy's own best-recall setting. Uses the real, already-tested Kalman
filter (Task 8) for realistic uncertainty growth, not a synthetic curve.

Two synthetic scenario families, both with exactly known truth:

  - "occlusion": the object reappears after a gap of known length -- a track
    that survives to reunion is a correct bridge (recall).
  - "exit": the object NEVER reappears -- any frame the track survives past
    the moment it truly left is a ghost (cost), measured as a DURATION, not
    just a yes/no rate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oatm.memory.confidence import ExistenceConfidenceTracker
from oatm.occlusion.termination import TerminationInputs, evaluate_termination
from oatm.tracking.kalman import KalmanBoxTracker


@dataclass
class FixedLifetimePolicy:
    max_missing_frames: int

    def run_gap(self, gap_length: int) -> bool:
        """Returns True if the track survives the whole gap (bridged)."""
        return gap_length <= self.max_missing_frames

    def ghost_duration(self, horizon: int) -> int:
        """How many frames an track with no real evidence keeps alive
        before this policy terminates it, for an object that never
        returns, capped at `horizon`."""
        return min(self.max_missing_frames, horizon)


@dataclass
class AdaptivePolicy:
    existence_floor: float
    beta: float
    alpha: float
    uncertainty_ceiling: float

    def _simulate(self, n_missing_frames: int) -> int:
        """Returns the number of frames the track survives (>=0) before the
        adaptive policy terminates it, given `n_missing_frames` of
        no-evidence steps (capped at n_missing_frames)."""
        KalmanBoxTracker.reset_id_counter()
        tracker = KalmanBoxTracker((100.0, 100.0, 140.0, 140.0), "object")
        conf_tracker = ExistenceConfidenceTracker(beta=self.beta, alpha=self.alpha)

        # Warm up with real observations first -- a freshly-constructed
        # Kalman filter starts with a deliberately huge velocity covariance
        # (it hasn't seen any real motion yet), which would otherwise trip
        # the uncertainty ceiling on the very first missing frame regardless
        # of policy. A real track entering a gap has already been observed
        # for a while, so this warm-up matches that realistic starting point.
        for step in range(1, 6):
            tracker.predict(dt=1.0)
            tracker.update((100.0 + 10.0 * step, 100.0, 140.0 + 10.0 * step, 140.0))
        conf_tracker.reset()

        survived = 0
        for _ in range(n_missing_frames):
            tracker.predict(dt=1.0)
            uncertainty = float(_trace(tracker))
            existence_confidence = conf_tracker.decay(dt_seconds=1.0, current_uncertainty=uncertainty)
            decision = evaluate_termination(TerminationInputs(
                localization_uncertainty=uncertainty, uncertainty_ceiling=self.uncertainty_ceiling,
                existence_confidence=existence_confidence, existence_floor=self.existence_floor,
            ))
            if decision.should_terminate:
                break
            survived += 1
        return survived

    def run_gap(self, gap_length: int) -> bool:
        return self._simulate(gap_length) >= gap_length

    def ghost_duration(self, horizon: int) -> int:
        return self._simulate(horizon)


def _trace(tracker: KalmanBoxTracker) -> float:
    return float(np.trace(tracker.P))


def evaluate_policy(policy, gap_lengths: list[int], ghost_horizon: int) -> dict:
    recall = sum(1 for g in gap_lengths if policy.run_gap(g)) / len(gap_lengths)
    ghost = policy.ghost_duration(ghost_horizon)
    return {"recall": recall, "ghost_duration_frames": ghost}

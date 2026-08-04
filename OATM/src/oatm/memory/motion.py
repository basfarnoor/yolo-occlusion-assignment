"""Phase 6 (Task 8): the two starting motion models, compared before any
occlusion classifier, appearance memory, or ego-motion compensation is
added.

1. `StationaryPredictor` -- freezes the last observed box exactly (Assignment
   2's idea). Has no notion of elapsed time at all.
2. `oatm.tracking.kalman.KalmanBoxTracker` -- the existing timestamp-aware
   constant-velocity filter (reused, not reimplemented here).

Both expose enough to compare center error against a KNOWN synthetic
trajectory and to read a localization-uncertainty value.
"""
from __future__ import annotations


class StationaryPredictor:
    """No motion model at all -- whatever box it last saw, it keeps
    reporting, unchanged, for as long as it's asked to predict."""

    def __init__(self, box: tuple[float, float, float, float]):
        self.box = box
        self.time_since_update = 0

    def predict(self, dt: float = 1.0) -> tuple[float, float, float, float]:
        self.time_since_update += 1
        return self.box  # never moves on its own

    def update(self, box: tuple[float, float, float, float]) -> None:
        self.box = box
        self.time_since_update = 0

    def localization_uncertainty(self) -> float:
        """No real state-estimation uncertainty exists for this model --
        a crude, honestly-labeled proxy: uncertainty grows linearly with
        frames since the last real observation."""
        return float(self.time_since_update)

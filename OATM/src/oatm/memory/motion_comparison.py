"""Task 8: runs both motion models over one synthetic regime fixture and
compares their prediction accuracy during the fixture's "gap" (the steps with
no real observation), against the fixture's exactly-known true boxes."""
from __future__ import annotations

import numpy as np

from oatm.memory.motion import StationaryPredictor
from oatm.memory.motion_regimes import MotionRegimeFixture
from oatm.tracking.geometry import center_error, iou
from oatm.tracking.kalman import KalmanBoxTracker


def _dts(timestamps: list[float]) -> list[float]:
    return [1.0] + [b - a for a, b in zip(timestamps, timestamps[1:])]


def run_comparison(fixture: MotionRegimeFixture) -> dict:
    KalmanBoxTracker.reset_id_counter()
    stationary = StationaryPredictor(fixture.true_boxes[0])
    kalman = KalmanBoxTracker(fixture.true_boxes[0], "object")

    dts = _dts(fixture.timestamps)
    gap_set = set(fixture.gap_indices)

    per_step = []
    for i in range(1, len(fixture.timestamps)):
        dt = dts[i]
        true_box = fixture.true_boxes[i]

        stationary_box = stationary.predict(dt)
        kalman_box = kalman.predict(dt)

        in_gap = i in gap_set
        if not in_gap:
            stationary.update(true_box)
            kalman.update(true_box)

        per_step.append({
            "step": i, "in_gap": in_gap,
            "stationary_center_error": center_error(stationary_box, true_box),
            "kalman_center_error": center_error(kalman_box, true_box),
            "stationary_iou": iou(stationary_box, true_box),
            "kalman_iou": iou(kalman_box, true_box),
            "stationary_uncertainty": stationary.localization_uncertainty(),
            "kalman_uncertainty": float(np.trace(kalman.P)),
        })

    gap_steps = [s for s in per_step if s["in_gap"]]
    mean_stationary_error = sum(s["stationary_center_error"] for s in gap_steps) / len(gap_steps)
    mean_kalman_error = sum(s["kalman_center_error"] for s in gap_steps) / len(gap_steps)
    mean_stationary_iou = sum(s["stationary_iou"] for s in gap_steps) / len(gap_steps)
    mean_kalman_iou = sum(s["kalman_iou"] for s in gap_steps) / len(gap_steps)

    kalman_uncertainties = [s["kalman_uncertainty"] for s in gap_steps]
    uncertainty_monotonic = all(b >= a - 1e-9 for a, b in zip(kalman_uncertainties, kalman_uncertainties[1:]))

    return {
        "regime": fixture.name, "description": fixture.description,
        "n_gap_steps": len(gap_steps),
        "mean_stationary_center_error": mean_stationary_error,
        "mean_kalman_center_error": mean_kalman_error,
        "mean_stationary_iou": mean_stationary_iou,
        "mean_kalman_iou": mean_kalman_iou,
        "kalman_beats_stationary_on_error": mean_kalman_error < mean_stationary_error,
        "kalman_uncertainty_grows_monotonically": uncertainty_monotonic,
        "kalman_uncertainty_first": kalman_uncertainties[0] if kalman_uncertainties else None,
        "kalman_uncertainty_last": kalman_uncertainties[-1] if kalman_uncertainties else None,
        "per_step": per_step,
    }

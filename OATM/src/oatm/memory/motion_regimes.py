"""Task 8: synthetic motion-regime fixtures with KNOWN ground truth, used to
compare `StationaryPredictor` against the timestamp-aware constant-velocity
Kalman filter honestly -- since the true trajectory is defined by this
module, not estimated, "who was closer" is an exact, checkable fact rather
than an opinion.

Each regime returns a list of (timestamp_seconds, true_box) for every step,
plus the indices that make up the "gap" -- the steps where the model only
gets to `predict()`, never `update()`, simulating a missing detection.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

BOX_SIZE = 40.0  # width and height stay fixed; only the center moves


def _box_at(cx: float, cy: float) -> tuple[float, float, float, float]:
    half = BOX_SIZE / 2
    return (cx - half, cy - half, cx + half, cy + half)


@dataclass
class MotionRegimeFixture:
    name: str
    description: str
    timestamps: list[float]
    true_boxes: list[tuple[float, float, float, float]]
    gap_indices: list[int]  # steps with no real observation -- predict() only


def stationary_regime() -> MotionRegimeFixture:
    n = 20
    timestamps = [float(i) for i in range(n)]
    boxes = [_box_at(200.0, 200.0) for _ in range(n)]
    return MotionRegimeFixture("stationary", "Object does not move at all.",
                                timestamps, boxes, gap_indices=list(range(8, 13)))


def smooth_motion_regime() -> MotionRegimeFixture:
    n = 20
    vx = 20.0  # px/s
    timestamps = [float(i) for i in range(n)]
    boxes = [_box_at(100.0 + vx * t, 200.0) for t in timestamps]
    return MotionRegimeFixture("smooth_motion", "Constant velocity, 20 px/s.",
                                timestamps, boxes, gap_indices=list(range(8, 13)))


def slow_motion_regime() -> MotionRegimeFixture:
    n = 20
    vx = 3.0  # px/s -- barely moving
    timestamps = [float(i) for i in range(n)]
    boxes = [_box_at(100.0 + vx * t, 200.0) for t in timestamps]
    return MotionRegimeFixture("slow_motion", "Constant but very slow velocity, 3 px/s.",
                                timestamps, boxes, gap_indices=list(range(8, 13)))


def unequal_timestamp_gaps_regime() -> MotionRegimeFixture:
    """Real elapsed time between frames alternates 0.05s / 0.2s, but the
    object's true velocity in px/second never changes -- a dt-aware model
    should track this correctly; a fixed one-step-per-call model would not."""
    vx = 40.0  # px/s
    dts = [0.05, 0.2] * 10
    timestamps = [0.0]
    for dt in dts:
        timestamps.append(timestamps[-1] + dt)
    boxes = [_box_at(100.0 + vx * t, 200.0) for t in timestamps]
    return MotionRegimeFixture("unequal_timestamp_gaps",
                                "Constant real-world velocity, irregular frame timing.",
                                timestamps, boxes, gap_indices=list(range(8, 13)))


def turning_motion_regime() -> MotionRegimeFixture:
    """A circular arc -- velocity DIRECTION keeps changing, directly
    violating the constant-velocity assumption."""
    n = 20
    radius = 150.0
    angular_rate = 0.15  # radians/step
    timestamps = [float(i) for i in range(n)]
    boxes = []
    for i in timestamps:
        angle = angular_rate * i
        cx = 300.0 + radius * math.sin(angle)
        cy = 300.0 - radius * math.cos(angle)
        boxes.append(_box_at(cx, cy))
    return MotionRegimeFixture("turning_motion", "Circular arc -- velocity direction keeps changing.",
                                timestamps, boxes, gap_indices=list(range(8, 13)))


def abrupt_motion_regime() -> MotionRegimeFixture:
    """Smooth motion, then a sudden large velocity change -- tests
    robustness when the constant-velocity assumption breaks mid-track."""
    n = 20
    timestamps = [float(i) for i in range(n)]
    boxes = []
    for i, t in enumerate(timestamps):
        if i < 10:
            cx = 100.0 + 15.0 * t
        else:
            cx = 100.0 + 15.0 * 10 + 60.0 * (t - 10)  # sudden speed-up
        boxes.append(_box_at(cx, 200.0))
    return MotionRegimeFixture("abrupt_motion", "Sudden large velocity change partway through.",
                                timestamps, boxes, gap_indices=list(range(12, 17)))


def missing_then_reappear_regime() -> MotionRegimeFixture:
    """Smooth motion with a longer missing window than the other regimes,
    to specifically study how far each model drifts before reappearance."""
    n = 20
    vx = 25.0
    timestamps = [float(i) for i in range(n)]
    boxes = [_box_at(100.0 + vx * t, 200.0) for t in timestamps]
    return MotionRegimeFixture("missing_then_reappear", "Smooth motion with a long missing window.",
                                timestamps, boxes, gap_indices=list(range(6, 15)))


ALL_REGIMES = [
    stationary_regime, smooth_motion_regime, slow_motion_regime,
    unequal_timestamp_gaps_regime, turning_motion_regime, abrupt_motion_regime,
    missing_then_reappear_regime,
]

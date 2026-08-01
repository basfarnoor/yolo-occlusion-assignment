"""ByteTrack paper reference: Zhang et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box," ECCV 2022 (https://arxiv.org/abs/2110.06864).

A from-scratch Kalman filter (NumPy only, no filterpy) for one tracked box,
reused and repaired from the Assignment 3 SORT experiment (Bewley et al.,
ICIP 2016) -- see reuse_audit.md. This is the paper's "motion prediction"
component, built on the same constant-velocity SORT-style state estimation
ByteTrack itself builds on; ByteTrack's own contribution is the two-stage
association logic in bytetrack_tracker.py, not a new motion model.

State vector (7 numbers): [cx, cy, s, r, vx, vy, vs]
  cx, cy -- box center
  s      -- box "scale" (area)
  r      -- aspect ratio (assumed constant)
  vx, vy, vs -- velocity of cx, cy, s, PER SECOND (repaired from Assignment 3,
                which implicitly assumed exactly one frame per time step
                regardless of the real interval between frames -- see
                reuse_audit.md, required repair #7). predict(dt) now scales
                the velocity contribution by the real elapsed time.

Measurement vector (4 numbers): [cx, cy, s, r] -- what a real YOLO detection
gives us, converted from (x1, y1, x2, y2) via geometry.box_to_state.
"""
from __future__ import annotations

import numpy as np

from geometry import box_to_state, state_to_box

STATE_DIM = 7
MEAS_DIM = 4
DEFAULT_DT = 1.0  # used when no real timestamp is available (e.g. synthetic tests)


class KalmanBoxTracker:
    """Wraps one Kalman filter for one tracked object."""

    _next_id = 0

    def __init__(self, box: tuple[float, float, float, float], class_name: str, timestamp: float = 0.0):
        # Measurement model: we only ever *observe* [cx, cy, s, r], never velocities directly.
        self.H = np.zeros((MEAS_DIM, STATE_DIM))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Process noise: how much we distrust the constant-velocity assumption itself.
        self.Q_base = np.eye(STATE_DIM)
        self.Q_base[4:, 4:] *= 0.01  # velocities assumed to change slowly

        # Measurement noise: how much we distrust a single YOLO detection.
        self.R = np.eye(MEAS_DIM) * 1.0
        self.R[2:, 2:] *= 10.0  # scale/aspect-ratio measurements are noisier

        # Initial state: zero velocity, high uncertainty on velocity (we haven't seen any yet).
        self.x = np.zeros(STATE_DIM)
        self.x[:4] = box_to_state(box)
        self.P = np.eye(STATE_DIM) * 10.0
        self.P[4:, 4:] *= 1000.0

        self.class_name = class_name
        self.id = KalmanBoxTracker._next_id
        KalmanBoxTracker._next_id += 1
        self.last_confidence = 0.0

        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_since_update = 0
        self.last_timestamp = timestamp
        # The raw detection box actually matched THIS frame, or None if no
        # match happened yet this frame -- reset by predict(), set by
        # update(). Callers must read this (never `current_box()`) whenever
        # they need the real, unsmoothed YOLO box (reuse_audit.md repair #1).
        self.last_raw_box: tuple[float, float, float, float] | None = box

    @classmethod
    def reset_id_counter(cls) -> None:
        """Only for deterministic tests -- real experiments should never call this mid-run."""
        cls._next_id = 0

    def _transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(STATE_DIM)
        F[0, 4] = dt  # cx += vx * dt
        F[1, 5] = dt  # cy += vy * dt
        F[2, 6] = dt  # s  += vs * dt
        return F

    def predict(self, dt: float = DEFAULT_DT) -> tuple[float, float, float, float]:
        """Step 1 of the loop: predict where the box should be, using motion
        alone, advanced by the real elapsed time `dt` (in seconds) since the
        last frame. Velocities are per-second, so this makes an 0.5s gap
        predict half as much displacement as a 1.0s gap -- unlike Assignment
        3's fixed one-step-per-call version."""
        dt = max(dt, 1e-6)
        F = self._transition_matrix(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q_base * dt
        self.last_raw_box = None  # no match yet this frame; update() will set it if one occurs
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return state_to_box(self.x)

    def update(self, box: tuple[float, float, float, float]) -> None:
        """Step 2 of the loop: correct the prediction using a real new detection."""
        z = box_to_state(box)
        y = z - self.H @ self.x  # innovation: how wrong was the prediction?
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain: how much to trust this observation
        self.x = self.x + K @ y
        self.P = (np.eye(STATE_DIM) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_raw_box = box

    def current_box(self) -> tuple[float, float, float, float]:
        return state_to_box(self.x)

    def velocity(self) -> tuple[float, float]:
        """Current estimated pixel velocity (vx, vy) per second."""
        return (float(self.x[4]), float(self.x[5]))

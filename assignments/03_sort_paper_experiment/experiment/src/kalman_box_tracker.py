"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

A small, from-scratch Kalman filter (NumPy only, no filterpy) for one
tracked box. This is the paper's "state estimation" component: it predicts
where a box should be next, then corrects that guess whenever a new
detection arrives -- exactly the "eyes closed / eyes open" analogy in
paper_map.md.

State vector (7 numbers): [cx, cy, s, r, vx, vy, vs]
  cx, cy -- box center
  s      -- box "scale" (area, per the paper)
  r      -- aspect ratio (assumed constant -- the paper does not track a
             rate of change for r)
  vx, vy, vs -- velocity of cx, cy, s (this is the paper's constant-velocity
                motion model)

Measurement vector (4 numbers): [cx, cy, s, r] -- what a real YOLO detection
gives us, converted from (x1, y1, x2, y2) via geometry.box_to_state.
"""
from __future__ import annotations

import numpy as np

from geometry import box_to_state, state_to_box

STATE_DIM = 7
MEAS_DIM = 4


class KalmanBoxTracker:
    """Wraps one Kalman filter for one tracked object."""

    _next_id = 0

    def __init__(self, box: tuple[float, float, float, float], class_name: str, timestamp: float = 0.0):
        # --- Constant-velocity motion model (the paper's state estimation idea) ---
        # x_{t+1} = F @ x_t   (cx, cy, s each get + their own velocity; r stays put)
        self.F = np.eye(STATE_DIM)
        self.F[0, 4] = 1.0  # cx += vx
        self.F[1, 5] = 1.0  # cy += vy
        self.F[2, 6] = 1.0  # s  += vs

        # Measurement model: we only ever *observe* [cx, cy, s, r], never velocities directly.
        self.H = np.zeros((MEAS_DIM, STATE_DIM))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Process noise: how much we distrust the constant-velocity assumption itself.
        self.Q = np.eye(STATE_DIM) * 1.0
        self.Q[4:, 4:] *= 0.01  # velocities assumed to change slowly

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

    @classmethod
    def reset_id_counter(cls) -> None:
        """Only for deterministic tests -- real experiments should never call this mid-run."""
        cls._next_id = 0

    def predict(self) -> tuple[float, float, float, float]:
        """Step 1 of the loop: predict where the box should be, using motion alone."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
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

    def current_box(self) -> tuple[float, float, float, float]:
        return state_to_box(self.x)

    def velocity(self) -> tuple[float, float]:
        """Current estimated pixel velocity (vx, vy) per frame step."""
        return (float(self.x[4]), float(self.x[5]))

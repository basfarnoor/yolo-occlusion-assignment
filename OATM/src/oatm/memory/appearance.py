"""Task 12: a frozen clear-view appearance anchor for one track. Pure numpy
-- no torch/image dependency here, so the update/freeze rules (the actual
thing worth testing) can be verified without a real embedding model or real
images. See `oatm.memory.embedder` for the actual frozen network that
produces the embeddings this class stores.
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class AppearanceAnchor:
    """Holds at most one embedding per track: the most recent CLEAR view.
    `update()` only ever writes when the caller says the frame is eligible
    (state OBSERVED_STRONG and a quality gate on box size/position) -- an
    occluder's own appearance must never be able to overwrite the target's
    memory, so callers must never mark a PREDICTED_HIDDEN/OBSERVED_WEAK frame
    eligible."""

    def __init__(self) -> None:
        self.embedding: np.ndarray | None = None

    def update(self, embedding: np.ndarray, eligible: bool) -> None:
        if eligible:
            self.embedding = embedding

    def similarity(self, embedding: np.ndarray) -> float | None:
        if self.embedding is None:
            return None
        return cosine_similarity(self.embedding, embedding)


def is_eligible_for_anchor_update(
    state: str, box: tuple[float, float, float, float],
    image_width: float, image_height: float,
    min_box_area: float = 400.0, boundary_margin_px: float = 5.0,
) -> bool:
    """Clear-view gate: only a confident, current, reasonably-sized,
    non-truncated detection may update the anchor. Never true for
    PREDICTED_HIDDEN or OBSERVED_WEAK -- an occluder's own appearance, or a
    low-confidence guess, must never be able to overwrite the target's
    remembered appearance."""
    if state != "OBSERVED_STRONG":
        return False
    x1, y1, x2, y2 = box
    if (x2 - x1) * (y2 - y1) < min_box_area:
        return False
    if x1 < boundary_margin_px or y1 < boundary_margin_px:
        return False
    if x2 > image_width - boundary_margin_px or y2 > image_height - boundary_margin_px:
        return False
    return True

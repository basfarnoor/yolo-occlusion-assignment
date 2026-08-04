"""Task 12: reconnects a PREDICTED_HIDDEN track to an otherwise-unclaimed
high-confidence detection using its frozen appearance anchor, run as a THIRD
association stage after the existing two-stage IoU association (Tasks 6/11)
has already failed to match a track by location alone. Two modes:

  - "appearance_only": pure cosine-similarity match -- location, scale, and
    motion direction are ignored entirely. This is the ablation's "appearance
    alone" arm, deliberately naive so a null/harmful result stays visible if
    appearance similarity alone is not discriminating enough on real data.
  - "dual": requires appearance similarity AND a location-consistency gate
    (motion still has to agree, appearance is not allowed to override
    physically impossible relocations).

Hungarian-matched with a forbidden cost for any pair failing either gate --
same pattern as `oatm.tracking.association`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from oatm.memory.appearance import cosine_similarity
from oatm.tracking.geometry import iou

FORBIDDEN_COST = 1e6


@dataclass
class HiddenTrackCandidate:
    class_name: str
    predicted_box: tuple[float, float, float, float]
    appearance_embedding: np.ndarray | None


def resolve_reconnection(
    hidden_tracks: list[HiddenTrackCandidate],
    detections: list[dict],
    mode: str,
    appearance_similarity_threshold: float = 0.7,
    location_iou_threshold: float = 0.05,
) -> list[tuple[int, int]]:
    """`detections` must each carry an `"embedding"` key (np.ndarray) alongside
    the usual class/x1..y2. Returns (detection_index, hidden_track_index)
    pairs -- one-to-one, never more than one detection per track or vice
    versa."""
    n_det, n_trk = len(detections), len(hidden_tracks)
    if n_det == 0 or n_trk == 0:
        return []

    cost = np.full((n_det, n_trk), FORBIDDEN_COST)
    for d, det in enumerate(detections):
        det_embedding = det.get("embedding")
        det_box = (det["x1"], det["y1"], det["x2"], det["y2"])
        for t, track in enumerate(hidden_tracks):
            if det["class"] != track.class_name:
                continue
            if track.appearance_embedding is None or det_embedding is None:
                continue
            similarity = cosine_similarity(track.appearance_embedding, det_embedding)
            if similarity < appearance_similarity_threshold:
                continue
            if mode == "dual" and iou(det_box, track.predicted_box) < location_iou_threshold:
                continue
            cost[d, t] = 1.0 - similarity

    det_idx, trk_idx = linear_sum_assignment(cost)
    matches = []
    for d, t in zip(det_idx, trk_idx):
        if cost[d, t] < FORBIDDEN_COST:
            matches.append((int(d), int(t)))
    return matches

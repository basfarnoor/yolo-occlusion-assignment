"""Pure camera-derived admission and exit checks for Selective OATM."""
from __future__ import annotations

from dataclasses import dataclass

from oatm.tracking.geometry import box_area

Box = tuple[float, float, float, float]


def intersection_area(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def target_coverage(target: Box, candidate_occluder: Box) -> float:
    area = box_area(target)
    return 0.0 if area <= 0.0 else intersection_area(target, candidate_occluder) / area


@dataclass(frozen=True)
class GateDecision:
    admit_hidden: bool
    predicted_exit: bool
    support_score: float
    reason: str


def decide_hidden_admission(
    predicted_box: Box,
    velocity: tuple[float, float],
    unclaimed_boxes: list[Box],
    track_hits: int,
    frames_missing: int,
    *,
    image_width: float,
    image_height: float,
    boundary_margin_px: float,
    min_track_hits: int,
    ordinary_miss_grace_frames: int,
    coverage_threshold: float,
    min_area_ratio: float,
) -> GateDecision:
    x1, y1, x2, y2 = predicted_box
    vx, vy = velocity
    margin = boundary_margin_px
    predicted_exit = (
        (x1 <= margin and vx < 0)
        or (x2 >= image_width - margin and vx > 0)
        or (y1 <= margin and vy < 0)
        or (y2 >= image_height - margin and vy > 0)
    )
    if predicted_exit:
        return GateDecision(False, True, 0.0, "predicted_exit")

    if track_hits < min_track_hits:
        return GateDecision(False, False, 0.0, "immature_track")

    target_area = box_area(predicted_box)
    scores = [
        target_coverage(predicted_box, box)
        for box in unclaimed_boxes
        if target_area > 0.0 and box_area(box) / target_area >= min_area_ratio
    ]
    support = max(scores, default=0.0)
    if support >= coverage_threshold:
        return GateDecision(True, False, support, "occluder_overlap")
    if frames_missing <= ordinary_miss_grace_frames:
        return GateDecision(True, False, support, "bounded_grace")
    return GateDecision(False, False, support, "insufficient_occlusion_evidence")

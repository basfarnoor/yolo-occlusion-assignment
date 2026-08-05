"""Target--occluder relational state, scoring, and clearance prediction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from oatm.tracking.geometry import box_area

Box = tuple[float, float, float, float]


class RelationPhase(StrEnum):
    FORMING = "FORMING"
    ACTIVE = "ACTIVE"
    CLEARING = "CLEARING"
    RESOLVED = "RESOLVED"
    FAILED = "FAILED"


def intersection_area(a: Box, b: Box) -> float:
    width = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return width * height


def target_coverage(target: Box, occluder: Box) -> float:
    area = box_area(target)
    return 0.0 if area <= 0.0 else intersection_area(target, occluder) / area


def translate_box(box: Box, velocity: tuple[float, float], steps: int) -> Box:
    dx, dy = velocity[0] * steps, velocity[1] * steps
    return (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)


def relative_geometry(target_box: Box, occluder_box: Box) -> tuple[float, float, float, float]:
    """Encode a target box in an occluder-centric coordinate system."""
    target_width = max(target_box[2] - target_box[0], 1e-6)
    target_height = max(target_box[3] - target_box[1], 1e-6)
    occluder_width = max(occluder_box[2] - occluder_box[0], 1e-6)
    occluder_height = max(occluder_box[3] - occluder_box[1], 1e-6)
    target_center = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
    occluder_center = ((occluder_box[0] + occluder_box[2]) / 2, (occluder_box[1] + occluder_box[3]) / 2)
    return (
        (target_center[0] - occluder_center[0]) / occluder_width,
        (target_center[1] - occluder_center[1]) / occluder_height,
        target_width / occluder_width,
        target_height / occluder_height,
    )


def target_from_relative_geometry(geometry: tuple[float, float, float, float], occluder_box: Box) -> Box:
    """Decode an occluder-relative target box using only current causal state."""
    occluder_width = max(occluder_box[2] - occluder_box[0], 1e-6)
    occluder_height = max(occluder_box[3] - occluder_box[1], 1e-6)
    center_x = (occluder_box[0] + occluder_box[2]) / 2 + geometry[0] * occluder_width
    center_y = (occluder_box[1] + occluder_box[3]) / 2 + geometry[1] * occluder_height
    width, height = geometry[2] * occluder_width, geometry[3] * occluder_height
    return (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2)


def expected_clearance_frames(
    target_box: Box,
    occluder_box: Box,
    target_velocity: tuple[float, float],
    occluder_velocity: tuple[float, float],
    coverage_threshold: float,
    horizon: int,
) -> int | None:
    for step in range(1, horizon + 1):
        target = translate_box(target_box, target_velocity, step)
        occluder = translate_box(occluder_box, occluder_velocity, step)
        if target_coverage(target, occluder) < coverage_threshold:
            return step
    return None


@dataclass(frozen=True)
class RelationFeatures:
    coverage: float
    scale_score: float
    depth_order_score: float
    trajectory_score: float
    maturity_score: float
    camera_quality: float


def relation_features(
    target_box: Box,
    occluder_box: Box,
    target_velocity: tuple[float, float],
    occluder_velocity: tuple[float, float],
    target_hits: int,
    camera_quality: float,
) -> RelationFeatures:
    coverage = target_coverage(target_box, occluder_box)
    target_area = max(box_area(target_box), 1e-6)
    area_ratio = box_area(occluder_box) / target_area
    scale_score = min(1.0, area_ratio)
    depth_order = 1.0 if occluder_box[3] >= target_box[3] else 0.35
    relative_speed = float(
        np.hypot(target_velocity[0] - occluder_velocity[0], target_velocity[1] - occluder_velocity[1])
    )
    trajectory_score = 1.0 / (1.0 + relative_speed / 25.0)
    maturity_score = min(1.0, target_hits / 4.0)
    return RelationFeatures(
        coverage, scale_score, depth_order, trajectory_score, maturity_score, camera_quality
    )


def occlusion_probability(features: RelationFeatures) -> float:
    coverage_score = min(1.0, features.coverage / 0.5)
    quality = features.camera_quality if features.camera_quality > 0.0 else 0.5
    score = (
        0.34 * coverage_score
        + 0.18 * features.scale_score
        + 0.16 * features.depth_order_score
        + 0.14 * features.trajectory_score
        + 0.12 * features.maturity_score
        + 0.06 * quality
    )
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class OcclusionRelation:
    target_track_id: int
    occluder_track_id: int
    start_frame: int
    last_supported_frame: int
    phase: RelationPhase
    probability: float
    coverage: float
    expected_clearance_frames: int | None
    relative_geometry: tuple[float, float, float, float]
    clearing_age_frames: int = 0

    def update_support(
        self, frame_index: int, probability: float, coverage: float, clearance: int | None
    ) -> None:
        self.last_supported_frame = frame_index
        self.probability = probability
        self.coverage = coverage
        self.expected_clearance_frames = clearance
        self.phase = RelationPhase.CLEARING if clearance == 1 else RelationPhase.ACTIVE
        self.clearing_age_frames = self.clearing_age_frames + 1 if self.phase == RelationPhase.CLEARING else 0

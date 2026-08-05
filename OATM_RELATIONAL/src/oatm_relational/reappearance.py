"""One-to-one relation-aware association after ByteTrack's two rounds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from oatm.tracking.geometry import box_area, center_error

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class ReappearanceCandidate:
    detection_index: int
    track_index: int
    score: float


def reappearance_score(
    detection_box: Box,
    predicted_box: Box,
    localization_uncertainty: float,
    expected_clearance_frames: int | None,
) -> float:
    diagonal = max(
        10.0, float(np.hypot(predicted_box[2] - predicted_box[0], predicted_box[3] - predicted_box[1]))
    )
    uncertainty_radius = float(np.sqrt(max(localization_uncertainty, 1.0))) * 2.0
    radius = min(150.0, max(diagonal * 1.5, uncertainty_radius))
    distance_score = max(0.0, 1.0 - center_error(detection_box, predicted_box) / radius)
    area_ratio = min(box_area(detection_box), box_area(predicted_box)) / max(
        max(box_area(detection_box), box_area(predicted_box)), 1e-6
    )
    timing_score = 1.0 if expected_clearance_frames in (None, 0, 1, 2) else 0.65
    return 0.60 * distance_score + 0.25 * area_ratio + 0.15 * timing_score


def associate_reappearances(
    detections: list[dict],
    detection_indices: list[int],
    tracks: list,
    track_indices: list[int],
    predicted_boxes: list[Box],
    threshold: float,
    visible_track_ids: set[int] | None = None,
) -> list[ReappearanceCandidate]:
    visible_track_ids = visible_track_ids or set()
    candidates = []
    for detection_index in detection_indices:
        detection = detections[detection_index]
        box = (detection["x1"], detection["y1"], detection["x2"], detection["y2"])
        for track_index in track_indices:
            track = tracks[track_index]
            if detection["class"] != track.kalman.class_name or track.relation is None:
                continue
            phase = getattr(track.relation.phase, "value", track.relation.phase)
            occluder_still_visible = track.relation.occluder_track_id in visible_track_ids
            if getattr(track, "hidden_frames", 1) <= 0 or (
                occluder_still_visible and phase not in {"CLEARING", "RESOLVED"}
            ):
                continue

            score = reappearance_score(
                box,
                predicted_boxes[track_index],
                float(np.trace(track.kalman.P)),
                track.relation.expected_clearance_frames,
            )
            if score >= threshold:
                candidates.append(ReappearanceCandidate(detection_index, track_index, score))
    selected = []
    used_detections: set[int] = set()
    used_tracks: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        if candidate.detection_index in used_detections or candidate.track_index in used_tracks:
            continue
        selected.append(candidate)
        used_detections.add(candidate.detection_index)
        used_tracks.add(candidate.track_index)
    return selected

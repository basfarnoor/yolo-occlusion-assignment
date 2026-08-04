"""Phase 2: global -> ego -> camera -> image projection of nuScenes 3D
annotations into CAM_FRONT. Privileged offline evaluation evidence only --
this module's output is never an input to online tracking (METHODOLOGY.md's
camera-only boundary; see also the `Build events / Primary inference /
Evaluate` table in IMPLEMENTATION_PLAN.md).

Geometry follows the standard nuScenes global -> ego-vehicle -> camera
transform chain (translation + quaternion rotation composed from ego pose and
calibrated-sensor records) and a pinhole projection with the camera's
intrinsic matrix. This is public, dataset-standard box-projection math.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint
from shapely.geometry import box as shapely_box

MIN_CAMERA_DISTANCE_M = 0.10  # a corner behind or at the camera plane is invalid

# MVP scope (IMPLEMENTATION_PLAN.md Phase 0): cars and pedestrians first.
# Every annotation is still projected and kept with its original_category --
# evaluation_class is only None so later phases can filter cleanly instead
# of this layer silently dropping rows a future phase might want.
EVALUATION_CLASS_MAP: dict[str, str] = {
    "vehicle.car": "car",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.personal_mobility": "pedestrian",
    "human.pedestrian.stroller": "pedestrian",
    "human.pedestrian.wheelchair": "pedestrian",
}


def map_evaluation_class(original_category: str) -> str | None:
    return EVALUATION_CLASS_MAP.get(original_category)


@dataclass
class ProjectedBox:
    x1: float
    y1: float
    x2: float
    y2: float
    center_depth_m: float
    truncation_fraction: float
    projection_status: str  # "accepted", "behind_camera", "outside_image", "degenerate"


def box_corners(translation: list[float], size: list[float], rotation: list[float]) -> np.ndarray:
    """Returns the box's 8 corners (3xN) in the same frame as `translation`.
    nuScenes size order is (width, length, height)."""
    width, length, h = size
    x_corners = length / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = width / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    corners = Quaternion(rotation).rotation_matrix @ corners
    corners = corners + np.array(translation).reshape(3, 1)
    return corners


def global_to_camera(
    corners_global: np.ndarray,
    ego_translation: list[float], ego_rotation: list[float],
    cs_translation: list[float], cs_rotation: list[float],
) -> np.ndarray:
    """Global frame -> ego-vehicle frame -> camera frame."""
    corners = corners_global - np.array(ego_translation).reshape(3, 1)
    corners = Quaternion(ego_rotation).inverse.rotation_matrix @ corners
    corners = corners - np.array(cs_translation).reshape(3, 1)
    corners = Quaternion(cs_rotation).inverse.rotation_matrix @ corners
    return corners


def view_points(corners_cam: np.ndarray, camera_intrinsic: list[list[float]]) -> np.ndarray:
    """Pinhole projection: 3D camera-frame points -> 2D pixel coordinates."""
    intrinsic = np.array(camera_intrinsic)
    points = intrinsic @ corners_cam
    points = points[:2, :] / points[2:3, :]
    return points


def post_process_coords(
    corner_coords: list[tuple[float, float]], image_size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    """Intersects the projected corner polygon's convex hull with the image
    rectangle. Returns None if the box's visible footprint does not overlap
    the image at all."""
    polygon = MultiPoint(corner_coords).convex_hull
    canvas = shapely_box(0, 0, image_size[0], image_size[1])
    if not polygon.intersects(canvas):
        return None
    intersection = polygon.intersection(canvas)
    coords = np.array(list(intersection.exterior.coords))
    return (
        float(coords[:, 0].min()), float(coords[:, 1].min()),
        float(coords[:, 0].max()), float(coords[:, 1].max()),
    )


def project_annotation(
    annotation: dict, ego_pose: dict, calibrated_sensor: dict, image_size: tuple[int, int] = (1600, 900),
) -> ProjectedBox:
    """Projects one sample_annotation into CAM_FRONT pixel space, or marks it
    rejected with a written status so difficult cases stay auditable."""
    corners_global = box_corners(annotation["translation"], annotation["size"], annotation["rotation"])
    corners_cam = global_to_camera(
        corners_global, ego_pose["translation"], ego_pose["rotation"],
        calibrated_sensor["translation"], calibrated_sensor["rotation"],
    )

    depths = corners_cam[2, :]
    if np.all(depths <= MIN_CAMERA_DISTANCE_M):
        return ProjectedBox(0, 0, 0, 0, 0.0, 0.0, "behind_camera")

    in_front = depths > MIN_CAMERA_DISTANCE_M
    if not np.all(in_front):
        corners_cam = corners_cam[:, in_front]
        if corners_cam.shape[1] < 3:
            return ProjectedBox(0, 0, 0, 0, 0.0, 0.0, "behind_camera")

    mean_depth = float(corners_cam[2, :].mean())
    points_2d = view_points(corners_cam, calibrated_sensor["camera_intrinsic"])
    corner_coords = [(float(points_2d[0, i]), float(points_2d[1, i])) for i in range(points_2d.shape[1])]

    unclipped_x1 = min(x for x, _ in corner_coords)
    unclipped_y1 = min(y for _, y in corner_coords)
    unclipped_x2 = max(x for x, _ in corner_coords)
    unclipped_y2 = max(y for _, y in corner_coords)
    unclipped_area = max(0.0, unclipped_x2 - unclipped_x1) * max(0.0, unclipped_y2 - unclipped_y1)

    clipped = post_process_coords(corner_coords, image_size)
    if clipped is None:
        return ProjectedBox(0, 0, 0, 0, mean_depth, 1.0, "outside_image")

    x1, y1, x2, y2 = clipped
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0 or not np.isfinite(area):
        return ProjectedBox(0, 0, 0, 0, mean_depth, 1.0, "degenerate")

    truncation_fraction = 1.0 - (area / unclipped_area) if unclipped_area > 0 else 0.0
    truncation_fraction = min(max(truncation_fraction, 0.0), 1.0)

    return ProjectedBox(x1, y1, x2, y2, mean_depth, truncation_fraction, "accepted")

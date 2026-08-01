"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Projects nuScenes official 3D annotation boxes into the CAM_FRONT image plane
at annotated keyframes. This is *privileged offline evaluation evidence only*:
the resulting 2D boxes are never given to the online ByteTrack/SORT tracker as
input. They exist purely so the tracker can be scored against an independent
reference instead of against its own output (the flaw identified in
Assignment 3's mentor review -- see reuse_audit.md, required repair #1).

Geometry follows the standard nuScenes global -> ego-vehicle -> camera
transform chain (translation + quaternion rotation composed from
`ego_pose.json` and `calibrated_sensor.json`) and a pinhole projection with the
camera's intrinsic matrix. This is public, dataset-standard box-projection
math, not the ByteTrack authors' tracker code.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box as shapely_box

MIN_CAMERA_DISTANCE_M = 0.10  # a corner behind or at the camera plane is invalid


@dataclass
class ProjectedBox:
    instance_token: str
    category: str
    visibility_level: str
    x1: float
    y1: float
    x2: float
    y2: float
    unclipped_x1: float
    unclipped_y1: float
    unclipped_x2: float
    unclipped_y2: float
    was_clipped: bool
    depth_m: float
    num_lidar_pts: int
    num_radar_pts: int
    rejected: bool
    reject_reason: str


def box_corners(translation: list[float], size: list[float], rotation: list[float]) -> np.ndarray:
    """Returns the box's 8 corners (3xN) in the same frame as `translation`.
    nuScenes size order is (width, length, height); corners follow the
    devkit's standard corner ordering."""
    w, l, h = size
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))
    corners = Quaternion(rotation).rotation_matrix @ corners
    corners = corners + np.array(translation).reshape(3, 1)
    return corners


def global_to_camera(corners_global: np.ndarray, ego_translation: list[float], ego_rotation: list[float],
                      cs_translation: list[float], cs_rotation: list[float]) -> np.ndarray:
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


def post_process_coords(corner_coords: list[tuple[float, float]], image_size: tuple[int, int]
                         ) -> tuple[float, float, float, float] | None:
    """Intersects the projected corner polygon's convex hull with the image
    rectangle. Returns None if the box's visible footprint does not overlap
    the image at all (e.g. it projects entirely outside the frame)."""
    polygon = MultiPoint(corner_coords).convex_hull
    canvas = shapely_box(0, 0, image_size[0], image_size[1])
    if not polygon.intersects(canvas):
        return None
    intersection = polygon.intersection(canvas)
    coords = np.array(list(intersection.exterior.coords))
    return float(coords[:, 0].min()), float(coords[:, 1].min()), float(coords[:, 0].max()), float(coords[:, 1].max())


def project_annotation(annotation: dict, ego_pose: dict, calibrated_sensor: dict, category: str,
                        visibility_level: str, image_size: tuple[int, int] = (1600, 900)) -> ProjectedBox:
    """Projects one sample_annotation into CAM_FRONT pixel space, or marks it
    rejected with a written reason (behind camera / outside frame / invalid
    area) so difficult cases stay auditable rather than silently dropped."""
    instance_token = annotation["instance_token"]
    base_kwargs = dict(
        instance_token=instance_token, category=category, visibility_level=visibility_level,
        x1=0.0, y1=0.0, x2=0.0, y2=0.0, unclipped_x1=0.0, unclipped_y1=0.0, unclipped_x2=0.0, unclipped_y2=0.0,
        was_clipped=False, depth_m=0.0, num_lidar_pts=annotation.get("num_lidar_pts", 0),
        num_radar_pts=annotation.get("num_radar_pts", 0),
    )

    corners_global = box_corners(annotation["translation"], annotation["size"], annotation["rotation"])
    corners_cam = global_to_camera(
        corners_global, ego_pose["translation"], ego_pose["rotation"],
        calibrated_sensor["translation"], calibrated_sensor["rotation"])

    depths = corners_cam[2, :]
    if np.all(depths <= MIN_CAMERA_DISTANCE_M):
        return ProjectedBox(**base_kwargs, rejected=True, reject_reason="all corners behind or at the camera plane")

    in_front = depths > MIN_CAMERA_DISTANCE_M
    if not np.all(in_front):
        # Some corners are behind the camera plane (the box straddles it).
        # Keep only the in-front corners for a defensible partial projection
        # rather than extrapolating through the camera center.
        corners_cam = corners_cam[:, in_front]
        if corners_cam.shape[1] < 3:
            return ProjectedBox(**base_kwargs, rejected=True,
                                 reject_reason="fewer than 3 corners in front of camera; projection undefined")

    mean_depth = float(corners_cam[2, :].mean())
    points_2d = view_points(corners_cam, calibrated_sensor["camera_intrinsic"])
    corner_coords = [(float(points_2d[0, i]), float(points_2d[1, i])) for i in range(points_2d.shape[1])]

    unclipped_x1 = min(x for x, _ in corner_coords)
    unclipped_y1 = min(y for _, y in corner_coords)
    unclipped_x2 = max(x for x, _ in corner_coords)
    unclipped_y2 = max(y for _, y in corner_coords)

    clipped = post_process_coords(corner_coords, image_size)
    if clipped is None:
        return ProjectedBox(**base_kwargs, rejected=True, reject_reason="projected box falls entirely outside the image")

    x1, y1, x2, y2 = clipped
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area <= 0.0 or not np.isfinite(area):
        return ProjectedBox(**base_kwargs, rejected=True, reject_reason="clipped box has non-positive or non-finite area")

    was_clipped = (
        abs(x1 - unclipped_x1) > 1e-6 or abs(y1 - unclipped_y1) > 1e-6 or
        abs(x2 - unclipped_x2) > 1e-6 or abs(y2 - unclipped_y2) > 1e-6
    )

    return ProjectedBox(
        instance_token=instance_token, category=category, visibility_level=visibility_level,
        x1=x1, y1=y1, x2=x2, y2=y2,
        unclipped_x1=unclipped_x1, unclipped_y1=unclipped_y1, unclipped_x2=unclipped_x2, unclipped_y2=unclipped_y2,
        was_clipped=was_clipped, depth_m=mean_depth,
        num_lidar_pts=annotation.get("num_lidar_pts", 0), num_radar_pts=annotation.get("num_radar_pts", 0),
        rejected=False, reject_reason="",
    )

"""Tests for src/projection.py (Task 4 required tests).
ByteTrack paper reference: Zhang et al., ECCV 2022 (arxiv.org/abs/2110.06864)."""
import numpy as np

from projection import box_corners, global_to_camera, project_annotation, view_points

IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z -- no rotation
IMAGE_SIZE = (1600, 900)
SIMPLE_INTRINSIC = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]


def _annotation(translation, size=(2.0, 4.0, 1.5), rotation=IDENTITY_ROTATION, instance_token="inst-1"):
    return {
        "instance_token": instance_token,
        "translation": list(translation),
        "size": list(size),
        "rotation": list(rotation),
        "num_lidar_pts": 10,
        "num_radar_pts": 2,
    }


def _identity_pose(translation=(0.0, 0.0, 0.0)):
    return {"translation": list(translation), "rotation": IDENTITY_ROTATION}


def _identity_calibrated_sensor(translation=(0.0, 0.0, 0.0)):
    return {"translation": list(translation), "rotation": IDENTITY_ROTATION,
            "camera_intrinsic": SIMPLE_INTRINSIC}


def test_known_coordinate_transform_places_box_directly_ahead():
    """A box 10m directly ahead of an identity ego pose and identity camera
    calibration should project near the principal point (cx, cy)."""
    ann = _annotation(translation=(0.0, 0.0, 10.0), size=(1.0, 1.0, 1.0))
    ego = _identity_pose()
    cs = _identity_calibrated_sensor()
    pb = project_annotation(ann, ego, cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    assert not pb.rejected
    center_x = (pb.x1 + pb.x2) / 2
    center_y = (pb.y1 + pb.y2) / 2
    assert abs(center_x - 800.0) < 5.0
    assert abs(center_y - 450.0) < 5.0
    assert abs(pb.depth_m - 10.0) < 1e-6


def test_behind_camera_box_is_rejected():
    """A box behind the camera (negative Z in camera frame) must be rejected,
    not silently projected to a nonsense location."""
    ann = _annotation(translation=(0.0, 0.0, -10.0), size=(1.0, 1.0, 1.0))
    ego = _identity_pose()
    cs = _identity_calibrated_sensor()
    pb = project_annotation(ann, ego, cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    assert pb.rejected
    assert "behind" in pb.reject_reason


def test_box_partially_outside_frame_is_clipped_to_image_boundary():
    """A box straddling the left image edge must be clipped to x1=0, with
    was_clipped=True and the unclipped extent preserved for audit. nuScenes
    size order is (width, length, height); length maps to the local X axis
    (box_corners' x_corners), so a large *length* extends the box along
    camera-frame X here, since ego/calibration are identity in this test."""
    ann = _annotation(translation=(-5.0, 0.0, 10.0), size=(1.0, 20.0, 1.0))
    ego = _identity_pose()
    cs = _identity_calibrated_sensor()
    pb = project_annotation(ann, ego, cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    assert not pb.rejected
    assert pb.was_clipped
    assert pb.x1 == 0.0
    assert pb.unclipped_x1 < 0.0


def test_fully_outside_frame_box_is_rejected():
    ann = _annotation(translation=(1000.0, 0.0, 10.0), size=(1.0, 1.0, 1.0))
    ego = _identity_pose()
    cs = _identity_calibrated_sensor()
    pb = project_annotation(ann, ego, cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    assert pb.rejected
    assert "outside the image" in pb.reject_reason


def test_accepted_box_has_positive_finite_area():
    ann = _annotation(translation=(0.0, 0.0, 10.0), size=(2.0, 4.0, 1.5))
    ego = _identity_pose()
    cs = _identity_calibrated_sensor()
    pb = project_annotation(ann, ego, cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    assert not pb.rejected
    area = (pb.x2 - pb.x1) * (pb.y2 - pb.y1)
    assert area > 0.0
    assert np.isfinite(area)


def test_ego_translation_shifts_the_projected_box():
    """Moving the ego vehicle forward by the same amount the box is ahead
    should move the box away from the camera (larger apparent... i.e. smaller
    depth), proving the ego transform is actually applied."""
    ann = _annotation(translation=(0.0, 0.0, 10.0), size=(1.0, 1.0, 1.0))
    cs = _identity_calibrated_sensor()

    pb_no_move = project_annotation(ann, _identity_pose(), cs, "vehicle.car", "v80-100", IMAGE_SIZE)
    pb_ego_forward = project_annotation(ann, _identity_pose(translation=(0.0, 0.0, 5.0)), cs,
                                         "vehicle.car", "v80-100", IMAGE_SIZE)
    assert pb_ego_forward.depth_m < pb_no_move.depth_m


def test_instance_and_category_identity_preserved_through_projection():
    ann = _annotation(translation=(0.0, 0.0, 10.0), instance_token="specific-instance-token-123")
    pb = project_annotation(ann, _identity_pose(), _identity_calibrated_sensor(),
                             "human.pedestrian.adult", "v60-80", IMAGE_SIZE)
    assert pb.instance_token == "specific-instance-token-123"
    assert pb.category == "human.pedestrian.adult"
    assert pb.visibility_level == "v60-80"


def test_box_corners_returns_eight_points_centered_on_translation():
    corners = box_corners(translation=[1.0, 2.0, 3.0], size=[2.0, 2.0, 2.0], rotation=IDENTITY_ROTATION)
    assert corners.shape == (3, 8)
    assert np.allclose(corners.mean(axis=1), [1.0, 2.0, 3.0])


def test_view_points_projects_point_on_axis_to_principal_point():
    points = view_points(np.array([[0.0], [0.0], [5.0]]), SIMPLE_INTRINSIC)
    assert abs(points[0, 0] - 800.0) < 1e-6
    assert abs(points[1, 0] - 450.0) < 1e-6


def test_global_to_camera_is_identity_when_all_poses_are_identity():
    corners_global = np.array([[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]])
    corners_cam = global_to_camera(corners_global, [0, 0, 0], IDENTITY_ROTATION, [0, 0, 0], IDENTITY_ROTATION)
    assert np.allclose(corners_cam, corners_global)

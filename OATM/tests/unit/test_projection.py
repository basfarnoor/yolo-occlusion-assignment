"""Required Task 3 tests: known coordinate transform, behind-camera
rejection, image-boundary clipping, positive finite box area, deterministic
ordering, identity preservation, and an independent cross-check using a
different rotation library (scipy) as the "independent reference" this task
requires."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from oatm.dataset.projection import (
    box_corners,
    global_to_camera,
    map_evaluation_class,
    project_annotation,
    view_points,
)

IDENTITY_ROTATION = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z
IMAGE_SIZE = (1600, 900)
SIMPLE_INTRINSIC = [[1000.0, 0.0, 800.0], [0.0, 1000.0, 450.0], [0.0, 0.0, 1.0]]


def _annotation(translation, size=(2.0, 4.0, 1.5), rotation=IDENTITY_ROTATION):
    return {"translation": list(translation), "size": list(size), "rotation": list(rotation),
            "num_lidar_pts": 10, "num_radar_pts": 2}


def _pose(translation=(0.0, 0.0, 0.0), rotation=IDENTITY_ROTATION):
    return {"translation": list(translation), "rotation": list(rotation)}


def _calibrated_sensor(translation=(0.0, 0.0, 0.0), rotation=IDENTITY_ROTATION):
    return {
        "translation": list(translation), "rotation": list(rotation),
        "camera_intrinsic": SIMPLE_INTRINSIC,
    }


def test_known_coordinate_transform_places_box_directly_ahead_at_the_principal_point():
    ann = _annotation(translation=(0.0, 0.0, 10.0), size=(1.0, 1.0, 1.0))
    projected = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    assert projected.projection_status == "accepted"
    center_x = (projected.x1 + projected.x2) / 2
    center_y = (projected.y1 + projected.y2) / 2
    assert abs(center_x - 800.0) < 5.0
    assert abs(center_y - 450.0) < 5.0
    assert abs(projected.center_depth_m - 10.0) < 1e-6


def test_behind_camera_box_is_rejected():
    ann = _annotation(translation=(0.0, 0.0, -10.0))
    projected = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    assert projected.projection_status == "behind_camera"


def test_box_partially_outside_frame_is_clipped_with_nonzero_truncation():
    # length (local X) extends the box along camera-frame X here since ego/cs are identity.
    ann = _annotation(translation=(-5.0, 0.0, 10.0), size=(1.0, 20.0, 1.0))
    projected = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    assert projected.projection_status == "accepted"
    assert projected.x1 == 0.0
    assert projected.truncation_fraction > 0.0


def test_fully_outside_frame_box_is_rejected():
    ann = _annotation(translation=(1000.0, 0.0, 10.0))
    projected = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    assert projected.projection_status == "outside_image"


def test_accepted_box_has_positive_finite_area():
    ann = _annotation(translation=(0.0, 0.0, 10.0))
    projected = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    area = (projected.x2 - projected.x1) * (projected.y2 - projected.y1)
    assert area > 0.0
    assert np.isfinite(area)


def test_deterministic_ordering_same_input_same_output():
    ann = _annotation(translation=(1.0, 2.0, 15.0))
    a = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    b = project_annotation(ann, _pose(), _calibrated_sensor(), IMAGE_SIZE)
    assert (a.x1, a.y1, a.x2, a.y2, a.center_depth_m) == (b.x1, b.y1, b.x2, b.y2, b.center_depth_m)


@pytest.mark.parametrize("original_category,expected", [
    ("vehicle.car", "car"),
    ("human.pedestrian.adult", "pedestrian"),
    ("movable_object.trafficcone", None),
    ("vehicle.bicycle", None),  # not in MVP scope yet -- preserved but unmapped
])
def test_evaluation_class_mapping_matches_mvp_scope(original_category, expected):
    assert map_evaluation_class(original_category) == expected


def test_box_corners_returns_eight_points_centered_on_translation():
    corners = box_corners(translation=[1.0, 2.0, 3.0], size=[2.0, 2.0, 2.0], rotation=IDENTITY_ROTATION)
    assert corners.shape == (3, 8)
    assert np.allclose(corners.mean(axis=1), [1.0, 2.0, 3.0])


def test_view_points_projects_point_on_axis_to_principal_point():
    points = view_points(np.array([[0.0], [0.0], [5.0]]), SIMPLE_INTRINSIC)
    assert abs(points[0, 0] - 800.0) < 1e-6
    assert abs(points[1, 0] - 450.0) < 1e-6


# --- Independent cross-check (required: "compare... with an independent reference") ---
# Re-implements the global -> ego -> camera transform using scipy's Rotation
# instead of pyquaternion -- a different library, same underlying physics.
# Agreement between the two confirms the primary implementation isn't just
# internally self-consistent but numerically correct.

def _scipy_global_to_camera(corners_global, ego_translation, ego_rotation_wxyz,
                              cs_translation, cs_rotation_wxyz):
    def wxyz_to_xyzw(q):
        w, x, y, z = q
        return [x, y, z, w]

    corners = corners_global - np.array(ego_translation).reshape(3, 1)
    ego_rot = Rotation.from_quat(wxyz_to_xyzw(ego_rotation_wxyz))
    corners = ego_rot.inv().as_matrix() @ corners

    corners = corners - np.array(cs_translation).reshape(3, 1)
    cs_rot = Rotation.from_quat(wxyz_to_xyzw(cs_rotation_wxyz))
    corners = cs_rot.inv().as_matrix() @ corners
    return corners


def _euler_z_to_wxyz(degrees: float) -> list[float]:
    """A proper, guaranteed-unit quaternion for a rotation of `degrees` about
    Z, generated by scipy itself so both implementations start from an
    identical, exactly-normalized rotation."""
    x, y, z, w = Rotation.from_euler("z", degrees, degrees=True).as_quat()
    return [w, x, y, z]


def _euler_xyz_to_wxyz(rx: float, ry: float, rz: float) -> list[float]:
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()
    return [w, x, y, z]


@pytest.mark.parametrize("ego_translation,ego_rotation,cs_translation,cs_rotation", [
    ((0.0, 0.0, 0.0), IDENTITY_ROTATION, (0.0, 0.0, 0.0), IDENTITY_ROTATION),
    ((10.0, -5.0, 0.0), _euler_z_to_wxyz(28.6), (1.5, 0.1, 1.4), _euler_xyz_to_wxyz(-90, 0, -90)),
    ((-3.0, 8.0, 0.2), _euler_z_to_wxyz(90.0), (1.7, 0.0, 1.5), _euler_xyz_to_wxyz(-88, 2, -91)),
])
def test_global_to_camera_matches_independent_scipy_implementation(
    ego_translation, ego_rotation, cs_translation, cs_rotation
):
    corners_global = box_corners([5.0, 2.0, 0.0], [2.0, 4.0, 1.5], [0.9, 0.1, 0.0, 0.0])

    mine = global_to_camera(corners_global, ego_translation, ego_rotation, cs_translation, cs_rotation)
    independent = _scipy_global_to_camera(
        corners_global, ego_translation, ego_rotation, cs_translation, cs_rotation
    )

    assert np.allclose(mine, independent, atol=1e-9), (
        "pyquaternion-based transform disagrees with an independent scipy-based "
        "reimplementation of the same global->ego->camera transform"
    )

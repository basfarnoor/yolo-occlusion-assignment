"""Required Task 6 test: identical and disjoint IoU cases (reused pattern
from Assignment 4's test_geometry.py)."""
from oatm.tracking.geometry import box_to_state, center_error, iou, state_to_box


def test_identical_boxes_have_iou_1():
    box = (10.0, 10.0, 50.0, 50.0)
    assert iou(box, box) == 1.0


def test_non_overlapping_boxes_have_iou_0():
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (100.0, 100.0, 110.0, 110.0)
    assert iou(box_a, box_b) == 0.0


def test_center_error_of_identical_boxes_is_zero():
    box = (10.0, 10.0, 50.0, 50.0)
    assert center_error(box, box) == 0.0


def test_center_error_matches_known_shift():
    box_a = (0.0, 0.0, 10.0, 10.0)
    box_b = (3.0, 4.0, 13.0, 14.0)
    assert abs(center_error(box_a, box_b) - 5.0) < 1e-6


def test_state_roundtrip_recovers_original_box():
    box = (12.0, 30.0, 60.0, 90.0)
    state = box_to_state(box)
    recovered = state_to_box(state)
    for a, b in zip(box, recovered):
        assert abs(a - b) < 1e-6

"""Required Task 6 test: one-to-one assignment (reused pattern from
Assignment 4's test_assignment.py)."""
from oatm.tracking.association import associate_detections_to_trackers


def test_two_detections_cannot_be_assigned_to_the_same_track():
    detections = [
        {"class": "car", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        {"class": "car", "x1": 1.0, "y1": 1.0, "x2": 11.0, "y2": 11.0},
    ]
    tracker_boxes = [(0.5, 0.5, 10.5, 10.5)]
    tracker_classes = ["car"]

    matches, unmatched_dets, _ = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.3)

    matched_tracker_indices = [t for _, t in matches]
    assert len(matched_tracker_indices) == len(set(matched_tracker_indices))
    assert len(matches) <= 1
    assert len(unmatched_dets) == 1


def test_incompatible_classes_never_match():
    detections = [{"class": "person", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]
    tracker_boxes = [(0.0, 0.0, 10.0, 10.0)]
    tracker_classes = ["car"]

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.3)

    assert matches == []
    assert unmatched_dets == [0]
    assert unmatched_trks == [0]


def test_below_threshold_iou_is_not_matched():
    detections = [{"class": "car", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]
    tracker_boxes = [(8.0, 8.0, 18.0, 18.0)]
    tracker_classes = ["car"]

    matches, _, _ = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.5)

    assert matches == []

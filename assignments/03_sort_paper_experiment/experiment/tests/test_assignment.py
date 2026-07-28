"""Tests for src/assignment.py. Paper: Bewley et al., ICIP 2016 (SORT)."""
from assignment import associate_detections_to_trackers


def test_two_detections_cannot_be_assigned_to_the_same_track():
    detections = [
        {"class": "car", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0},
        {"class": "car", "x1": 1.0, "y1": 1.0, "x2": 11.0, "y2": 11.0},  # near-identical to the first
    ]
    tracker_boxes = [(0.5, 0.5, 10.5, 10.5)]  # a single predicted track near both detections
    tracker_classes = ["car"]

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.3)

    matched_tracker_indices = [t for _, t in matches]
    assert len(matched_tracker_indices) == len(set(matched_tracker_indices)), \
        "no tracker index should appear in more than one match"
    assert len(matches) <= 1, "only one detection can be assigned to the single track"
    assert len(unmatched_dets) == 1, "the other detection must be left unmatched, not double-assigned"


def test_incompatible_classes_never_match():
    detections = [{"class": "person", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]
    tracker_boxes = [(0.0, 0.0, 10.0, 10.0)]  # perfect overlap, but...
    tracker_classes = ["car"]  # ...wrong class

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.3)

    assert matches == []
    assert unmatched_dets == [0]
    assert unmatched_trks == [0]


def test_below_threshold_iou_is_not_matched():
    detections = [{"class": "car", "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0}]
    tracker_boxes = [(8.0, 8.0, 18.0, 18.0)]  # small overlap only
    tracker_classes = ["car"]

    matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
        detections, tracker_boxes, tracker_classes, iou_threshold=0.5)

    assert matches == []

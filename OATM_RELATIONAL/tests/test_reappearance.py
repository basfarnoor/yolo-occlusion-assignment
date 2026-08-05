from types import SimpleNamespace

import numpy as np

from oatm_relational.reappearance import associate_reappearances


def test_reappearance_assignment_is_one_to_one():
    kalman = SimpleNamespace(class_name="car", P=np.eye(7) * 4)
    relation = SimpleNamespace(expected_clearance_frames=1, phase="CLEARING", occluder_track_id=2)
    tracks = [SimpleNamespace(kalman=kalman, relation=relation)]
    detections = [
        {"class": "car", "x1": 105, "y1": 100, "x2": 145, "y2": 140},
        {"class": "car", "x1": 110, "y1": 100, "x2": 150, "y2": 140},
    ]
    matches = associate_reappearances(detections, [0, 1], tracks, [0], [(100, 100, 140, 140)], threshold=0.2)
    assert len(matches) == 1
    assert matches[0].track_index == 0


def test_far_detection_is_rejected_even_with_large_uncertainty():
    kalman = SimpleNamespace(class_name="car", P=np.eye(7) * 10000)
    relation = SimpleNamespace(expected_clearance_frames=1, phase="CLEARING", occluder_track_id=2)
    tracks = [SimpleNamespace(kalman=kalman, relation=relation, hidden_frames=4)]
    detections = [{"class": "car", "x1": 500, "y1": 100, "x2": 540, "y2": 140}]
    matches = associate_reappearances(
        detections,
        [0],
        tracks,
        [0],
        [(100, 100, 140, 140)],
        threshold=0.55,
    )
    assert matches == []

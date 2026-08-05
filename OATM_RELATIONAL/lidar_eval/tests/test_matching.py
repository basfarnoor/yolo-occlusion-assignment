from __future__ import annotations

import pandas as pd
from lidar_eval.matching import _hungarian_pairs, evaluate_method


def test_hungarian_matching_is_one_to_one_and_class_scoped() -> None:
    predictions = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 8, "y1": 0, "x2": 18, "y2": 10},
    ]
    truth = [
        {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"x1": 9, "y1": 0, "x2": 19, "y2": 10},
    ]
    pairs = _hungarian_pairs(predictions, truth, iou_threshold=0.3)
    assert {(prediction, target) for prediction, target, _ in pairs} == {(0, 0), (1, 1)}


def test_evaluate_method_keeps_gt_denominator_and_records_unmatched_output() -> None:
    keyframes = pd.DataFrame(
        [
            {
                "scene_token": "scene",
                "sample_data_token": "frame",
                "frame_index": 0,
                "is_keyframe": True,
            }
        ]
    )
    truth = pd.DataFrame(
        [
            {
                "scene_token": "scene",
                "sample_data_token": "frame",
                "instance_token": "car-instance",
                "annotation_token": "annotation",
                "evaluation_class": "car",
                "visibility_token": "1",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "center_depth_m": 25.0,
                "num_lidar_pts": 0,
                "num_radar_pts": 0,
                "truncation_fraction": 0.0,
                "projection_status": "accepted",
            }
        ]
    )
    outputs = pd.DataFrame(
        [
            {
                "method_name": "method",
                "scene_token": "scene",
                "sample_data_token": "frame",
                "frame_index": 0,
                "track_id": 1,
                "class_name": "car",
                "state": "PREDICTED_HIDDEN",
                "evidence_source": "motion_prediction",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
            },
            {
                "method_name": "method",
                "scene_token": "scene",
                "sample_data_token": "frame",
                "frame_index": 0,
                "track_id": 2,
                "class_name": "car",
                "state": "OBSERVED_STRONG",
                "evidence_source": "strong_detection",
                "x1": 100.0,
                "y1": 100.0,
                "x2": 110.0,
                "y2": 110.0,
            },
        ]
    )

    matches, unmatched = evaluate_method(outputs, truth, keyframes, iou_threshold=0.3)

    assert len(matches) == 1
    assert bool(matches.iloc[0].matched)
    assert matches.iloc[0].track_id == 1
    assert matches.iloc[0].state == "PREDICTED_HIDDEN"
    assert unmatched.track_id.tolist() == [2]

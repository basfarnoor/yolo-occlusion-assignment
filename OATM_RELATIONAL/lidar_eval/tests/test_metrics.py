from __future__ import annotations

import pandas as pd
from lidar_eval.metrics import identity_metrics, stratified_metrics, summarize_method


def _match(frame_index: int, matched: bool, track_id: int | None) -> dict:
    return {
        "method_name": "method",
        "scene_token": "scene",
        "sample_data_token": f"frame-{frame_index}",
        "frame_index": frame_index,
        "instance_token": "instance",
        "annotation_token": f"annotation-{frame_index}",
        "evaluation_class": "car",
        "visibility_token": "1",
        "center_depth_m": 30.0,
        "num_lidar_pts": 0,
        "num_radar_pts": 0,
        "truncation_fraction": 0.1,
        "matched": matched,
        "track_id": track_id,
        "state": "PREDICTED_HIDDEN" if matched else None,
        "evidence_source": "motion_prediction" if matched else None,
        "iou": 0.5 if matched else None,
        "center_error_px": 2.0 if matched else None,
    }


def test_identity_metrics_count_switch_and_fragmentation() -> None:
    matches = pd.DataFrame(
        [_match(0, True, 1), _match(1, False, None), _match(2, True, 2)]
    )
    identity = identity_metrics(matches, prediction_count=2)
    assert identity["id_switches"] == 1
    assert identity["fragmentations"] == 1
    assert identity["trajectories"] == 1


def test_summary_separates_predictions_and_strata_keep_zero_point_rows() -> None:
    matches = pd.DataFrame([_match(0, True, 1), _match(1, False, None)])
    false_positives = pd.DataFrame(
        [
            {
                "state": "PREDICTED_HIDDEN",
                "method_name": "method",
                "scene_token": "scene",
                "track_id": 3,
            }
        ]
    )
    summary = summarize_method(matches, false_positives, iou_threshold=0.3)
    assert summary["tp"] == 1
    assert summary["fn"] == 1
    assert summary["fp"] == 1
    assert summary["predicted_hidden_tp"] == 1
    assert summary["predicted_hidden_fp"] == 1

    strata = stratified_metrics(
        matches,
        {
            "depth_edges_m": [20.0, 40.0],
            "lidar_point_edges": [1, 5],
            "truncation_threshold": 0.25,
        },
    )
    zero_points = strata[(strata.dimension == "lidar_support") & (strata.value == "0 points")]
    assert zero_points.iloc[0].gt_annotations == 2
    assert zero_points.iloc[0].recall == 0.5

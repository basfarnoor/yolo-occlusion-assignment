"""Method A tests: YOLO-only has no temporal memory at all."""
from oatm.tracking.yolo_only import run_yolo_only_frame


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def test_only_high_score_detections_are_reported():
    outputs = run_yolo_only_frame(
        [_det(0, 0, 40, 40, conf=0.9), _det(50, 0, 90, 40, conf=0.2)],
        high_score_threshold=0.5, scene_token="s", sample_data_token="sd", frame_index=0,
        method_name="yolo_only", run_id="r1",
    )
    assert len(outputs) == 1
    assert outputs[0].evidence_source == "strong_detection"
    assert outputs[0].state == "OBSERVED_STRONG"


def test_no_track_persists_past_a_single_frame():
    """Calling twice with the same box must not imply the same identity --
    YOLO-only has no cross-frame memory at all."""
    args = dict(high_score_threshold=0.5, scene_token="s", sample_data_token="sd",
                method_name="yolo_only", run_id="r1")
    frame1 = run_yolo_only_frame([_det(0, 0, 40, 40)], frame_index=0, **args)
    frame2 = run_yolo_only_frame([_det(0, 0, 40, 40)], frame_index=1, **args)
    # Each call is fully independent -- track_id is just a per-call index,
    # never a claim of persistent identity across frames.
    assert frame1[0].track_id == frame2[0].track_id == 0
    assert frame1[0].memory_age_frames == 0
    assert frame2[0].memory_age_frames == 0

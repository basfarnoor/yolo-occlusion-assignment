"""Smoke tests for the canonical typed data contracts (records.py).
These records aren't populated by any real pipeline yet (that starts in
later phases) -- this only proves the schemas themselves are well-formed
and reject invalid data."""
import pytest
from pydantic import ValidationError

from oatm.records import (
    DetectorObservationRecord,
    FrameIndexRecord,
    OcclusionEventRecord,
    ProjectedGroundTruthRecord,
    TrackerOutputRecord,
)


def test_frame_index_record_accepts_valid_data():
    record = FrameIndexRecord(
        scene_token="scene-1", sample_token="sample-1", sample_data_token="sd-1",
        timestamp_us=1234567890, frame_index=0, is_keyframe=True,
        image_path="samples/CAM_FRONT/x.jpg", prev_token=None, next_token="sd-2",
        calibrated_sensor_token="cs-1", ego_pose_token="ep-1",
    )
    assert record.schema_version == 1
    assert record.frame_index == 0


def test_frame_index_record_rejects_negative_frame_index():
    with pytest.raises(ValidationError):
        FrameIndexRecord(
            scene_token="scene-1", sample_token=None, sample_data_token="sd-1",
            timestamp_us=1, frame_index=-1, is_keyframe=False,
            image_path="x.jpg", prev_token=None, next_token=None,
            calibrated_sensor_token="cs-1", ego_pose_token="ep-1",
        )


def test_detector_observation_record_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        DetectorObservationRecord(
            scene_token="scene-1", sample_data_token="sd-1", frame_index=0, detection_id=0,
            model_name="yolo", model_weights_hash="abc", detected_class="car",
            confidence=1.5, x1=0, y1=0, x2=10, y2=10, inference_time_ms=5.0, cache_key="k",
        )


def test_tracker_output_record_state_and_evidence_are_independent_fields():
    record = TrackerOutputRecord(
        scene_token="scene-1", sample_data_token="sd-1", frame_index=3, method_name="oatm",
        run_id="run-1", track_id=7, state="PREDICTED_HIDDEN", evidence_source=None,
        x1=0, y1=0, x2=10, y2=10, detector_confidence=None,
        existence_confidence=0.6, identity_confidence=0.9, localization_uncertainty=12.5,
        memory_age_frames=4, memory_age_seconds=0.33, termination_reason=None,
    )
    assert record.state == "PREDICTED_HIDDEN"
    assert record.evidence_source is None, "a hidden prediction must never carry a detection evidence source"


def test_occlusion_event_record_requires_a_split():
    with pytest.raises(ValidationError):
        OcclusionEventRecord(
            event_id="e1", scene_token="scene-1", instance_token="inst-1",
            pre_frame_index=0, start_frame_index=1, end_frame_index=2, post_frame_index=3,
            event_source="natural", visibility_pattern="high-low-high",
            review_status="accepted",
        )  # missing required `split`


def test_projected_ground_truth_record_rejects_negative_truncation():
    with pytest.raises(ValidationError):
        ProjectedGroundTruthRecord(
            scene_token="scene-1", sample_data_token="sd-1", instance_token="inst-1",
            annotation_token="ann-1", original_category="vehicle.car", evaluation_class="car",
            visibility_token="4", x1=0, y1=0, x2=10, y2=10, center_depth_m=5.0,
            num_lidar_pts=3, num_radar_pts=0, truncation_fraction=-0.1,
            projection_status="accepted",
        )

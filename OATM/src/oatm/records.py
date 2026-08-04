"""Canonical typed data contracts for OATM.

These mirror the "Canonical data contracts" section of
OATM/IMPLEMENTATION_PLAN.md exactly. Defining them now, in Phase 0, means
every later phase (frame indexing, projection, detection, event mining,
tracking) writes to the same agreed-upon shape instead of improvising its own
row format. No phase is implemented yet -- these are empty containers ready
to be filled in.

Every record carries `schema_version` so a future format change is visible
and explicit rather than silently breaking older saved data.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

EvidenceSource = str  # one of: "strong_detection", "weak_detection", "motion_prediction", or None
TrackState = str  # one of: OBSERVED_STRONG, OBSERVED_WEAK, PREDICTED_HIDDEN, LOST, EXITED
EventSource = str  # one of: "natural", "controlled_visual", "detector_intervention"


class FrameIndexRecord(BaseModel):
    """One row per CAM_FRONT frame (Phase 1)."""

    schema_version: int = 1
    scene_token: str
    sample_token: str | None
    sample_data_token: str
    timestamp_us: int
    frame_index: int = Field(ge=0, description="Zero-based position within its scene.")
    is_keyframe: bool
    image_path: str = Field(description="Path relative to the nuScenes data root.")
    prev_token: str | None
    next_token: str | None
    calibrated_sensor_token: str
    ego_pose_token: str


class ProjectedGroundTruthRecord(BaseModel):
    """One row per keyframe annotation projected into CAM_FRONT (Phase 2).

    Privileged offline evaluation evidence only -- never an input to online
    tracking. See METHODOLOGY.md's camera-only boundary.
    """

    schema_version: int = 1
    scene_token: str
    sample_data_token: str
    instance_token: str
    annotation_token: str
    original_category: str
    evaluation_class: str
    visibility_token: str
    x1: float
    y1: float
    x2: float
    y2: float
    center_depth_m: float
    num_lidar_pts: int = Field(ge=0)
    num_radar_pts: int = Field(ge=0)
    truncation_fraction: float = Field(ge=0.0, le=1.0)
    projection_status: str = Field(description='e.g. "accepted", "behind_camera", "outside_image".')


class DetectorObservationRecord(BaseModel):
    """One row per raw detector box (Phase 4).

    The raw box is never replaced by a tracker-smoothed box anywhere
    downstream -- see reuse_audit.md in Assignment 4 for why that distinction
    matters.
    """

    schema_version: int = 1
    scene_token: str
    sample_data_token: str
    frame_index: int = Field(ge=0)
    detection_id: int = Field(ge=0, description="Frame-local index among this frame's boxes.")
    model_name: str
    model_weights_hash: str
    detected_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    x1: float
    y1: float
    x2: float
    y2: float
    inference_time_ms: float = Field(ge=0.0)
    cache_key: str


class OcclusionEventRecord(BaseModel):
    """One row per candidate or reviewed occlusion event (Phase 3 / 5)."""

    schema_version: int = 1
    event_id: str
    scene_token: str
    instance_token: str
    pre_frame_index: int = Field(ge=0)
    start_frame_index: int = Field(ge=0)
    end_frame_index: int = Field(ge=0)
    post_frame_index: int = Field(ge=0)
    event_source: EventSource
    visibility_pattern: str
    possible_occluder_instance_token: str | None = None
    review_status: str = Field(description='e.g. "accepted", "rejected", "unsure".')
    rejection_reason: str | None = None
    split: str = Field(description='Scene-derived split: "development", "validation", or "test".')


class TrackerOutputRecord(BaseModel):
    """One row per track per frame, for any method (baseline or OATM) (Phase 6+).

    A current detection and a memory-only prediction must never share an
    ambiguous evidence label -- `evidence_source` is always exactly one of
    the allowed values, never blank when `state` implies real evidence.
    """

    schema_version: int = 1
    scene_token: str
    sample_data_token: str
    frame_index: int = Field(ge=0)
    method_name: str
    run_id: str
    track_id: int
    state: TrackState
    evidence_source: EvidenceSource | None
    x1: float
    y1: float
    x2: float
    y2: float
    # The RAW detection box actually matched this frame (before any Kalman
    # correction), or None when evidence_source is motion_prediction. Any
    # code needing "the real YOLO box" must use these, never x1..y2 above --
    # x1..y2 is the tracker's smoothed state. This exact confusion was
    # Assignment 4's most serious bug (see its reuse_audit.md, repair #1);
    # this field exists so it can never quietly happen again here.
    raw_detection_x1: float | None = None
    raw_detection_y1: float | None = None
    raw_detection_x2: float | None = None
    raw_detection_y2: float | None = None
    detector_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    existence_confidence: float = Field(ge=0.0, le=1.0)
    identity_confidence: float = Field(ge=0.0, le=1.0)
    localization_uncertainty: float = Field(ge=0.0)
    memory_age_frames: int = Field(ge=0)
    memory_age_seconds: float = Field(ge=0.0)
    termination_reason: str | None = None

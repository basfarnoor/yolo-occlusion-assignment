"""Typed public output contracts for relational tracking."""

from __future__ import annotations

from oatm.records import TrackerOutputRecord
from pydantic import Field


class RelationalTrackerOutput(TrackerOutputRecord):
    schema_version: int = 2
    occluder_track_id: int | None = None
    occlusion_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    expected_clearance_frames: int | None = Field(default=None, ge=0)
    relation_phase: str | None = None
    camera_motion_quality: float = Field(default=0.0, ge=0.0, le=1.0)

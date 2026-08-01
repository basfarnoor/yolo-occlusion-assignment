"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Shared per-frame track output record for both trackers in this experiment
(sort_tracker.py's single-stage baseline and bytetrack_tracker.py's two-stage
BYTE tracker). Every output row states exactly one evidence source, so a
prediction is never mistaken for a real detection downstream (required repair
#1, and the paper-map distinction between visual evidence and motion memory).
"""
from __future__ import annotations

from dataclasses import dataclass

EVIDENCE_HIGH_SCORE = "high_score_detection"
EVIDENCE_LOW_SCORE = "low_score_detection"
EVIDENCE_PREDICTION = "motion_prediction"


@dataclass
class TrackOutput:
    track_id: int
    box: tuple[float, float, float, float]
    class_name: str
    evidence_source: str  # one of EVIDENCE_HIGH_SCORE / EVIDENCE_LOW_SCORE / EVIDENCE_PREDICTION
    hits: int
    hit_streak: int
    time_since_update: int
    age: int
    confirmed: bool
    velocity: tuple[float, float]
    confidence: float
    # The RAW detection box actually matched this frame (before Kalman
    # correction), or None when evidence_source is EVIDENCE_PREDICTION. This
    # is what any downstream code needing "the real YOLO box" must use --
    # `box` above is the tracker's smoothed state and must never be treated
    # as a raw detection (this exact confusion was Assignment 3's most
    # serious flaw; see reuse_audit.md, required repair #1).
    raw_detection_box: tuple[float, float, float, float] | None = None

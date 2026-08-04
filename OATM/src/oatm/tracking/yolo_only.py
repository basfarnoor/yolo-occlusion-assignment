"""Method A: YOLO-only. No tracker, no temporal memory at all -- a detection
baseline, not a tracker. `track_id` carries no real cross-frame identity by
design (a fresh, meaningless index every frame)."""
from __future__ import annotations

from oatm.records import TrackerOutputRecord


def run_yolo_only_frame(
    detections: list[dict], high_score_threshold: float,
    scene_token: str, sample_data_token: str, frame_index: int, method_name: str, run_id: str,
) -> list[TrackerOutputRecord]:
    outputs = []
    for i, d in enumerate(detections):
        if d.get("confidence", 0.0) < high_score_threshold:
            continue
        outputs.append(TrackerOutputRecord(
            scene_token=scene_token, sample_data_token=sample_data_token, frame_index=frame_index,
            method_name=method_name, run_id=run_id,
            track_id=i,  # no real identity -- see module docstring
            state="OBSERVED_STRONG", evidence_source="strong_detection",
            x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
            detector_confidence=d["confidence"],
            existence_confidence=1.0, identity_confidence=1.0,  # not modeled by this baseline
            localization_uncertainty=0.0,  # a current detection, not a prediction
            memory_age_frames=0, memory_age_seconds=0.0, termination_reason=None,
        ))
    return outputs

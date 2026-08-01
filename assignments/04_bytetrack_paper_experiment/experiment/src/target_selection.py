"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 10: discovers "natural tracks" (which raw per-frame YOLO detections
belong to the same physical object across a whole clip) by running the real
ByteTrackTracker over each clip's full, un-gapped detection stream, then
deterministically selects eligible target tracks for the controlled
confidence-demotion and complete-absence experiments.

This linking pass exists only to identify *which raw detections belong to the
same object* -- the resulting boxes used as pseudo-ground-truth in the
controlled trials are the ORIGINAL RAW YOLO detections themselves (kept
verbatim from detections.csv), never the tracker's Kalman-corrected output.
That distinction is what repairs Assignment 3's most serious flaw (see
reuse_audit.md, required repair #1).
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from run_methods import new_bytetrack_tracker
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
EDGE_MARGIN_PX = 25
ALLOWED_CLASSES = ("car", "truck", "bus", "person", "bicycle", "motorcycle")


@dataclass
class NaturalTarget:
    clip: str
    track_id: int
    class_name: str
    frame_numbers: list[int] = field(default_factory=list)
    raw_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    raw_confidences: list[float] = field(default_factory=list)


def build_natural_targets(detections_by_frame: dict, clip_frame_numbers: dict, cfg: dict
                           ) -> list[NaturalTarget]:
    """Runs the real ByteTrackTracker (frozen config) over each clip's full
    detection stream. Records a target's raw box/confidence for a frame ONLY
    when that frame's output evidence_source is an actual detection (high or
    low score) -- motion-only frames are excluded, so raw_boxes are always
    genuine YOLO output, never Kalman-only extrapolation."""
    targets_by_id: dict[tuple[str, int], NaturalTarget] = {}

    for clip, frame_numbers in clip_frame_numbers.items():
        tracker = new_bytetrack_tracker(cfg)
        for frame_no in frame_numbers:
            dets = detections_by_frame.get((clip, frame_no), [])
            outputs = tracker.update(dets, timestamp=float(frame_no))
            for o in outputs:
                if o.evidence_source not in (EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE):
                    continue
                if o.raw_detection_box is None:
                    continue  # defensive; should never happen when evidence_source is a real detection
                key = (clip, o.track_id)
                if key not in targets_by_id:
                    targets_by_id[key] = NaturalTarget(clip=clip, track_id=o.track_id, class_name=o.class_name)
                nt = targets_by_id[key]
                nt.frame_numbers.append(frame_no)
                nt.raw_boxes.append(o.raw_detection_box)  # the REAL YOLO box, never the Kalman-smoothed o.box
                nt.raw_confidences.append(o.confidence)
    return list(targets_by_id.values())


def _touches_edge(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 < EDGE_MARGIN_PX or x2 > IMAGE_WIDTH - EDGE_MARGIN_PX


def select_eligible_targets(targets: list[NaturalTarget], min_track_length: int, min_confidence: float,
                             max_targets: int, seed: int) -> tuple[list[NaturalTarget], list[str]]:
    """Deterministic, logged selection -- same discipline as Assignment 3's
    track_selection.py (reuse_audit.md): every rule applied and every
    relaxation recorded, no manual picking."""
    log: list[str] = []

    def eligible(t: NaturalTarget) -> tuple[bool, str]:
        if t.class_name not in ALLOWED_CLASSES:
            return False, f"class '{t.class_name}' not in allowed set {ALLOWED_CLASSES}"
        if len(t.frame_numbers) < min_track_length:
            return False, f"only {len(t.frame_numbers)} frames, needs >= {min_track_length}"
        if _touches_edge(t.raw_boxes[0]) or _touches_edge(t.raw_boxes[-1]):
            return False, "begins or ends at the image boundary"
        avg_conf = sum(t.raw_confidences) / len(t.raw_confidences)
        if avg_conf < min_confidence:
            return False, f"average raw confidence {avg_conf:.2f} below {min_confidence}"
        return True, "ok"

    selected, reasons = [], []
    for t in sorted(targets, key=lambda t: (t.clip, t.track_id)):
        ok, reason = eligible(t)
        reasons.append((t.clip, t.track_id, ok, reason))
        if ok:
            selected.append(t)

    log.append(f"Eligibility pass: {len(selected)} of {len(targets)} natural targets eligible "
               f"(min_track_length={min_track_length}, min_confidence={min_confidence}).")
    for clip, tid, ok, reason in reasons:
        log.append(f"  - {clip} track {tid}: {'ELIGIBLE' if ok else 'rejected'} ({reason})")

    if len(selected) > max_targets:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_targets)
        log.append(f"More than {max_targets} eligible targets -- deterministically sampled {max_targets} "
                    f"using random seed {seed}.")

    log.append(f"Final selection: {len(selected)} target(s): "
               + ", ".join(f"{t.clip}#{t.track_id}" for t in selected))
    return selected, log

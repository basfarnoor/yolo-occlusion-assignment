"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 6: automatically builds "natural" tracks from the real (non-gapped)
YOLO detections using the same SortTracker from Task 4, then deterministically
selects eligible track segments for the artificial-occlusion experiment.
No manual annotation -- eligibility is decided entirely by the rules below.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field

from sort_tracker import SortTracker

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
EDGE_MARGIN_PX = 25


@dataclass
class NaturalTrack:
    clip: str
    track_id: int
    class_name: str
    frame_numbers: list[int] = field(default_factory=list)
    boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


def build_natural_tracks(detections_by_frame: dict[tuple[str, int], list[dict]],
                          clip_frame_numbers: dict[str, list[int]]) -> list[NaturalTrack]:
    """Run the real SORT tracker over each clip's full, un-gapped detection
    stream to discover which detections belong to the same physical object
    over time. This is the "linking" step that turns independent per-frame
    detections into candidate track segments."""
    tracks_by_id: dict[tuple[str, int], NaturalTrack] = {}

    for clip, frame_numbers in clip_frame_numbers.items():
        tracker = SortTracker(max_age=1, min_hits=1, iou_threshold=0.3)
        for frame_no in frame_numbers:
            dets = detections_by_frame.get((clip, frame_no), [])
            outputs = tracker.update(dets)
            for o in outputs:
                if not o.matched_this_frame:
                    continue  # only record frames with a real detection, not a bridged gap
                key = (clip, o.track_id)
                if key not in tracks_by_id:
                    tracks_by_id[key] = NaturalTrack(clip=clip, track_id=o.track_id, class_name=o.class_name)
                nt = tracks_by_id[key]
                nt.frame_numbers.append(frame_no)
                nt.boxes.append(o.box)
                nt.confidences.append(o.confidence)
    return list(tracks_by_id.values())


def _displacement(track: NaturalTrack) -> float:
    (x1a, y1a, x2a, y2a) = track.boxes[0]
    (x1b, y1b, x2b, y2b) = track.boxes[-1]
    ax, ay = (x1a + x2a) / 2, (y1a + y2a) / 2
    bx, by = (x1b + x2b) / 2, (y1b + y2b) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _touches_edge(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 < EDGE_MARGIN_PX or x2 > IMAGE_WIDTH - EDGE_MARGIN_PX


def select_eligible_tracks(
    tracks: list[NaturalTrack],
    min_track_length: int,
    min_track_length_floor: int,
    max_eligible_tracks: int,
    min_confidence: float,
    min_displacement_px: float,
    allowed_classes: tuple[str, ...],
    seed: int,
) -> tuple[list[NaturalTrack], list[str]]:
    """Returns (selected_tracks, log_lines) -- log_lines records every rule
    applied and every relaxation, per the assignment's "no silent guessing" spirit."""
    log: list[str] = []
    current_min_length = min_track_length

    def eligible(track: NaturalTrack, min_length: int) -> tuple[bool, str]:
        if track.class_name not in allowed_classes:
            return False, f"class '{track.class_name}' not in allowed set {allowed_classes}"
        if len(track.frame_numbers) < min_length:
            return False, f"only {len(track.frame_numbers)} frames, needs >= {min_length}"
        if _touches_edge(track.boxes[0]) or _touches_edge(track.boxes[-1]):
            return False, "begins or ends at the image boundary"
        avg_conf = sum(track.confidences) / len(track.confidences) if track.confidences else 1.0
        if avg_conf < min_confidence:
            return False, f"average confidence {avg_conf:.2f} below {min_confidence}"
        disp = _displacement(track)
        if disp < min_displacement_px:
            return False, f"displacement {disp:.1f}px below {min_displacement_px}px minimum"
        return True, "ok"

    selected: list[NaturalTrack] = []
    while True:
        selected = []
        reasons = []
        for t in sorted(tracks, key=lambda t: (t.clip, t.track_id)):
            ok, reason = eligible(t, current_min_length)
            reasons.append((t.clip, t.track_id, ok, reason))
            if ok:
                selected.append(t)

        log.append(f"Eligibility pass at min_track_length={current_min_length}: "
                    f"{len(selected)} of {len(tracks)} natural tracks eligible.")
        for clip, tid, ok, reason in reasons:
            status = "ELIGIBLE" if ok else "rejected"
            log.append(f"  - {clip} track {tid}: {status} ({reason})")

        if len(selected) >= 3 or current_min_length <= min_track_length_floor:
            break
        current_min_length -= 1
        log.append(f"Fewer than 3 eligible tracks -- relaxing min_track_length to {current_min_length}.")

    if len(selected) > max_eligible_tracks:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_eligible_tracks)
        log.append(f"More than {max_eligible_tracks} eligible tracks -- deterministically sampled "
                    f"{max_eligible_tracks} using random seed {seed}.")

    log.append(f"Final selection: {len(selected)} track segment(s): "
               + ", ".join(f"{t.clip}#{t.track_id}" for t in selected))
    return selected, log

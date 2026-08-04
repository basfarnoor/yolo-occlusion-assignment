"""Phase 5 (Task 7): two explicitly separate controlled-occlusion event
families, built on top of real target tracks discovered from the cached
detections.

- `detector_intervention`: demotes or removes the target's own detection row
  for a window of frames. Image pixels are never touched. Isolates tracker
  behavior only -- must never be called visual occlusion.
- `controlled_visual`: paints a seeded rectangular mask over the target's box
  on a LOCAL COPY of the frame image, then reruns the same frozen detector on
  that copy. The original nuScenes file is never opened for writing.

Every altered frame records enough (source path, target, seed, mask
parameters, coverage, duration, cache key) to be recreated exactly.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

DETECTOR_INTERVENTION = "detector_intervention"
CONTROLLED_VISUAL = "controlled_visual"

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
EDGE_MARGIN_PX = 25
ALLOWED_CLASSES = ("car", "person")


@dataclass
class NaturalTarget:
    scene_token: str
    track_id: int
    class_name: str
    frame_numbers: list[int] = field(default_factory=list)
    sample_data_tokens: list[str] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    raw_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    raw_confidences: list[float] = field(default_factory=list)


def _touches_edge(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return x1 < EDGE_MARGIN_PX or x2 > IMAGE_WIDTH - EDGE_MARGIN_PX


def select_eligible_targets(
    targets: list[NaturalTarget], min_track_length: int, min_confidence: float,
    max_targets: int, seed: int,
) -> tuple[list[NaturalTarget], list[str]]:
    """Deterministic, logged selection -- same discipline as Assignment 4's
    track_selection.py: every rule applied, every relaxation/cap recorded."""
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
    for t in sorted(targets, key=lambda t: (t.scene_token, t.track_id)):
        ok, reason = eligible(t)
        reasons.append((t.scene_token, t.track_id, ok, reason))
        if ok:
            selected.append(t)

    log.append(f"Eligibility pass: {len(selected)} of {len(targets)} natural targets eligible "
               f"(min_track_length={min_track_length}, min_confidence={min_confidence}).")
    for scene_token, tid, ok, reason in reasons:
        log.append(f"  - {scene_token[:8]} track {tid}: {'ELIGIBLE' if ok else 'rejected'} ({reason})")

    if len(selected) > max_targets:
        rng = random.Random(seed)
        selected = rng.sample(selected, max_targets)
        log.append(f"More than {max_targets} eligible targets -- deterministically sampled {max_targets} "
                    f"using random seed {seed}.")

    log.append(f"Final selection: {len(selected)} target(s): "
               + ", ".join(f"{t.scene_token[:8]}#{t.track_id}" for t in selected))
    return selected, log


def apply_seeded_mask(
    image: Image.Image, box: tuple[float, float, float, float], coverage_fraction: float, seed: int,
) -> tuple[Image.Image, tuple[float, float, float, float]]:
    """Paints a seeded rectangular mask covering `coverage_fraction` of the
    target box's area, centered on the box. Mutates and returns the SAME
    image object passed in -- callers must pass a copy, never the original
    loaded frame, and must never write back over an original nuScenes file.
    Returns (image, mask_box) so the exact mask region is recorded too."""
    rng = np.random.default_rng(seed)
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    scale = coverage_fraction ** 0.5
    mw, mh = w * scale, h * scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    mask_box = (cx - mw / 2, cy - mh / 2, cx + mw / 2, cy + mh / 2)

    gray = int(rng.integers(60, 140))
    jitter = int(rng.integers(-15, 15))
    fill_color = tuple(max(0, min(255, gray + jitter * k)) for k in (1, 0, -1))  # a slightly non-neutral gray

    draw = ImageDraw.Draw(image)
    draw.rectangle(mask_box, fill=fill_color)
    return image, mask_box


def build_controlled_windows(
    target: NaturalTarget, durations: list[int], coverages: list[float],
) -> list[dict]:
    """Deterministically places one centered window per (duration, coverage)
    combination in the middle of the target's natural frame span."""
    n = len(target.frame_numbers)
    mid_idx = n // 2
    windows = []
    for duration in durations:
        start_idx = mid_idx - duration // 2
        end_idx = start_idx + duration
        if start_idx < 2 or end_idx > n - 2:
            continue  # need lead-in/lookback frames on both sides
        for coverage in coverages:
            windows.append({
                "duration": duration, "coverage": coverage,
                "window_indices": list(range(start_idx, end_idx)),
            })
    return windows

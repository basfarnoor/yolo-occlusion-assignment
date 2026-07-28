"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Three methods evaluated on identical input: the same per-frame boxes and
the same artificially withheld frames.

  Baseline A -- YOLO only: no box during the gap (measures detector coverage).
  Baseline B -- Static last-seen memory (Assignment 2): freeze the last real
                box, unmoved, through the gap.
  Method   C -- SORT motion memory (Task 4): a KalmanBoxTracker fed the
                pre-gap boxes, then predicted forward through the gap with
                no update() calls (a real missing detection).

All three see the same track history and the same gap -- this module is the
single place that enforces that, so no baseline can accidentally get an
easier or harder input than the others.
"""
from __future__ import annotations

from dataclasses import dataclass

from kalman_box_tracker import KalmanBoxTracker

Box = tuple[float, float, float, float]


@dataclass
class GapFrameResult:
    frame_offset: int  # 0-indexed position within the gap
    yolo_only_box: Box | None
    static_memory_box: Box
    sort_box: Box
    sort_velocity: tuple[float, float]


def run_three_baselines(
    track_boxes: list[Box],
    class_name: str,
    gap_start_idx: int,
    gap_len: int,
    force_zero_velocity: bool = False,
) -> list[GapFrameResult]:
    """track_boxes: the real (pseudo-ground-truth) box for every frame this
    track was actually visible, in time order. gap_start_idx: index of the
    first frame to treat as artificially missing. gap_len: how many
    consecutive frames to withhold (must leave at least one real frame
    before and after the gap).
    """
    if gap_start_idx <= 0 or gap_start_idx + gap_len >= len(track_boxes):
        raise ValueError("gap must have at least one real frame before and after it")

    KalmanBoxTracker.reset_id_counter()
    tracker = KalmanBoxTracker(track_boxes[0], class_name)

    # Feed every real detection strictly before the gap so the filter can
    # learn the track's motion, exactly as SortTracker would during normal operation.
    for i in range(1, gap_start_idx):
        tracker.predict()
        tracker.update(track_boxes[i])

    if force_zero_velocity:
        tracker.x[4:] = 0.0  # ablation: strip the learned velocity, keep everything else

    last_real_box = track_boxes[gap_start_idx - 1]

    results = []
    for offset in range(gap_len):
        predicted_box = tracker.predict()  # no update() -- this frame's detection is withheld
        results.append(GapFrameResult(
            frame_offset=offset,
            yolo_only_box=None,
            static_memory_box=last_real_box,
            sort_box=predicted_box,
            sort_velocity=tracker.velocity(),
        ))
    return results

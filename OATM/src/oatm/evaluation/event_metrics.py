"""Task 11: per-event metrics, computed identically for natural, controlled-
visual, and detector-intervention events (the event window and reference
frames are the only thing that differs between families -- the metric
definitions themselves must stay the same so methods and families are
comparable). The experimental unit here is one event for one method, not a
repeated frame row (see STUDENT_IMPLEMENTATION_ASSIGNMENT.md, Task 11).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from oatm.evaluation.linking import find_track_id_for_reference_box
from oatm.tracking.geometry import iou

RECOVERY_IOU_THRESHOLD = 0.3


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_error(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay = _center(a)
    bx, by = _center(b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass
class EventMetricsInputs:
    event_id: str
    method_name: str
    reference_box: tuple[float, float, float, float]
    reference_class: str
    pre_frame_index: int
    hidden_frame_indices: list[int]
    recovery_search_frame_indices: list[int]
    outputs_by_frame: dict[int, list[dict]]
    gt_box_by_frame: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)


@dataclass
class EventMetricsResult:
    event_id: str
    method_name: str
    target_linked: bool
    n_hidden_frames: int
    n_hidden_frames_alive: int
    hidden_frame_coverage: float | None
    fully_bridged: bool | None
    n_hidden_frames_with_gt: int
    mean_center_error_px: float | None
    mean_iou: float | None
    recovery_status: str  # "same_id", "new_id", "not_recovered", "n/a"
    recovery_latency_frames: int | None


def compute_event_metrics(inputs: EventMetricsInputs) -> EventMetricsResult:
    pre_outputs = inputs.outputs_by_frame.get(inputs.pre_frame_index, [])
    target_track_id = find_track_id_for_reference_box(
        pre_outputs, inputs.reference_box, inputs.reference_class
    )
    if target_track_id is None:
        return EventMetricsResult(
            event_id=inputs.event_id, method_name=inputs.method_name, target_linked=False,
            n_hidden_frames=len(inputs.hidden_frame_indices), n_hidden_frames_alive=0,
            hidden_frame_coverage=None, fully_bridged=None, n_hidden_frames_with_gt=0,
            mean_center_error_px=None, mean_iou=None,
            recovery_status="n/a", recovery_latency_frames=None,
        )

    n_alive = 0
    center_errors: list[float] = []
    ious: list[float] = []
    for frame_idx in inputs.hidden_frame_indices:
        rows = inputs.outputs_by_frame.get(frame_idx, [])
        match = next((r for r in rows if r["track_id"] == target_track_id), None)
        if match is None:
            continue
        n_alive += 1
        gt_box = inputs.gt_box_by_frame.get(frame_idx)
        if gt_box is None:
            continue
        reported_box = (match["x1"], match["y1"], match["x2"], match["y2"])
        center_errors.append(_center_error(reported_box, gt_box))
        ious.append(iou(reported_box, gt_box))

    n_hidden = len(inputs.hidden_frame_indices)
    hidden_frame_coverage = n_alive / n_hidden if n_hidden else None

    recovery_status = "not_recovered"
    recovery_latency_frames: int | None = None
    for latency, frame_idx in enumerate(inputs.recovery_search_frame_indices):
        gt_box = inputs.gt_box_by_frame.get(frame_idx)
        if gt_box is None:
            continue
        rows = inputs.outputs_by_frame.get(frame_idx, [])
        recovered_row = next((
            r for r in rows
            if r["class_name"] == inputs.reference_class
            and r["state"] in ("OBSERVED_STRONG", "OBSERVED_WEAK")
            and iou((r["x1"], r["y1"], r["x2"], r["y2"]), gt_box) >= RECOVERY_IOU_THRESHOLD
        ), None)
        if recovered_row is not None:
            recovery_status = "same_id" if recovered_row["track_id"] == target_track_id else "new_id"
            recovery_latency_frames = latency
            break

    return EventMetricsResult(
        event_id=inputs.event_id, method_name=inputs.method_name, target_linked=True,
        n_hidden_frames=n_hidden, n_hidden_frames_alive=n_alive,
        hidden_frame_coverage=hidden_frame_coverage,
        fully_bridged=(n_alive == n_hidden) if n_hidden else None,
        n_hidden_frames_with_gt=len(center_errors),
        mean_center_error_px=(sum(center_errors) / len(center_errors)) if center_errors else None,
        mean_iou=(sum(ious) / len(ious)) if ious else None,
        recovery_status=recovery_status, recovery_latency_frames=recovery_latency_frames,
    )


def compute_yolo_only_event_metrics(inputs: EventMetricsInputs) -> EventMetricsResult:
    """yolo_only's `track_id` is a fresh, meaningless per-frame index (see
    reuse_audit.md and Task 6's own baseline_summary.md) -- reusing
    `compute_event_metrics`'s track_id-equality logic on it would let
    coincidentally-equal index numbers from UNRELATED detections in
    different frames register as false "coverage" and false "same_id
    recovery". This variant never looks at track_id at all: every frame is
    judged solely by whether ANY same-class detection box overlaps the real
    ground-truth location at that frame (raw single-frame detection
    recurrence, the same notion Assignment 1's YOLO-occlusion experiment
    already measured, reimplemented correctly here). `recovery_status` is
    therefore only ever "detected" or "not_recovered" -- "same_id"/"new_id"
    would imply an identity concept this method does not have."""
    pre_rows = inputs.outputs_by_frame.get(inputs.pre_frame_index, [])
    reference_detected = any(
        r["class_name"] == inputs.reference_class
        and iou((r["x1"], r["y1"], r["x2"], r["y2"]), inputs.reference_box) >= RECOVERY_IOU_THRESHOLD
        for r in pre_rows
    )
    if not reference_detected:
        return EventMetricsResult(
            event_id=inputs.event_id, method_name=inputs.method_name, target_linked=False,
            n_hidden_frames=len(inputs.hidden_frame_indices), n_hidden_frames_alive=0,
            hidden_frame_coverage=None, fully_bridged=None, n_hidden_frames_with_gt=0,
            mean_center_error_px=None, mean_iou=None,
            recovery_status="n/a", recovery_latency_frames=None,
        )

    n_alive = 0
    center_errors: list[float] = []
    ious: list[float] = []
    for frame_idx in inputs.hidden_frame_indices:
        gt_box = inputs.gt_box_by_frame.get(frame_idx)
        rows = inputs.outputs_by_frame.get(frame_idx, [])
        if gt_box is not None:
            match = next((
                r for r in rows
                if r["class_name"] == inputs.reference_class
                and iou((r["x1"], r["y1"], r["x2"], r["y2"]), gt_box) >= RECOVERY_IOU_THRESHOLD
            ), None)
        else:
            match = None
        if match is None:
            continue
        n_alive += 1
        reported_box = (match["x1"], match["y1"], match["x2"], match["y2"])
        center_errors.append(_center_error(reported_box, gt_box))
        ious.append(iou(reported_box, gt_box))

    n_hidden = len(inputs.hidden_frame_indices)
    hidden_frame_coverage = n_alive / n_hidden if n_hidden else None

    recovery_status = "not_recovered"
    recovery_latency_frames: int | None = None
    for latency, frame_idx in enumerate(inputs.recovery_search_frame_indices):
        gt_box = inputs.gt_box_by_frame.get(frame_idx)
        if gt_box is None:
            continue
        rows = inputs.outputs_by_frame.get(frame_idx, [])
        recovered = any(
            r["class_name"] == inputs.reference_class
            and iou((r["x1"], r["y1"], r["x2"], r["y2"]), gt_box) >= RECOVERY_IOU_THRESHOLD
            for r in rows
        )
        if recovered:
            recovery_status = "detected"
            recovery_latency_frames = latency
            break

    return EventMetricsResult(
        event_id=inputs.event_id, method_name=inputs.method_name, target_linked=True,
        n_hidden_frames=n_hidden, n_hidden_frames_alive=n_alive,
        hidden_frame_coverage=hidden_frame_coverage,
        fully_bridged=(n_alive == n_hidden) if n_hidden else None,
        n_hidden_frames_with_gt=len(center_errors),
        mean_center_error_px=(sum(center_errors) / len(center_errors)) if center_errors else None,
        mean_iou=(sum(ious) / len(ious)) if ious else None,
        recovery_status=recovery_status, recovery_latency_frames=recovery_latency_frames,
    )

"""Task 11: each tracking method assigns its OWN track_id numbers, so the
"same real object" has a different ID under YOLO-only, SORT, ByteTrack, and
OATM. Before any cross-method occlusion metric can be computed, we need one
common anchor: which of THIS method's track_ids, at a known reference frame,
corresponds to the real object we care about. Always resolved from a
reference box at a reference frame that is NOT inside the hidden window --
never from a future frame inside the window itself.
"""
from __future__ import annotations

from oatm.evaluation.ground_truth import DETECTOR_TO_EVAL_CLASS
from oatm.tracking.geometry import iou


def find_track_id_for_reference_box(
    frame_outputs: list[dict],
    reference_box: tuple[float, float, float, float],
    reference_class: str,
    iou_threshold: float = 0.3,
) -> int | None:
    """Returns the track_id (from one method's outputs at one frame) whose
    box best overlaps `reference_box`, restricted to the same class and to
    an IoU at or above `iou_threshold`. None if no candidate qualifies."""
    best_track_id, best_iou = None, iou_threshold
    for row in frame_outputs:
        if row["class_name"] != reference_class:
            continue
        candidate_box = (row["x1"], row["y1"], row["x2"], row["y2"])
        score = iou(reference_box, candidate_box)
        if score >= best_iou:
            best_iou = score
            best_track_id = row["track_id"]
    return best_track_id


def resolve_instance_token(
    frame_gt: list[dict],
    reference_box: tuple[float, float, float, float],
    detector_class: str,
    iou_threshold: float = 0.3,
) -> str | None:
    """For controlled-experiment targets (identified only by a ByteTrack-
    derived box, not a nuScenes instance_token), finds the real ground-truth
    object at the same frame that this box actually corresponds to, so
    center-error/IoU-while-hidden can be measured against real 3D-projected
    truth rather than another tracker's own output."""
    eval_class = DETECTOR_TO_EVAL_CLASS.get(detector_class)
    if eval_class is None:
        return None
    best_instance, best_iou = None, iou_threshold
    for row in frame_gt:
        if row["evaluation_class"] != eval_class:
            continue
        score = iou(reference_box, row["box"])
        if score >= best_iou:
            best_iou = score
            best_instance = row["instance_token"]
    return best_instance

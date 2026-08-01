"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Decides which predicted track goes with which new detection. The paper
builds an IoU cost matrix between predicted boxes and detections, then
solves it with the Hungarian algorithm for an optimal one-to-one matching
(rather than greedily grabbing the first good-looking match).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from geometry import iou

# Cost for a pair that must never be matched (wrong class, or below IoU threshold).
FORBIDDEN_COST = 1e6


def associate_detections_to_trackers(
    detections: list[dict],
    tracker_boxes: list[tuple[float, float, float, float]],
    tracker_classes: list[str],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections (list of {"class","x1","y1","x2","y2",...}) to predicted
    tracker boxes. Only same-class pairs above iou_threshold may be matched.

    Returns (matches, unmatched_detection_indices, unmatched_tracker_indices),
    where matches is a list of (detection_index, tracker_index) pairs.
    """
    n_det, n_trk = len(detections), len(tracker_boxes)
    if n_det == 0 or n_trk == 0:
        return [], list(range(n_det)), list(range(n_trk))

    iou_matrix = np.zeros((n_det, n_trk))
    for d in range(n_det):
        det_box = (detections[d]["x1"], detections[d]["y1"], detections[d]["x2"], detections[d]["y2"])
        for t in range(n_trk):
            if detections[d]["class"] != tracker_classes[t]:
                continue
            iou_matrix[d, t] = iou(det_box, tracker_boxes[t])

    cost_matrix = 1.0 - iou_matrix
    cost_matrix[iou_matrix == 0.0] = FORBIDDEN_COST  # incompatible class or zero overlap

    det_idx, trk_idx = linear_sum_assignment(cost_matrix)

    matches = []
    matched_dets, matched_trks = set(), set()
    for d, t in zip(det_idx, trk_idx):
        if iou_matrix[d, t] >= iou_threshold:
            matches.append((int(d), int(t)))
            matched_dets.add(d)
            matched_trks.add(t)

    unmatched_detections = [d for d in range(n_det) if d not in matched_dets]
    unmatched_trackers = [t for t in range(n_trk) if t not in matched_trks]
    return matches, unmatched_detections, unmatched_trackers

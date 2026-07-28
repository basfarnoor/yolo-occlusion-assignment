"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Basic bounding-box geometry: the small building blocks every other module
in this experiment is built from. The paper represents each tracked box's
state as center position, scale (area), and aspect ratio -- this module
converts between that representation and ordinary (x1, y1, x2, y2) boxes.
"""
from __future__ import annotations

import numpy as np


def box_center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_width_height(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x2 - x1, y2 - y1)


def box_area(box: tuple[float, float, float, float]) -> float:
    w, h = box_width_height(box)
    return max(0.0, w) * max(0.0, h)


def iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Intersection over Union -- 1.0 for identical boxes, 0.0 for no overlap."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_error(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    """Pixel distance between the two boxes' centers."""
    ax, ay = box_center(box_a)
    bx, by = box_center(box_b)
    return float(np.hypot(ax - bx, ay - by))


def box_to_state(box: tuple[float, float, float, float]) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to the SORT paper's [cx, cy, s, r] representation:
    center x, center y, scale (area), aspect ratio (width / height)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2.0, y1 + h / 2.0
    s = w * h  # "scale" in the paper is just the box's area
    r = w / h if h > 0 else 1.0  # aspect ratio
    return np.array([cx, cy, s, r], dtype=float)


def state_to_box(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert [cx, cy, s, r] back to (x1, y1, x2, y2)."""
    cx, cy, s, r = state[:4]
    s = max(s, 1e-6)
    w = np.sqrt(s * r)
    h = s / w if w > 0 else 0.0
    x1, y1 = cx - w / 2.0, cy - h / 2.0
    x2, y2 = cx + w / 2.0, cy + h / 2.0
    return (float(x1), float(y1), float(x2), float(y2))

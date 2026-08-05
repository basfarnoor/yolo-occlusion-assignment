"""Causal background-motion estimation for image-space track compensation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraMotionEstimate:
    affine: np.ndarray
    quality: float
    n_matches: int
    n_inliers: int
    used_fallback: bool


def identity_estimate() -> CameraMotionEstimate:
    return CameraMotionEstimate(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0.0, 0, 0, True)


def apply_affine_to_box(box: Box, affine: np.ndarray) -> Box:
    corners = np.array(
        [[[box[0], box[1]]], [[box[2], box[1]]], [[box[2], box[3]]], [[box[0], box[3]]]],
        dtype=np.float32,
    )
    moved = cv2.transform(corners, affine).reshape(-1, 2)
    return (
        float(moved[:, 0].min()),
        float(moved[:, 1].min()),
        float(moved[:, 0].max()),
        float(moved[:, 1].max()),
    )


def foreground_mask(shape: tuple[int, ...], boxes: list[Box]) -> np.ndarray:
    mask = np.full(shape[:2], 255, dtype=np.uint8)
    height, width = shape[:2]
    for x1, y1, x2, y2 in boxes:
        left, top = max(0, int(x1)), max(0, int(y1))
        right, bottom = min(width - 1, int(x2)), min(height - 1, int(y2))
        if right > left and bottom > top:
            cv2.rectangle(mask, (left, top), (right, bottom), 0, thickness=-1)
    return mask


class CameraMotionEstimator:
    def __init__(
        self,
        max_features: int = 1000,
        min_matches: int = 12,
        min_inlier_ratio: float = 0.35,
        ransac_threshold_px: float = 3.0,
        max_translation_px: float = 60.0,
    ) -> None:
        self.orb = cv2.ORB_create(nfeatures=max_features)
        self.min_matches = min_matches
        self.min_inlier_ratio = min_inlier_ratio
        self.ransac_threshold_px = ransac_threshold_px
        self.max_translation_px = max_translation_px

    def estimate(
        self, previous: np.ndarray | None, current: np.ndarray | None, boxes: list[Box]
    ) -> CameraMotionEstimate:
        if previous is None or current is None or previous.shape[:2] != current.shape[:2]:
            return identity_estimate()
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if previous.ndim == 3 else previous
        current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if current.ndim == 3 else current
        mask = foreground_mask(previous_gray.shape, boxes)
        key_a, desc_a = self.orb.detectAndCompute(previous_gray, mask)
        key_b, desc_b = self.orb.detectAndCompute(current_gray, mask)
        if desc_a is None or desc_b is None:
            return identity_estimate()
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(desc_a, desc_b)
        pairs = sorted(pairs, key=lambda pair: pair.distance)
        if len(pairs) < self.min_matches:
            return identity_estimate()
        source = np.float32([key_a[p.queryIdx].pt for p in pairs])
        destination = np.float32([key_b[p.trainIdx].pt for p in pairs])
        affine, inliers = cv2.estimateAffinePartial2D(
            source,
            destination,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold_px,
        )
        if affine is None or inliers is None:
            return identity_estimate()
        n_inliers = int(inliers.sum())
        quality = n_inliers / len(pairs)
        if quality < self.min_inlier_ratio:
            return identity_estimate()
        # Estimate partial affine robustly, but apply only bounded translation:
        # compounding per-frame scale causes exponential long-sequence drift.
        translation = np.array([[1.0, 0.0, affine[0, 2]], [0.0, 1.0, affine[1, 2]]], dtype=float)
        if float(np.hypot(translation[0, 2], translation[1, 2])) > self.max_translation_px:
            return identity_estimate()
        return CameraMotionEstimate(translation, quality, len(pairs), n_inliers, False)

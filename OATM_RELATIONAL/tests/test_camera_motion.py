import cv2
import numpy as np

from oatm_relational.camera_motion import CameraMotionEstimator, apply_affine_to_box


def test_affine_translation_moves_box_exactly():
    affine = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0]])
    assert apply_affine_to_box((10, 20, 30, 40), affine) == (17.0, 17.0, 37.0, 37.0)


def test_blank_frames_use_explicit_identity_fallback():
    estimator = CameraMotionEstimator(min_matches=8)
    blank = np.zeros((120, 160), dtype=np.uint8)
    estimate = estimator.estimate(blank, blank, [])
    assert estimate.used_fallback
    assert estimate.quality == 0.0


def test_implausibly_large_translation_uses_fallback():
    estimator = CameraMotionEstimator(max_translation_px=5.0, min_matches=8)
    rng = np.random.default_rng(7)
    previous = np.zeros((200, 280), dtype=np.uint8)
    for x, y in rng.integers([15, 15], [265, 185], size=(80, 2)):
        cv2.circle(previous, (int(x), int(y)), 2, 255, -1)
    current = cv2.warpAffine(previous, np.float32([[1, 0, 15], [0, 1, 0]]), (280, 200))
    assert estimator.estimate(previous, current, []).used_fallback


def test_feature_translation_is_recovered_causally():
    rng = np.random.default_rng(42)
    previous = np.zeros((240, 320), dtype=np.uint8)
    for x, y in rng.integers([15, 15], [305, 225], size=(100, 2)):
        cv2.circle(previous, (int(x), int(y)), 2, 255, -1)
    transform = np.float32([[1, 0, 6], [0, 1, 4]])
    current = cv2.warpAffine(previous, transform, (320, 240))
    estimate = CameraMotionEstimator(min_matches=8, min_inlier_ratio=0.3).estimate(previous, current, [])
    assert not estimate.used_fallback
    assert abs(estimate.affine[0, 2] - 6) < 1.5
    assert abs(estimate.affine[1, 2] - 4) < 1.5

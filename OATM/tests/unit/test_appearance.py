import numpy as np

from oatm.memory.appearance import AppearanceAnchor, cosine_similarity, is_eligible_for_anchor_update


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert abs(cosine_similarity(a, b)) < 1e-9


def test_anchor_starts_empty():
    anchor = AppearanceAnchor()
    assert anchor.embedding is None
    assert anchor.similarity(np.array([1.0, 0.0])) is None


def test_anchor_updates_only_when_eligible():
    anchor = AppearanceAnchor()
    v1 = np.array([1.0, 0.0])
    anchor.update(v1, eligible=False)
    assert anchor.embedding is None, "an ineligible frame must never write the anchor"
    anchor.update(v1, eligible=True)
    assert anchor.embedding is not None


def test_anchor_never_overwritten_by_ineligible_update_once_set():
    """Regression-style test for the exact rule Task 12 requires: once a
    track has a real clear-view anchor, an occluder's own appearance (fed in
    as an ineligible update, e.g. while the track is PREDICTED_HIDDEN) must
    never be able to overwrite it."""
    anchor = AppearanceAnchor()
    real_view = np.array([1.0, 0.0, 0.0])
    occluder_view = np.array([0.0, 1.0, 0.0])
    anchor.update(real_view, eligible=True)
    anchor.update(occluder_view, eligible=False)
    assert np.array_equal(anchor.embedding, real_view)


def test_anchor_replaces_with_latest_eligible_view():
    anchor = AppearanceAnchor()
    v1, v2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    anchor.update(v1, eligible=True)
    anchor.update(v2, eligible=True)
    assert np.array_equal(anchor.embedding, v2)


def test_eligibility_requires_observed_strong_state():
    box = (100, 100, 140, 140)
    assert not is_eligible_for_anchor_update("PREDICTED_HIDDEN", box, 1600, 900)
    assert not is_eligible_for_anchor_update("OBSERVED_WEAK", box, 1600, 900)
    assert is_eligible_for_anchor_update("OBSERVED_STRONG", box, 1600, 900)


def test_eligibility_rejects_tiny_boxes():
    tiny_box = (100, 100, 105, 105)
    assert not is_eligible_for_anchor_update("OBSERVED_STRONG", tiny_box, 1600, 900)


def test_eligibility_rejects_boundary_truncated_boxes():
    edge_box = (0, 100, 40, 140)
    assert not is_eligible_for_anchor_update("OBSERVED_STRONG", edge_box, 1600, 900)

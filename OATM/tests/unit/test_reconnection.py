import numpy as np

from oatm.tracking.reconnection import HiddenTrackCandidate, resolve_reconnection


def _det(x1, y1, x2, y2, cls, embedding):
    return {"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2, "embedding": np.array(embedding)}


def test_basic_reconnection_by_matching_appearance():
    tracks = [HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0]))]
    dets = [_det(500, 500, 540, 540, "car", [1.0, 0.0, 0.0])]  # far away, but same appearance
    matches = resolve_reconnection(tracks, dets, mode="appearance_only")
    assert matches == [(0, 0)]


def test_different_class_never_matches_even_with_identical_appearance():
    tracks = [HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0]))]
    dets = [_det(2, 2, 42, 42, "person", [1.0, 0.0, 0.0])]
    matches = resolve_reconnection(tracks, dets, mode="appearance_only")
    assert matches == []


def test_below_similarity_threshold_never_matches():
    tracks = [HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0]))]
    dets = [_det(2, 2, 42, 42, "car", [0.0, 1.0, 0.0])]  # orthogonal -- similarity 0.0
    matches = resolve_reconnection(tracks, dets, mode="appearance_only", appearance_similarity_threshold=0.7)
    assert matches == []


def test_dual_mode_rejects_a_match_that_is_spatially_impossible():
    tracks = [HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0]))]
    # Great appearance match, but nowhere near the predicted location.
    dets = [_det(1000, 800, 1040, 840, "car", [1.0, 0.0, 0.0])]
    appearance_only = resolve_reconnection(tracks, dets, mode="appearance_only")
    dual = resolve_reconnection(tracks, dets, mode="dual")
    assert appearance_only == [(0, 0)], "appearance_only must ignore location entirely"
    assert dual == [], "dual mode must still require basic location plausibility"


def test_hard_negative_nearby_same_class_objects_disambiguated_by_appearance():
    """The exact hard-negative case Task 12 requires: two same-class hidden
    tracks, one near the reappearing detection's position and one far away.
    A motion/IoU-only method would wrongly favor the nearby one. Appearance
    must correctly identify the real match even though it is the FAR track,
    because the reappearing detection's true appearance matches it, not the
    visually-different track that merely happens to sit at the old
    position."""
    track_near_but_wrong = HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0]))
    track_far_but_right = HiddenTrackCandidate("car", (500, 500, 540, 540), np.array([0.0, 1.0, 0.0]))
    # Reappears right where the "wrong" track was predicted, but its real
    # appearance matches the "right" track.
    dets = [_det(2, 2, 42, 42, "car", [0.0, 1.0, 0.0])]

    matches = resolve_reconnection(
        [track_near_but_wrong, track_far_but_right], dets, mode="appearance_only",
    )
    assert matches == [(0, 1)], "must match the track whose real appearance matches, not the nearer decoy"


def test_one_to_one_matching_never_double_assigns():
    tracks = [
        HiddenTrackCandidate("car", (0, 0, 40, 40), np.array([1.0, 0.0, 0.0])),
        HiddenTrackCandidate("car", (100, 0, 140, 40), np.array([1.0, 0.0, 0.0])),
    ]
    dets = [_det(2, 2, 42, 42, "car", [1.0, 0.0, 0.0])]  # only one real detection this frame
    matches = resolve_reconnection(tracks, dets, mode="appearance_only")
    assert len(matches) == 1
    matched_track_indices = {t for _, t in matches}
    assert len(matched_track_indices) == 1


def test_no_anchor_never_matches():
    tracks = [HiddenTrackCandidate("car", (0, 0, 40, 40), None)]
    dets = [_det(2, 2, 42, 42, "car", [1.0, 0.0, 0.0])]
    assert resolve_reconnection(tracks, dets, mode="appearance_only") == []


def test_empty_inputs_return_no_matches():
    assert resolve_reconnection([], [], mode="dual") == []

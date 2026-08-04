"""Tests for the Task 12 appearance-reconnection tracker. Uses synthetic,
manually-set embeddings (not the real frozen network) so these stay fast and
self-contained -- the real network is covered separately in
test_embedder.py, and the pure matching/anchor logic in test_reconnection.py
and test_appearance.py. These tests check the WIRING: does the tracker
correctly freeze/update anchors and use them to reconnect, end to end."""
import numpy as np

from oatm.tracking.kalman import KalmanBoxTracker
from oatm.tracking.oatm_appearance_adapter import OATMAppearanceTracker


def _det(x1, y1, x2, y2, conf=0.9, cls="car", embedding=None):
    d = {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    if embedding is not None:
        d["embedding"] = np.array(embedding)
    return d


def test_motion_only_mode_never_reconnects_across_a_large_jump():
    """The ablation's baseline arm: with appearance disabled, a detection far
    from the predicted box must birth a NEW track, exactly like plain
    OATMTracker -- never silently reconnect via appearance."""
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMAppearanceTracker(appearance_mode="motion_only")
    out1 = tracker.update([_det(100, 100, 140, 140, embedding=[1.0, 0.0, 0.0])], timestamp=0.0)
    original_id = out1[0].track_id

    # Same object class, huge appearance similarity, but nowhere near the
    # predicted location, and appearance is disabled -- must NOT reconnect.
    out2 = tracker.update([_det(1000, 800, 1040, 840, embedding=[1.0, 0.0, 0.0])], timestamp=1.0)
    new_ids = {o.track_id for o in out2}
    assert original_id not in new_ids or len(new_ids) > 1
    assert any(tid != original_id for tid in new_ids), "a far-away detection must birth a new track"


def test_appearance_only_mode_reconnects_across_a_large_jump_via_matching_embedding():
    KalmanBoxTracker.reset_id_counter()
    # max_grace_frames_without_evidence widened to 3: with no occluder object
    # in this synthetic scene at all, the state machine's own grace period
    # (Task 9, frozen at 1 by default) would otherwise expire and terminate
    # the track as an ordinary miss one frame before reconnection could ever
    # run -- a real event usually has continuous occluder-overlap evidence
    # keeping PREDICTED_HIDDEN alive instead of relying on the grace period.
    tracker = OATMAppearanceTracker(
        appearance_mode="appearance_only", appearance_similarity_threshold=0.7,
        max_grace_frames_without_evidence=3,
    )
    out1 = tracker.update([_det(100, 100, 140, 140, embedding=[1.0, 0.0, 0.0])], timestamp=0.0)
    original_id = out1[0].track_id

    # Nothing else in the scene, so the track goes PREDICTED_HIDDEN with no
    # occluder evidence needed (grace period covers frame 1).
    tracker.update([], timestamp=1.0)

    # Reappears somewhere completely different but with the SAME appearance.
    out3 = tracker.update([_det(1000, 800, 1040, 840, embedding=[1.0, 0.0, 0.0])], timestamp=2.0)
    assert any(o.track_id == original_id and o.state == "OBSERVED_STRONG" for o in out3), (
        "appearance_only must reconnect the original track_id via matching embedding, "
        "even at a completely different location"
    )


def test_dual_mode_refuses_reconnection_without_location_plausibility():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMAppearanceTracker(
        appearance_mode="dual", appearance_similarity_threshold=0.7, max_grace_frames_without_evidence=3,
    )
    out1 = tracker.update([_det(100, 100, 140, 140, embedding=[1.0, 0.0, 0.0])], timestamp=0.0)
    original_id = out1[0].track_id
    tracker.update([], timestamp=1.0)

    out3 = tracker.update([_det(1000, 800, 1040, 840, embedding=[1.0, 0.0, 0.0])], timestamp=2.0)
    reconnected = any(o.track_id == original_id and o.state == "OBSERVED_STRONG" for o in out3)
    assert not reconnected, "dual mode must reject a spatially-impossible reconnection"


def test_anchor_is_never_updated_while_hidden():
    """The appearance anchor must freeze during occlusion so an occluder's
    own appearance can never overwrite it."""
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMAppearanceTracker(appearance_mode="appearance_only")
    tracker.update([_det(100, 100, 140, 140, embedding=[1.0, 0.0, 0.0])], timestamp=0.0)
    track = tracker.tracks[0]
    anchor_after_first_view = track.appearance_anchor.embedding.copy()

    # A different, unrelated, unclaimed detection overlaps the predicted box
    # -- classic occluder evidence -- but must never touch the anchor even
    # though the tracker sees ITS box/class in this frame's inputs.
    tracker.update(
        [_det(100, 100, 140, 140, cls="truck", embedding=[0.0, 0.0, 1.0])], timestamp=1.0,
    )
    assert np.array_equal(track.appearance_anchor.embedding, anchor_after_first_view), (
        "the anchor must stay frozen while the track is not OBSERVED_STRONG"
    )


def test_reconnection_never_creates_a_duplicate_birth_from_the_same_detection():
    KalmanBoxTracker.reset_id_counter()
    tracker = OATMAppearanceTracker(
        appearance_mode="appearance_only", new_track_threshold=0.6, max_grace_frames_without_evidence=3,
    )
    tracker.update([_det(100, 100, 140, 140, embedding=[1.0, 0.0, 0.0])], timestamp=0.0)
    tracker.update([], timestamp=1.0)

    out3 = tracker.update([_det(1000, 800, 1040, 840, conf=0.9, embedding=[1.0, 0.0, 0.0])], timestamp=2.0)
    assert len(out3) == 1, "a reconnected detection must not ALSO birth a separate new track"

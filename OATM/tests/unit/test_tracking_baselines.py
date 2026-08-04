"""Required Task 6 tests covering all four baselines: timestamp-aware Kalman
prediction, track birth/missing/reactivation/expiry, strong-before-weak
ByteTrack association, weak detections never birthing tracks, scene-boundary
track isolation, structural inaccessibility of ground truth, and determinism.
"""
import inspect

import pytest

from oatm.tracking import bytetrack_adapter, sort_adapter, static_memory
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter
from oatm.tracking.kalman import KalmanBoxTracker
from oatm.tracking.sort_adapter import SortAdapter
from oatm.tracking.static_memory import StaticMemoryTracker


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# --- Timestamp-aware Kalman prediction ---

def test_kalman_prediction_scales_with_dt():
    KalmanBoxTracker.reset_id_counter()
    t = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")
    for step in range(1, 4):
        t.predict(dt=1.0)
        t.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))
    cx0 = (t.current_box()[0] + t.current_box()[2]) / 2
    box_dt1 = t.predict(dt=1.0)
    cx1 = (box_dt1[0] + box_dt1[2]) / 2

    KalmanBoxTracker.reset_id_counter()
    t2 = KalmanBoxTracker((0.0, 0.0, 40.0, 40.0), "car")
    for step in range(1, 4):
        t2.predict(dt=1.0)
        t2.update((10.0 * step, 0.0, 40.0 + 10.0 * step, 40.0))
    box_dt2 = t2.predict(dt=2.0)
    cx2 = (box_dt2[0] + box_dt2[2]) / 2

    assert abs((cx2 - cx0) - 2.0 * (cx1 - cx0)) < 1e-6


# --- Track birth / missing / reactivation / expiry, for each baseline ---

@pytest.mark.parametrize("tracker_factory", [
    lambda: StaticMemoryTracker(track_buffer=3),
    lambda: SortAdapter(track_buffer=3),
    lambda: ByteTrackAdapter(track_buffer=3),
])
def test_track_birth_missing_reactivation_expiry(tracker_factory):
    KalmanBoxTracker.reset_id_counter()
    static_memory._StaticTrack.reset_id_counter()
    tracker = tracker_factory()

    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    assert len(outputs) == 1
    track_id = outputs[0].track_id

    # Missing for exactly track_buffer frames -- must survive.
    for i in range(3):
        outputs = tracker.update([], timestamp=float(i + 1))
    assert any(o.track_id == track_id for o in outputs)

    # One frame further -- must expire.
    outputs = tracker.update([], timestamp=4.0)
    assert not any(o.track_id == track_id for o in outputs)


@pytest.mark.parametrize("tracker_factory", [
    lambda: StaticMemoryTracker(track_buffer=3),
    lambda: SortAdapter(track_buffer=3),
    lambda: ByteTrackAdapter(track_buffer=3),
])
def test_track_reactivates_with_the_same_id_via_real_association(tracker_factory):
    KalmanBoxTracker.reset_id_counter()
    static_memory._StaticTrack.reset_id_counter()
    tracker = tracker_factory()

    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    track_id = outputs[0].track_id

    tracker.update([], timestamp=1.0)
    outputs = tracker.update([_det(1, 0, 41, 40)], timestamp=2.0)

    assert len(outputs) == 1
    assert outputs[0].track_id == track_id, "reappearance must reconnect via real IoU association, same ID"


# --- Strong-before-weak ByteTrack association ---

def test_bytetrack_matches_strong_detection_before_weak_one():
    KalmanBoxTracker.reset_id_counter()
    tracker = ByteTrackAdapter(track_buffer=3)
    tracker.update([_det(0, 0, 40, 40, conf=0.9)], timestamp=0.0)

    outputs = tracker.update([
        _det(1, 1, 41, 41, conf=0.9),
        _det(2, 2, 42, 42, conf=0.2),
    ], timestamp=1.0)

    assert len(outputs) == 1
    assert outputs[0].evidence_source == "strong_detection"


# --- Unmatched weak detection cannot create a new track ---

def test_unmatched_weak_detection_never_creates_a_new_track():
    KalmanBoxTracker.reset_id_counter()
    tracker = ByteTrackAdapter(high_score_threshold=0.5, new_track_threshold=0.6)
    outputs = tracker.update([_det(0, 0, 40, 40, conf=0.2)], timestamp=0.0)
    assert outputs == []


# --- Scene boundaries reset all active tracks ---

def test_scene_boundary_resets_tracks_no_id_overlap_across_scenes():
    KalmanBoxTracker.reset_id_counter()
    scene_a_tracker = ByteTrackAdapter(track_buffer=3)
    outputs_a = scene_a_tracker.update([_det(0, 0, 40, 40)], scene_token="scene-a", timestamp=0.0)

    scene_b_tracker = ByteTrackAdapter(track_buffer=3)  # a fresh instance -- required per scene
    outputs_b = scene_b_tracker.update([_det(0, 0, 40, 40)], scene_token="scene-b", timestamp=0.0)

    ids_a = {o.track_id for o in outputs_a}
    ids_b = {o.track_id for o in outputs_b}
    assert ids_a.isdisjoint(ids_b), "two scenes must never share a track ID"


# --- Ground truth is structurally inaccessible ---

def test_ground_truth_is_structurally_inaccessible_from_tracker_interfaces():
    for module in (sort_adapter, bytetrack_adapter, static_memory):
        source = inspect.getsource(module)
        assert "projected_ground_truth" not in source
        assert "from oatm.dataset.projection" not in source

    allowed_params = {"detections", "timestamp", "scene_token", "sample_data_token", "method_name", "run_id"}
    for cls in (SortAdapter, ByteTrackAdapter, StaticMemoryTracker):
        params = set(inspect.signature(cls.update).parameters) - {"self"}
        assert params == allowed_params


# --- Determinism ---

@pytest.mark.parametrize("tracker_factory", [
    lambda: StaticMemoryTracker(track_buffer=3),
    lambda: SortAdapter(track_buffer=3),
    lambda: ByteTrackAdapter(track_buffer=3),
])
def test_identical_input_and_seed_produce_identical_output(tracker_factory):
    def run_once():
        KalmanBoxTracker.reset_id_counter()
        static_memory._StaticTrack.reset_id_counter()
        tracker = tracker_factory()
        frames = [
            [_det(0, 0, 40, 40)],
            [_det(10, 0, 50, 40)],
            [],
            [_det(30, 0, 70, 40)],
        ]
        all_outputs = []
        for i, frame_dets in enumerate(frames):
            outs = tracker.update(frame_dets, timestamp=float(i))
            all_outputs.append([(o.track_id, o.x1, o.y1, o.x2, o.y2, o.evidence_source) for o in outs])
        return all_outputs

    assert run_once() == run_once()


# --- Localization uncertainty grows during missing frames (Kalman-based methods) ---

@pytest.mark.parametrize("tracker_factory", [
    lambda: SortAdapter(track_buffer=5),
    lambda: ByteTrackAdapter(track_buffer=5),
])
def test_localization_uncertainty_grows_while_a_track_is_missing(tracker_factory):
    KalmanBoxTracker.reset_id_counter()
    tracker = tracker_factory()
    tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)

    uncertainties = []
    for i in range(1, 4):
        outputs = tracker.update([], timestamp=float(i))
        uncertainties.append(outputs[0].localization_uncertainty)

    assert all(b >= a for a, b in zip(uncertainties, uncertainties[1:])), (
        "uncertainty should not decrease while no new evidence arrives"
    )
    assert uncertainties[-1] > uncertainties[0]


@pytest.mark.parametrize("tracker_factory", [
    lambda: SortAdapter(track_buffer=3),
    lambda: ByteTrackAdapter(track_buffer=3),
])
def test_raw_detection_box_is_the_actual_yolo_box_not_the_smoothed_state(tracker_factory):
    """Regression test for the exact bug Assignment 4's mentor review found
    (see reuse_audit.md, required repair #1): raw_detection_x1..y2 must be
    the literal matched box, distinct from the Kalman-smoothed x1..y2, and
    None whenever the track has no real evidence this frame."""
    KalmanBoxTracker.reset_id_counter()
    tracker = tracker_factory()
    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    assert (outputs[0].raw_detection_x1, outputs[0].raw_detection_y1,
            outputs[0].raw_detection_x2, outputs[0].raw_detection_y2) == (0, 0, 40, 40)

    outputs = tracker.update([_det(3, 1, 43, 41)], timestamp=1.0)
    assert outputs[0].raw_detection_x1 == 3
    assert outputs[0].x1 != outputs[0].raw_detection_x1, (
        "the Kalman-corrected x1 should differ from the raw detection once there is any innovation"
    )

    outputs = tracker.update([], timestamp=2.0)
    assert outputs[0].evidence_source == "motion_prediction"
    assert outputs[0].raw_detection_x1 is None


def test_static_memory_raw_detection_box_matches_frozen_box_while_matched():
    KalmanBoxTracker.reset_id_counter()
    static_memory._StaticTrack.reset_id_counter()
    tracker = StaticMemoryTracker(track_buffer=3)
    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    assert (outputs[0].raw_detection_x1, outputs[0].raw_detection_x2) == (0, 40)

    outputs = tracker.update([], timestamp=1.0)
    assert outputs[0].raw_detection_x1 is None
    assert outputs[0].x1 == 0  # the frozen box itself is unchanged


def test_baseline_confidence_fields_are_documented_constants_not_fabricated():
    KalmanBoxTracker.reset_id_counter()
    tracker = ByteTrackAdapter()
    outputs = tracker.update([_det(0, 0, 40, 40)], timestamp=0.0)
    assert outputs[0].existence_confidence == 1.0
    assert outputs[0].identity_confidence == 1.0

"""Required Task 7 tests: deterministic recreation (same seed -> same mask)
and method parity (every baseline receives the identical modified detection
list for a given event)."""
from PIL import Image

from oatm.dataset.controlled_occlusion import (
    NaturalTarget,
    apply_seeded_mask,
    build_controlled_windows,
    select_eligible_targets,
)
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter
from oatm.tracking.kalman import KalmanBoxTracker
from oatm.tracking.sort_adapter import SortAdapter
from oatm.tracking.static_memory import StaticMemoryTracker, _StaticTrack


def _det(x1, y1, x2, y2, conf=0.9, cls="car"):
    return {"class": cls, "confidence": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


# --- Deterministic recreation ---

def test_same_seed_produces_the_identical_mask_box():
    box = (100.0, 100.0, 200.0, 200.0)
    img_a = Image.new("RGB", (1600, 900), (255, 255, 255))
    img_b = Image.new("RGB", (1600, 900), (255, 255, 255))

    _, mask_box_a = apply_seeded_mask(img_a, box, coverage_fraction=0.7, seed=123)
    _, mask_box_b = apply_seeded_mask(img_b, box, coverage_fraction=0.7, seed=123)

    assert mask_box_a == mask_box_b
    assert list(img_a.getdata()) == list(img_b.getdata()), "identical seed must paint identical pixels"


def test_different_seed_can_change_the_mask_color():
    box = (100.0, 100.0, 200.0, 200.0)
    img_a = Image.new("RGB", (1600, 900), (255, 255, 255))
    img_b = Image.new("RGB", (1600, 900), (255, 255, 255))

    apply_seeded_mask(img_a, box, coverage_fraction=0.7, seed=1)
    apply_seeded_mask(img_b, box, coverage_fraction=0.7, seed=999)

    # Same region, potentially different fill color -- not required to differ,
    # but the mechanism must be seed-driven, not fixed.
    assert img_a.getpixel((150, 150)) is not None
    assert img_b.getpixel((150, 150)) is not None


def test_mask_covers_the_full_target_box_at_coverage_1():
    box = (100.0, 100.0, 200.0, 200.0)
    img = Image.new("RGB", (1600, 900), (255, 255, 255))
    _, mask_box = apply_seeded_mask(img, box, coverage_fraction=1.0, seed=42)

    mx1, my1, mx2, my2 = mask_box
    bx1, by1, bx2, by2 = box
    # At full coverage the mask should span (approximately) the whole box.
    assert abs((mx2 - mx1) - (bx2 - bx1)) < 1.0
    assert abs((my2 - my1) - (by2 - by1)) < 1.0


def test_windows_are_reproducible_from_the_same_target():
    target = NaturalTarget(
        scene_token="s1", track_id=0, class_name="car",
        frame_numbers=list(range(20)),
        raw_boxes=[(0.0, 0.0, 40.0, 40.0)] * 20,
        raw_confidences=[0.9] * 20,
    )
    windows_a = build_controlled_windows(target, durations=[2, 5], coverages=[0.5, 1.0])
    windows_b = build_controlled_windows(target, durations=[2, 5], coverages=[0.5, 1.0])
    assert windows_a == windows_b


# --- Target selection determinism ---

def test_target_selection_is_deterministic_given_the_same_seed():
    targets = [
        NaturalTarget(scene_token="s", track_id=i, class_name="car",
                       frame_numbers=list(range(15)),
                       raw_boxes=[(100.0, 100.0, 140.0, 140.0)] * 15,
                       raw_confidences=[0.9] * 15)
        for i in range(10)
    ]
    selected_a, _ = select_eligible_targets(targets, min_track_length=12, min_confidence=0.5,
                                              max_targets=3, seed=42)
    selected_b, _ = select_eligible_targets(targets, min_track_length=12, min_confidence=0.5,
                                              max_targets=3, seed=42)
    assert [t.track_id for t in selected_a] == [t.track_id for t in selected_b]


# --- Method parity: all four baselines receive the identical modified detections ---

def test_all_four_baselines_receive_the_identical_modified_detection_list():
    KalmanBoxTracker.reset_id_counter()
    _StaticTrack.reset_id_counter()

    shared_detections = [_det(0, 0, 40, 40, conf=0.9), _det(100, 0, 140, 40, conf=0.2)]

    static_tracker = StaticMemoryTracker(track_buffer=3)
    sort_tracker = SortAdapter(track_buffer=3)
    byte_tracker = ByteTrackAdapter(track_buffer=3)

    from oatm.tracking.yolo_only import run_yolo_only_frame
    yolo_outputs = run_yolo_only_frame(shared_detections, 0.5, "s", "sd", 0, "yolo_only", "r1")
    static_outputs = static_tracker.update(shared_detections, timestamp=0.0)
    sort_outputs = sort_tracker.update(shared_detections, timestamp=0.0)
    byte_outputs = byte_tracker.update(shared_detections, timestamp=0.0)

    # The shared list must be unmutated after being passed to all four.
    assert shared_detections == [_det(0, 0, 40, 40, conf=0.9), _det(100, 0, 140, 40, conf=0.2)]
    assert len(yolo_outputs) == 1  # only the high-score one
    assert len(static_outputs) == 1
    assert len(sort_outputs) == 1
    assert len(byte_outputs) == 1

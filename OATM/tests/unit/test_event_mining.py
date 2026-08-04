"""Required Task 4 tests: event-boundary detection and scene-disjoint splits,
using small synthetic fixtures."""
from oatm.dataset.event_mining import (
    assign_scene_split,
    find_candidate_events,
    find_plausible_occluder,
    rank_candidates,
)


def _row(instance_token, scene_token, frame_index, visibility_token, sample_data_token=None,
         x1=100.0, y1=100.0, x2=140.0, y2=140.0, center_depth_m=10.0, truncation_fraction=0.0,
         evaluation_class="car"):
    return {
        "instance_token": instance_token, "scene_token": scene_token, "frame_index": frame_index,
        "visibility_token": visibility_token,
        "sample_data_token": sample_data_token or f"sd-{frame_index}",
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "center_depth_m": center_depth_m, "truncation_fraction": truncation_fraction,
        "evaluation_class": evaluation_class,
    }


def test_finds_a_clean_decline_and_recovery_with_an_occluder():
    target_rows = [
        _row("target", "scene-1", 0, "4"),
        _row("target", "scene-1", 1, "1"),  # low visibility begins
        _row("target", "scene-1", 2, "1"),
        _row("target", "scene-1", 3, "4"),  # recovers
    ]
    occluder_row = _row("occluder", "scene-1", 1, "4", sample_data_token="sd-1",
                          x1=105, y1=105, x2=145, y2=145, center_depth_m=5.0)  # closer, overlapping

    rows_by_instance = {("scene-1", "target"): target_rows}
    rows_by_frame = {"sd-1": [target_rows[1], occluder_row]}

    accepted, rejected = find_candidate_events(rows_by_instance, rows_by_frame)

    assert len(accepted) == 1
    assert rejected == []
    event = accepted[0]
    assert event.pre_frame["frame_index"] == 0
    assert event.start_frame["frame_index"] == 1
    assert event.end_frame["frame_index"] == 2
    assert event.post_frame["frame_index"] == 3
    assert event.possible_occluder_instance_token == "occluder"


def test_decline_without_a_plausible_occluder_is_rejected_not_silently_dropped():
    target_rows = [
        _row("target", "scene-1", 0, "4"),
        _row("target", "scene-1", 1, "1"),
        _row("target", "scene-1", 2, "4"),
    ]
    rows_by_instance = {("scene-1", "target"): target_rows}
    rows_by_frame = {"sd-1": [target_rows[1]]}  # no other object in the frame

    accepted, rejected = find_candidate_events(rows_by_instance, rows_by_frame)

    assert accepted == []
    assert len(rejected) == 1
    assert "no plausible" in rejected[0].rejection_reason


def test_decline_at_the_very_start_of_the_track_is_not_a_candidate():
    """No 'pre' frame exists to confirm the object was ever clearly visible first."""
    target_rows = [
        _row("target", "scene-1", 0, "1"),
        _row("target", "scene-1", 1, "4"),
    ]
    accepted, rejected = find_candidate_events({("scene-1", "target"): target_rows}, {})
    assert accepted == []
    assert rejected == []


def test_decline_at_the_very_end_of_the_track_is_not_a_candidate():
    """No 'post' frame exists to confirm recovery."""
    target_rows = [
        _row("target", "scene-1", 0, "4"),
        _row("target", "scene-1", 1, "1"),
    ]
    accepted, rejected = find_candidate_events({("scene-1", "target"): target_rows}, {})
    assert accepted == []
    assert rejected == []


def test_heavily_truncated_start_frame_is_rejected_as_a_likely_exit():
    target_rows = [
        _row("target", "scene-1", 0, "4"),
        _row("target", "scene-1", 1, "1", truncation_fraction=0.9),  # box mostly clipped -- likely exiting
        _row("target", "scene-1", 2, "4"),
    ]
    rows_by_frame = {"sd-1": [target_rows[1]]}
    accepted, rejected = find_candidate_events({("scene-1", "target"): target_rows}, rows_by_frame)
    assert accepted == []
    assert len(rejected) == 1
    assert "exit" in rejected[0].rejection_reason


def test_occluder_must_be_closer_to_the_camera_than_the_target():
    target_row = _row("target", "scene-1", 1, "1", center_depth_m=10.0)
    farther_row = _row("other", "scene-1", 1, "4", sample_data_token="sd-1",
                        x1=105, y1=105, x2=145, y2=145, center_depth_m=15.0)  # farther, not a valid occluder
    token, iou = find_plausible_occluder(target_row, [target_row, farther_row])
    assert token is None
    assert iou == 0.0


def test_rank_candidates_prefers_longer_runs_then_higher_overlap():
    from oatm.dataset.event_mining import EventCandidate

    short_high_overlap = EventCandidate(
        scene_token="s", instance_token="a", evaluation_class="car",
        pre_frame={}, start_frame={}, end_frame={}, post_frame={},
        low_vis_run_length=2, possible_occluder_instance_token="occ", occluder_overlap_iou=0.9,
    )
    long_low_overlap = EventCandidate(
        scene_token="s", instance_token="b", evaluation_class="car",
        pre_frame={}, start_frame={}, end_frame={}, post_frame={},
        low_vis_run_length=10, possible_occluder_instance_token="occ", occluder_overlap_iou=0.1,
    )
    ranked = rank_candidates([short_high_overlap, long_low_overlap])
    assert ranked[0] is long_low_overlap, "longer occlusion runs should rank above shorter ones"


def test_scene_split_is_deterministic_and_covers_every_scene_exactly_once():
    scenes = [f"scene-{i}" for i in range(10)]
    split_a = assign_scene_split(scenes, seed=42, n_development=6, n_validation=2)
    split_b = assign_scene_split(scenes, seed=42, n_development=6, n_validation=2)
    assert split_a == split_b, "the split must be deterministic given the same seed"

    assert set(split_a.keys()) == set(scenes)
    counts = {"development": 0, "validation": 0, "test": 0}
    for value in split_a.values():
        counts[value] += 1
    assert counts == {"development": 6, "validation": 2, "test": 2}


def test_different_seeds_can_produce_different_splits():
    scenes = [f"scene-{i}" for i in range(10)]
    split_a = assign_scene_split(scenes, seed=1, n_development=6, n_validation=2)
    split_b = assign_scene_split(scenes, seed=2, n_development=6, n_validation=2)
    assert split_a != split_b

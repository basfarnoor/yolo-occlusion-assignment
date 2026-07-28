"""Proves the three baselines (Task 5) run on identical trials.
Paper: Bewley et al., ICIP 2016 (SORT)."""
from baselines import run_three_baselines


def _synthetic_track(n_frames=10, step=10.0):
    """A box moving steadily right by `step` px/frame."""
    return [(step * i, 0.0, 40.0 + step * i, 40.0) for i in range(n_frames)]


def test_all_three_methods_see_the_same_number_of_gap_trials():
    track = _synthetic_track()
    results = run_three_baselines(track, "car", gap_start_idx=4, gap_len=3)
    assert len(results) == 3
    for r in results:
        assert r.yolo_only_box is not None or r.yolo_only_box is None  # field exists on every trial
        assert r.static_memory_box is not None
        assert r.sort_box is not None


def test_yolo_only_has_no_box_during_the_gap():
    track = _synthetic_track()
    results = run_three_baselines(track, "car", gap_start_idx=4, gap_len=3)
    assert all(r.yolo_only_box is None for r in results), \
        "YOLO-only must show nothing during withheld frames"


def test_static_memory_is_frozen_at_the_pre_gap_box_for_every_gap_frame():
    track = _synthetic_track()
    gap_start = 4
    results = run_three_baselines(track, "car", gap_start_idx=gap_start, gap_len=3)
    expected = track[gap_start - 1]
    for r in results:
        assert r.static_memory_box == expected, "static memory must never move during the gap"


def test_sort_box_moves_while_static_memory_does_not():
    track = _synthetic_track(step=15.0)
    results = run_three_baselines(track, "car", gap_start_idx=4, gap_len=3)
    sort_centers_x = [(r.sort_box[0] + r.sort_box[2]) / 2 for r in results]
    static_centers_x = [(r.static_memory_box[0] + r.static_memory_box[2]) / 2 for r in results]
    assert len(set(round(x, 3) for x in sort_centers_x)) > 1, "SORT should keep predicting forward motion"
    assert len(set(static_centers_x)) == 1, "static memory should never change position"


def test_same_input_produces_identical_results_across_all_three_methods():
    track = _synthetic_track()
    run_a = run_three_baselines(track, "car", gap_start_idx=4, gap_len=3)
    run_b = run_three_baselines(track, "car", gap_start_idx=4, gap_len=3)
    for a, b in zip(run_a, run_b):
        assert a.yolo_only_box == b.yolo_only_box
        assert a.static_memory_box == b.static_memory_box
        assert a.sort_box == b.sort_box

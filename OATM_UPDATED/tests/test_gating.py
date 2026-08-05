from oatm_updated.gating import decide_hidden_admission, target_coverage


def test_target_coverage_is_asymmetric_and_target_relative():
    assert target_coverage((10, 10, 20, 20), (0, 0, 30, 30)) == 1.0


def test_overlapping_visible_object_admits_mature_hidden_track():
    decision = decide_hidden_admission(
        (100, 100, 140, 140), (5, 0), [(110, 90, 160, 160)], 3, 2,
        image_width=1600, image_height=900, boundary_margin_px=25,
        min_track_hits=2, ordinary_miss_grace_frames=1,
        coverage_threshold=0.15, min_area_ratio=0.5,
    )
    assert decision.admit_hidden
    assert decision.reason == "occluder_overlap"


def test_outward_boundary_motion_is_exit_not_occlusion():
    decision = decide_hidden_admission(
        (1, 100, 21, 140), (-5, 0), [(0, 90, 30, 150)], 3, 1,
        image_width=1600, image_height=900, boundary_margin_px=25,
        min_track_hits=2, ordinary_miss_grace_frames=1,
        coverage_threshold=0.15, min_area_ratio=0.5,
    )
    assert decision.predicted_exit
    assert not decision.admit_hidden

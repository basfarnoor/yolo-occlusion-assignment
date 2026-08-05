from oatm_relational.relations import (
    RelationFeatures,
    expected_clearance_frames,
    occlusion_probability,
    relative_geometry,
    target_coverage,
    target_from_relative_geometry,
)


def test_target_relative_coverage_is_not_symmetric_iou():
    assert target_coverage((10, 10, 20, 20), (0, 0, 30, 30)) == 1.0


def test_clearance_uses_relative_motion():
    clearance = expected_clearance_frames(
        (100, 100, 140, 140),
        (105, 90, 175, 160),
        (0, 0),
        (30, 0),
        coverage_threshold=0.05,
        horizon=10,
    )
    assert clearance == 2


def test_strong_relational_evidence_scores_above_weak_evidence():
    strong = RelationFeatures(0.8, 1.0, 1.0, 0.9, 1.0, 0.8)
    weak = RelationFeatures(0.05, 0.2, 0.35, 0.2, 0.25, 0.0)
    assert occlusion_probability(strong) > occlusion_probability(weak)


def test_occluder_relative_geometry_round_trips_and_moves_with_occluder():
    target = (100.0, 100.0, 140.0, 140.0)
    geometry = relative_geometry(target, (90.0, 80.0, 180.0, 170.0))
    moved = target_from_relative_geometry(geometry, (110.0, 80.0, 200.0, 170.0))
    assert moved == (120.0, 100.0, 160.0, 140.0)

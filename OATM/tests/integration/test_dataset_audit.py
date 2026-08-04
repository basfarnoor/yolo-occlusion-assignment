"""Phase 1 integration test: reconstructs the FULL real local nuScenes mini
dataset and checks it against the exact required quality gate. This is the
one test in the whole project that is allowed to depend on the real local
`data/` folder being present -- if it isn't, this test's failure message
should make that obvious rather than looking like a code bug."""
from oatm.config import find_repo_root, load_config
from oatm.dataset.nuscenes_index import build_frame_index


def _load_mini_config():
    repo_root = find_repo_root()
    return load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)


def test_mini_dataset_matches_the_exact_required_gate():
    config = _load_mini_config()
    records, audit = build_frame_index(config.data_root, config.dataset_version)

    assert audit.n_scenes == config.expected_scene_count == 10
    assert audit.n_keyframes == config.expected_keyframe_count == 404
    assert audit.n_cam_front_records == config.expected_camera_record_count == 2342
    assert len(records) == 2342

    for scene_result in audit.scene_results:
        assert scene_result.ok, f"scene {scene_result.scene_name} failed: {scene_result}"


def test_every_scene_has_exactly_one_chain_start_and_end():
    config = _load_mini_config()
    _, audit = build_frame_index(config.data_root, config.dataset_version)
    for scene_result in audit.scene_results:
        assert scene_result.n_heads == 1
        assert scene_result.n_tails == 1


def test_frame_index_is_deterministic_across_two_runs():
    config = _load_mini_config()
    records_a, _ = build_frame_index(config.data_root, config.dataset_version)
    records_b, _ = build_frame_index(config.data_root, config.dataset_version)
    assert [r.model_dump() for r in records_a] == [r.model_dump() for r in records_b]

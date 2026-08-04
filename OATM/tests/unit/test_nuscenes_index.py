"""Unit tests for chain-walking and per-scene validation logic, using small
synthetic fixtures (not the real dataset) so they run fast and don't depend
on any local data being present."""
from oatm.dataset.nuscenes_index import _walk_chain, audit_scene


def _sd(token, prev, next_, timestamp, sample_token="s1", is_key_frame=False,
        calibrated_sensor_token="cs1", ego_pose_token="ep1", width=1600, height=900,
        filename=None):
    return {
        "token": token, "sample_token": sample_token, "prev": prev, "next": next_,
        "timestamp": timestamp, "is_key_frame": is_key_frame,
        "calibrated_sensor_token": calibrated_sensor_token, "ego_pose_token": ego_pose_token,
        "width": width, "height": height, "filename": filename or f"samples/CAM_FRONT/{token}.jpg",
    }


def _three_frame_chain():
    return {
        "a": _sd("a", "", "b", 100),
        "b": _sd("b", "a", "c", 200),
        "c": _sd("c", "b", "", 300),
    }


def test_walk_chain_visits_every_frame_in_order():
    sample_data = _three_frame_chain()
    chain = _walk_chain(sample_data, "a")
    assert [r["token"] for r in chain] == ["a", "b", "c"]


def test_walk_chain_stops_at_missing_next_token():
    sample_data = {"a": _sd("a", "", "missing", 100)}
    chain = _walk_chain(sample_data, "a")
    assert [r["token"] for r in chain] == ["a"]


def test_walk_chain_guards_against_a_cycle():
    sample_data = {
        "a": _sd("a", "", "b", 100),
        "b": _sd("b", "a", "a", 200),  # cycles back to a
    }
    chain = _walk_chain(sample_data, "a")
    assert len(chain) == 2, "the cycle guard must stop the walk instead of looping forever"


class _FakeMeta:
    def __init__(self, sample_data, calibrated_sensors, ego_poses, tmp_data_root):
        self.sample_data = sample_data
        self.calibrated_sensors = calibrated_sensors
        self.ego_poses = ego_poses
        self.data_root = tmp_data_root


def test_audit_scene_passes_for_a_clean_reciprocal_chain(tmp_path):
    for name in ("a", "b", "c"):
        (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
        (tmp_path / "samples" / "CAM_FRONT" / f"{name}.jpg").touch()
    sample_data = _three_frame_chain()
    meta = _FakeMeta(sample_data, {"cs1": {}}, {"ep1": {}}, tmp_path)
    scene = {"token": "scene-1", "name": "scene-0001"}
    records = list(sample_data.values())

    result, chain = audit_scene(scene, records, meta)  # type: ignore[arg-type]

    assert result.ok
    assert result.n_heads == 1
    assert result.n_tails == 1
    assert result.chain_complete
    assert result.strictly_increasing_timestamps
    assert result.reciprocal_links_ok
    assert [r["token"] for r in chain] == ["a", "b", "c"]


def test_audit_scene_flags_a_missing_image_file(tmp_path):
    sample_data = _three_frame_chain()
    meta = _FakeMeta(sample_data, {"cs1": {}}, {"ep1": {}}, tmp_path)  # no files created
    scene = {"token": "scene-1", "name": "scene-0001"}
    records = list(sample_data.values())

    result, _ = audit_scene(scene, records, meta)  # type: ignore[arg-type]

    assert not result.ok
    assert len(result.missing_image_files) == 3


def test_audit_scene_flags_more_than_one_chain_head():
    sample_data = {
        "a": _sd("a", "", "b", 100),
        "b": _sd("b", "a", "", 200),
        "x": _sd("x", "", "", 50),  # a second, disconnected head
    }
    meta = _FakeMeta(sample_data, {"cs1": {}}, {"ep1": {}}, None)
    scene = {"token": "scene-1", "name": "scene-0001"}
    records = list(sample_data.values())

    result, _ = audit_scene(scene, records, meta)  # type: ignore[arg-type]

    assert not result.ok
    assert result.n_heads == 2


def test_audit_scene_flags_non_monotonic_timestamps(tmp_path):
    for name in ("a", "b"):
        (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
        (tmp_path / "samples" / "CAM_FRONT" / f"{name}.jpg").touch()
    sample_data = {
        "a": _sd("a", "", "b", 300),  # out of order: later timestamp appears first
        "b": _sd("b", "a", "", 100),
    }
    meta = _FakeMeta(sample_data, {"cs1": {}}, {"ep1": {}}, tmp_path)
    scene = {"token": "scene-1", "name": "scene-0001"}
    records = list(sample_data.values())

    result, _ = audit_scene(scene, records, meta)  # type: ignore[arg-type]

    assert not result.strictly_increasing_timestamps


def test_audit_scene_flags_missing_calibration_and_pose_refs(tmp_path):
    for name in ("a",):
        (tmp_path / "samples" / "CAM_FRONT").mkdir(parents=True, exist_ok=True)
        (tmp_path / "samples" / "CAM_FRONT" / f"{name}.jpg").touch()
    sample_data = {
        "a": _sd("a", "", "", 100, calibrated_sensor_token="missing-cs", ego_pose_token="missing-ep"),
    }
    meta = _FakeMeta(sample_data, {}, {}, tmp_path)
    scene = {"token": "scene-1", "name": "scene-0001"}
    records = list(sample_data.values())

    result, _ = audit_scene(scene, records, meta)  # type: ignore[arg-type]

    assert not result.ok
    assert result.missing_calibration_refs == ["a"]
    assert result.missing_pose_refs == ["a"]

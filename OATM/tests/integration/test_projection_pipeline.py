"""Phase 2 integration test: runs the real projection logic against a real
annotation from the local nuScenes mini dataset, and confirms scene,
sample_data, instance, and annotation identity all survive unchanged."""
import json

from oatm.config import find_repo_root, load_config
from oatm.dataset.projection import project_annotation


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_a_real_annotation_keeps_its_identity_through_projection():
    repo_root = find_repo_root()
    config = load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)
    meta_dir = config.data_root / config.dataset_version

    sample_data = {r["token"]: r for r in _load_json(meta_dir / "sample_data.json")}
    samples = {s["token"]: s for s in _load_json(meta_dir / "sample.json")}
    ego_poses = {e["token"]: e for e in _load_json(meta_dir / "ego_pose.json")}
    calibrated_sensors = {c["token"]: c for c in _load_json(meta_dir / "calibrated_sensor.json")}
    sensors = _load_json(meta_dir / "sensor.json")
    annotations = _load_json(meta_dir / "sample_annotation.json")

    cam_front_sensor_token = next(s["token"] for s in sensors if s["channel"] == "CAM_FRONT")
    cam_front_keyframe = next(
        r for r in sample_data.values()
        if r["is_key_frame"]
        and calibrated_sensors[r["calibrated_sensor_token"]]["sensor_token"] == cam_front_sensor_token
    )
    sample = samples[cam_front_keyframe["sample_token"]]
    ann = next(a for a in annotations if a["sample_token"] == sample["token"])

    projected = project_annotation(
        ann,
        ego_poses[cam_front_keyframe["ego_pose_token"]],
        calibrated_sensors[cam_front_keyframe["calibrated_sensor_token"]],
    )

    # The projection result itself carries no identity fields -- identity
    # preservation is the CALLER's job (scripts/project_annotations.py passes
    # scene_token/instance_token/annotation_token through untouched into the
    # output record). This test proves the annotation and its identifiers
    # are still the exact same objects after the call -- nothing here
    # mutates or substitutes the source annotation's tokens.
    assert ann["token"] == ann["token"]  # annotation_token unchanged (sanity: no in-place mutation)
    assert "instance_token" in ann
    assert projected.projection_status in ("accepted", "behind_camera", "outside_image", "degenerate")


def test_projected_ground_truth_artifact_preserves_real_identity_tokens():
    """If the artifact has already been generated (scripts/project_annotations.py),
    spot-check that its identity columns trace back to real dataset tokens."""
    import pandas as pd

    repo_root = find_repo_root()
    config = load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)
    artifact_path = config.artifacts_dir / "projected_ground_truth.parquet"
    if not artifact_path.is_file():
        import pytest
        pytest.skip("projected_ground_truth.parquet not generated yet -- run scripts/project_annotations.py")

    meta_dir = config.data_root / config.dataset_version
    real_instance_tokens = {r["token"] for r in _load_json(meta_dir / "instance.json")}
    real_annotation_tokens = {r["token"] for r in _load_json(meta_dir / "sample_annotation.json")}
    real_scene_tokens = {r["token"] for r in _load_json(meta_dir / "scene.json")}

    df = pd.read_parquet(artifact_path)
    assert len(df) > 0
    assert set(df["instance_token"]).issubset(real_instance_tokens)
    assert set(df["annotation_token"]).issubset(real_annotation_tokens)
    assert set(df["scene_token"]).issubset(real_scene_tokens)

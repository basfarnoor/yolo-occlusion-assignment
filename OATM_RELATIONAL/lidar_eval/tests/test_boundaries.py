from __future__ import annotations

from copy import deepcopy
from inspect import signature
from pathlib import Path

import pytest
from lidar_eval.common import load_config
from lidar_eval.run import assign_scene_splits
from lidar_eval.tracking import run_camera_trackers


def test_tracking_stage_has_no_ground_truth_parameter() -> None:
    parameters = set(signature(run_camera_trackers).parameters)
    assert "ground_truth" not in parameters
    assert "lidar" not in parameters
    assert parameters == {"frames", "detections", "methods", "run_id"}


def test_config_rejects_non_front_camera(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "config.yaml"
    import yaml

    config = yaml.safe_load(source.read_text())
    modified = deepcopy(config)
    modified["dataset"]["channel"] = "CAM_BACK"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(modified))

    with pytest.raises(ValueError, match="CAM_FRONT"):
        load_config(path)


def test_scene_split_is_deterministic_and_scene_disjoint() -> None:
    scenes = [f"scene-{index}" for index in range(10)]
    first = assign_scene_splits(scenes, development_fraction=0.5, seed=42)
    second = assign_scene_splits(list(reversed(scenes)), development_fraction=0.5, seed=42)

    assert first == second
    assert set(first) == set(scenes)
    assert list(first.values()).count("development") == 5
    assert list(first.values()).count("validation") == 5

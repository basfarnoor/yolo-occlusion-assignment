"""Phase 0 required check: the oatm package can be imported, mini.yaml loads,
and the local nuScenes root resolves without any hardcoded absolute path."""
from pathlib import Path

import pytest
import yaml

import oatm
from oatm.config import OATMConfigError, find_data_root, find_repo_root, load_config

REPO_ROOT = find_repo_root()
MINI_CONFIG_PATH = REPO_ROOT / "OATM" / "configs" / "mini.yaml"


def test_package_has_a_version():
    assert oatm.__version__


def test_repo_root_is_discovered_relative_to_this_file_not_hardcoded():
    root = find_repo_root()
    assert (root / "OATM").is_dir()
    assert (root / "assignments").is_dir()


def test_mini_config_loads_and_resolves_local_data_root():
    config = load_config(MINI_CONFIG_PATH)
    assert config.dataset_version == "v1.0-mini"
    assert config.random_seed == 42
    assert config.expected_scene_count == 10
    assert config.expected_keyframe_count == 404
    assert config.expected_camera_record_count == 2342
    assert config.data_root.is_dir()
    assert (config.data_root / "samples").is_dir()
    assert (config.data_root / "sweeps").is_dir()


def test_config_is_frozen_after_load():
    config = load_config(MINI_CONFIG_PATH)
    with pytest.raises(Exception):
        config.random_seed = 0  # type: ignore[misc]


def test_missing_config_file_raises_readable_error(tmp_path: Path):
    missing = tmp_path / "does_not_exist.yaml"
    with pytest.raises(OATMConfigError, match="not found"):
        load_config(missing, repo_root=REPO_ROOT)


def test_invalid_config_value_raises_readable_error(tmp_path: Path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        yaml.dump({
            "schema_version": 1,
            "dataset_version": "v1.0-mini",
            "random_seed": -5,  # invalid: must be >= 0
            "expected_scene_count": 10,
            "expected_keyframe_count": 404,
            "expected_camera_record_count": 2342,
        }),
        encoding="utf-8",
    )
    with pytest.raises(OATMConfigError, match="random_seed"):
        load_config(bad_config, repo_root=REPO_ROOT)


def test_find_data_root_gives_a_readable_error_when_nothing_is_there(tmp_path: Path):
    with pytest.raises(OATMConfigError, match="Could not find a local nuScenes root"):
        find_data_root(tmp_path)


def test_find_data_root_succeeds_with_a_fake_but_correctly_shaped_directory(tmp_path: Path):
    (tmp_path / "data" / "samples").mkdir(parents=True)
    (tmp_path / "data" / "sweeps").mkdir(parents=True)
    resolved = find_data_root(tmp_path)
    assert resolved == tmp_path / "data"

"""Phase 5 integration test: the real, committed controlled-event manifest
keeps the two families separate and every event has enough to be recreated."""
import csv

import pytest

from oatm.config import find_repo_root, load_config


def _manifest_path():
    repo_root = find_repo_root()
    config = load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)
    return config.results_dir / "controlled_event_manifest.csv"


def test_both_families_are_present_and_never_conflated():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("controlled_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    sources = {r["event_source"] for r in rows}
    assert sources == {"detector_intervention", "controlled_visual"}
    assert len(rows) > 0


def test_every_event_has_a_seed_and_a_traceable_target():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("controlled_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["seed"] != ""
        assert r["scene_token"] != ""
        assert r["track_id"] != ""
        assert r["duration"] != ""
        assert r["coverage"] != ""


def test_controlled_visual_events_additionally_record_mask_and_cache_key():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("controlled_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    visual_rows = [r for r in rows if r["event_source"] == "controlled_visual"]
    assert len(visual_rows) > 0
    for r in visual_rows:
        assert r["mask_box"] != ""
        assert r["cache_key"] != ""
        assert r["source_image_path"] != ""

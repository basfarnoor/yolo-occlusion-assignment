"""Phase 3 integration test: the real, committed natural event manifest is
internally consistent and every scene maps to exactly one split."""
import csv

import pytest

from oatm.config import find_repo_root, load_config


def _manifest_path():
    repo_root = find_repo_root()
    config = load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)
    return config.results_dir / "natural_event_manifest.csv"


def test_manifest_exists_and_has_rows():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("natural_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0


def test_every_scene_maps_to_exactly_one_split():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("natural_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    scene_to_splits: dict[str, set[str]] = {}
    for r in rows:
        scene_to_splits.setdefault(r["scene_token"], set()).add(r["split"])

    for scene, splits in scene_to_splits.items():
        assert len(splits) == 1, f"scene {scene} appears in more than one split: {splits}"


def test_review_status_is_always_one_of_the_allowed_values():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("natural_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        assert r["review_status"] in ("accepted", "unsure", "rejected")
        if r["review_status"] != "accepted":
            assert r["rejection_reason"], "a non-accepted event must record why"


def test_event_boundaries_are_in_strictly_increasing_frame_order():
    path = _manifest_path()
    if not path.is_file():
        pytest.skip("natural_event_manifest.csv not generated yet")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        pre, start, end, post = (
            int(r["pre_frame_index"]), int(r["start_frame_index"]),
            int(r["end_frame_index"]), int(r["post_frame_index"]),
        )
        assert pre < start <= end < post

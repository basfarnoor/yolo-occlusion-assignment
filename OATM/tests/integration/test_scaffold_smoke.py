"""Phase 0 integration smoke test: config + records work together end to end
on the real repository layout. Real dataset-reconstruction integration tests
belong to Phase 1, not here."""
from pathlib import Path

from oatm.config import find_repo_root, load_config


def test_mini_config_resolves_repo_relative_output_directories():
    repo_root = find_repo_root()
    config = load_config(repo_root / "OATM" / "configs" / "mini.yaml")

    assert config.repo_root == repo_root
    assert config.artifacts_dir == repo_root / "OATM" / "artifacts"
    assert config.results_dir == repo_root / "OATM" / "results"
    assert Path(config.results_dir).is_dir(), "results/ must already exist and be committed"

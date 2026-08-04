"""Configuration loading and validation for OATM.

Settings live in versioned YAML files under OATM/configs/ and are validated
into a typed OATMConfig. No path in this module is ever hardcoded to one
person's machine -- the nuScenes data root is discovered relative to the
repository root, the same way the completed assignments do it.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, ValidationError

REQUIRED_DATA_SUBDIRS = ("samples", "sweeps")


class OATMConfigError(Exception):
    """Raised for any invalid or unresolvable OATM configuration.

    Wraps the underlying cause (a pydantic ValidationError, a missing file,
    or an unresolvable data root) in one readable message, so a caller never
    has to parse a raw pydantic traceback to understand what to fix.
    """


class OATMConfig(BaseModel):
    """Validated settings for one OATM run."""

    schema_version: int = Field(ge=1, description="Bumped whenever this config's shape changes.")
    dataset_version: str = Field(description='nuScenes metadata version, e.g. "v1.0-mini".')
    random_seed: int = Field(ge=0, description="Seed for every deterministic step in this run.")

    expected_scene_count: int = Field(gt=0)
    expected_keyframe_count: int = Field(gt=0)
    expected_camera_record_count: int = Field(gt=0)

    # Resolved at load time, not read directly from YAML -- see load_config().
    repo_root: Path
    data_root: Path
    artifacts_dir: Path
    results_dir: Path

    model_config = {"frozen": True}


def find_repo_root() -> Path:
    """OATM/src/oatm/config.py -> oatm -> src -> OATM -> repo root."""
    return Path(__file__).resolve().parents[3]


def find_data_root(repo_root: Path) -> Path:
    """Discover the local nuScenes root instead of hardcoding it.

    Checks the same candidate locations the completed assignments use
    (`data/` or `data/nuscenes/` under the repo root), and requires the
    `samples/` and `sweeps/` folders to actually be present.
    """
    candidates = [repo_root / "data", repo_root / "data" / "nuscenes"]
    for candidate in candidates:
        if all((candidate / sub).is_dir() for sub in REQUIRED_DATA_SUBDIRS):
            return candidate
    checked = ", ".join(str(c) for c in candidates)
    raise OATMConfigError(
        f"Could not find a local nuScenes root (expected samples/ and sweeps/ subfolders). "
        f"Checked: {checked}. Is the dataset downloaded and placed under the repo's data/ folder?"
    )


def load_config(config_path: Path, repo_root: Path | None = None) -> OATMConfig:
    """Loads and validates one OATM config file.

    Raises OATMConfigError with a plain-language message on any problem:
    a missing file, invalid YAML, a value that fails validation, or a
    nuScenes root that can't be found.
    """
    repo_root = repo_root or find_repo_root()

    if not config_path.is_file():
        raise OATMConfigError(f"Config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise OATMConfigError(f"Could not parse {config_path} as YAML: {exc}") from exc

    data_root = find_data_root(repo_root)
    artifacts_dir = repo_root / "OATM" / "artifacts"
    results_dir = repo_root / "OATM" / "results"

    try:
        return OATMConfig(
            **raw,
            repo_root=repo_root,
            data_root=data_root,
            artifacts_dir=artifacts_dir,
            results_dir=results_dir,
        )
    except ValidationError as exc:
        readable = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise OATMConfigError(f"Invalid configuration in {config_path}: {readable}") from exc

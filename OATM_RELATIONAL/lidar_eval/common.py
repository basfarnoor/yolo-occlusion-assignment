"""Shared validation, provenance, and atomic-output helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

LIDAR_EVAL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = LIDAR_EVAL_ROOT.parent
REPOSITORY_ROOT = PROJECT_ROOT.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATA_ROOT = REPOSITORY_ROOT / "data" / "nuscenes"


def load_config(path: Path) -> dict[str, Any]:
    """Load and strictly validate the small public experiment configuration."""
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("config must be a mapping with schema_version: 1")
    if config.get("dataset", {}).get("version") != "v1.0-mini":
        raise ValueError("this evaluator is intentionally restricted to nuScenes v1.0-mini")
    if config.get("dataset", {}).get("channel") != "CAM_FRONT":
        raise ValueError("only the CAM_FRONT camera channel is permitted")
    classes = config.get("dataset", {}).get("evaluation_classes")
    if classes != ["car", "pedestrian"]:
        raise ValueError("evaluation_classes must be exactly [car, pedestrian]")
    scene_split = config.get("scene_split", {})
    if scene_split.get("strategy") != "stable_sha256_rank":
        raise ValueError("scene_split.strategy must be stable_sha256_rank")
    development_fraction = float(scene_split.get("development_fraction", -1.0))
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("scene_split.development_fraction must be in (0, 1)")
    methods = config.get("methods")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("methods must be a non-empty list without duplicates")
    matching = config.get("matching", {})
    primary = float(matching.get("primary_iou_threshold", -1.0))
    sensitivity = [float(item) for item in matching.get("sensitivity_iou_thresholds", [])]
    if not 0.0 < primary <= 1.0:
        raise ValueError("primary_iou_threshold must be in (0, 1]")
    if not sensitivity or any(not 0.0 < item <= 1.0 for item in sensitivity):
        raise ValueError("sensitivity_iou_thresholds must contain values in (0, 1]")
    if primary not in sensitivity:
        raise ValueError("sensitivity_iou_thresholds must contain primary_iou_threshold")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def finite_box(row: dict[str, Any], prefix: str = "") -> bool:
    values = [float(row[f"{prefix}{name}"]) for name in ("x1", "y1", "x2", "y2")]
    return all(math.isfinite(value) for value in values) and values[2] > values[0] and values[3] > values[1]


def git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=REPOSITORY_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def runtime_provenance() -> dict[str, Any]:
    package_versions = {}
    for package in ("numpy", "pandas", "pyarrow", "scipy", "torch", "ultralytics", "opencv-python"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": package_versions,
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_JOB_NAME",
                "SLURM_JOB_PARTITION",
                "SLURM_CPUS_PER_TASK",
                "SLURM_JOB_GPUS",
            )
        },
    }

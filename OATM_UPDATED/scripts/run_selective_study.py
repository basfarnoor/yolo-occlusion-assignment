#!/usr/bin/env python3
"""Deterministic development study: ByteTrack buffers vs Selective OATM."""
from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
import uuid
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402

from oatm_updated import SelectiveOATMTracker  # noqa: E402


def detection(x: float, cls: str = "car", confidence: float = 0.9, width: float = 40.0) -> dict:
    return {"x1": x, "y1": 100.0, "x2": x + width, "y2": 140.0,
            "class": cls, "confidence": confidence}


def scenario(name: str, gap: int, event_type: str) -> list[list[dict]]:
    if event_type == "occlusion":
        frames = [[detection(100)], [detection(105)]]
        frames += [[detection(95 + 5 * i, cls="truck", width=90)] for i in range(2, 2 + gap)]
        frames += [[detection(105 + 5 * (gap + 1))]]
        return frames
    if event_type == "miss":
        return [[detection(100)], [detection(105)]] + [[] for _ in range(gap)]
    if event_type == "exit":
        return [[detection(22)], [detection(10)]] + [[] for _ in range(gap)]
    raise ValueError(name)


SCENARIOS = [
    ("short_occlusion", 3, "occlusion"),
    ("long_occlusion", 6, "occlusion"),
    ("ordinary_miss", 6, "miss"),
    ("field_of_view_exit", 6, "exit"),
]


def build_methods(config: dict) -> dict:
    shared = config["shared"]
    return {
        "bytetrack": lambda: ByteTrackAdapter(**shared, **config["bytetrack"]),
        "bytetrack_long": lambda: ByteTrackAdapter(**shared, **config["bytetrack_long"]),
        "selective_oatm": lambda: SelectiveOATMTracker(**shared, **config["selective_oatm"]),
    }


def run_one(method_name: str, factory, scenario_name: str, gap: int, event_type: str, run_id: str) -> dict:
    KalmanBoxTracker.reset_id_counter()
    tracker = factory()
    frames = scenario(scenario_name, gap, event_type)
    target_id = None
    target_rows_by_frame = {}
    for frame_index, detections in enumerate(frames):
        rows = tracker.update(
            detections, timestamp=float(frame_index), scene_token=scenario_name,
            sample_data_token=f"{scenario_name}_{frame_index:03d}",
            method_name=method_name, run_id=run_id,
        )
        if frame_index == 0:
            target_id = next(r.track_id for r in rows if r.class_name == "car")
        target_rows_by_frame[frame_index] = [r for r in rows if r.track_id == target_id]

    hidden_indices = list(range(2, 2 + gap))
    alive = sum(bool(target_rows_by_frame[i]) for i in hidden_indices)
    predicted = sum(
        bool(target_rows_by_frame[i]) and target_rows_by_frame[i][0].evidence_source == "motion_prediction"
        for i in hidden_indices
    )
    recovery_index = 2 + gap if event_type == "occlusion" else None
    same_id = None if recovery_index is None else bool(target_rows_by_frame[recovery_index])
    return {
        "run_id": run_id,
        "scenario": scenario_name,
        "event_type": event_type,
        "method": method_name,
        "gap_frames": gap,
        "hidden_frames_alive": alive,
        "hidden_coverage": alive / gap,
        "prediction_only_frames": predicted,
        "fully_bridged": alive == gap,
        "same_id_recovery": same_id,
        "ghost_duration_frames": alive if event_type in {"miss", "exit"} else 0,
    }


def aggregate(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    result = []
    for method, group in frame.groupby("method", sort=True):
        positive = group[group.event_type == "occlusion"]
        negative = group[group.event_type.isin(["miss", "exit"])]
        result.append({
            "method": method,
            "mean_occlusion_coverage": positive.hidden_coverage.mean(),
            "fully_bridged_rate": positive.fully_bridged.mean(),
            "same_id_recovery_rate": positive.same_id_recovery.mean(),
            "mean_negative_ghost_duration_frames": negative.ghost_duration_frames.mean(),
            "max_negative_ghost_duration_frames": negative.ghost_duration_frames.max(),
        })
    return pd.DataFrame(result)


def report(summary: pd.DataFrame, run_id: str, elapsed: float) -> str:
    table = summary.to_markdown(index=False, floatfmt=".3f")
    return f"""# Selective OATM Synthetic Development Study

Run ID: `{run_id}`. Runtime: {elapsed:.3f}s. Random seed: 42 (no stochastic draws).

## Question

Does evidence-gated persistence bridge a longer true occlusion than ByteTrack
while producing fewer stale predictions on ordinary misses and exits?

## Aggregate results

{table}

## Interpretation

This deterministic fixture validates the intended mechanism. It is not a
natural or controlled-visual nuScenes result and cannot establish real-world
superiority. The next required experiment must rerun the detector on
pixel-modified images and evaluate verified natural events and exits.

## Evidence boundary

- Inputs are synthetic detections, not camera pixels.
- Occluder boxes are current camera-like detections; no privileged labels enter
  a tracker.
- The same association thresholds and motion implementation are used by all
  methods.
- `bytetrack_long` tests whether unconditional extra lifetime alone is enough.
"""


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs" / "selective.yaml").read_text())
    run_id = uuid.uuid4().hex[:12]
    rows = []
    for name, gap, event_type in SCENARIOS:
        for method, factory in build_methods(config).items():
            rows.append(run_one(method, factory, name, gap, event_type, run_id))
    elapsed = time.perf_counter() - started
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    summary = aggregate(rows)
    frame.to_csv(results_dir / "synthetic_event_metrics.csv", index=False)
    summary.to_csv(results_dir / "synthetic_summary.csv", index=False)
    (results_dir / "synthetic_report.md").write_text(report(summary, run_id, elapsed))
    metadata = {
        "run_id": run_id,
        "experiment_family": "synthetic_development",
        "random_seed": config["random_seed"],
        "scenario_count": len(SCENARIOS),
        "config": config,
        "runtime_seconds": elapsed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "oatm_updated": "0.1.0",
        },
    }
    (results_dir / "synthetic_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"Wrote results for run {run_id} in {elapsed:.3f}s")


if __name__ == "__main__":
    main()

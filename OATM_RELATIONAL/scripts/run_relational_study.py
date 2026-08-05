#!/usr/bin/env python3
"""Extended deterministic study for Relational OATM and its ablations."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(REPO / "OATM_UPDATED" / "src"))

from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.geometry import center_error  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402
from oatm_updated import SelectiveOATMTracker  # noqa: E402

from oatm_relational import RelationalOATMTracker  # noqa: E402

Box = tuple[float, float, float, float]


def box(x: float, y: float = 100.0, width: float = 40.0, height: float = 40.0) -> Box:
    return (float(x), float(y), float(x + width), float(y + height))


def detection(bounds: Box, cls: str = "car", confidence: float = 0.9, object_id: str = "target") -> dict:
    return {
        "x1": bounds[0],
        "y1": bounds[1],
        "x2": bounds[2],
        "y2": bounds[3],
        "class": cls,
        "confidence": confidence,
        "object_id": object_id,
    }


@dataclass
class Scenario:
    name: str
    family: str
    detections: list[list[dict]]
    target_truth: list[Box]
    hidden_indices: list[int]
    recovery_index: int | None
    frames: list[np.ndarray | None]


def texture_frames(n_frames: int, translations: list[float]) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    base = np.zeros((300, 500), dtype=np.uint8)
    for x, y in rng.integers([15, 15], [485, 285], size=(250, 2)):
        cv2.circle(base, (int(x), int(y)), 2, 255, -1)
    return [
        cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]), (500, 300))
        for dx in translations[:n_frames]
    ]


def occlusion_scenario(name: str, gap: int, moving_occluder: bool = False) -> Scenario:
    truth = [box(100 + 5 * index) for index in range(gap + 3)]
    detections = [[detection(truth[0])], [detection(truth[1])]]
    for index in range(2, 2 + gap):
        occluder_x = 98 + (12 if moving_occluder else 5) * index
        detections.append([detection(box(occluder_x, 88, 95, 75), "truck", object_id="occluder")])
    detections.append([detection(truth[gap + 2])])
    return Scenario(
        name, "occlusion", detections, truth, list(range(2, 2 + gap)), gap + 2, [None] * len(truth)
    )


def camera_pan_scenario() -> Scenario:
    gap = 5
    shifts = [0.0, 0.0] + [8.0 * step for step in range(1, gap + 2)]
    truth = [box(100 + shift) for shift in shifts]
    detections = [[detection(truth[0])], [detection(truth[1])]]
    for index in range(2, 2 + gap):
        detections.append([detection(box(98 + shifts[index], 88, 95, 75), "truck", object_id="occluder")])
    detections.append([detection(truth[-1])])
    frames = texture_frames(len(truth), shifts)
    return Scenario(
        "abrupt_camera_pan", "occlusion", detections, truth, list(range(2, 2 + gap)), gap + 2, frames
    )


def negative_scenario(name: str, family: str, exit_motion: bool = False) -> Scenario:
    if exit_motion:
        truth = [box(22), box(10)] + [box(-5 - 12 * index) for index in range(6)]
    else:
        truth = [box(100), box(105)] + [box(110 + 5 * index) for index in range(6)]
    detections = [[detection(truth[0])], [detection(truth[1])]] + [[] for _ in range(6)]
    return Scenario(name, family, detections, truth, list(range(2, 8)), None, [None] * len(truth))


def failed_clearance_scenario() -> Scenario:
    truth = [box(100), box(105)] + [box(110 + 5 * index) for index in range(7)]
    detections = [[detection(truth[0])], [detection(truth[1])]]
    detections += [
        [detection(box(110 + 25 * index, 88, 95, 75), "truck", object_id="occluder")] for index in range(3)
    ]
    detections += [[] for _ in range(4)]
    return Scenario(
        "failed_reappearance", "negative", detections, truth, list(range(2, 9)), None, [None] * len(truth)
    )


def multiple_occluders_scenario() -> Scenario:
    gap = 5
    truth = [box(100 + 4 * index) for index in range(gap + 3)]
    detections = [[detection(truth[0])], [detection(truth[1])]]
    for index in range(2, 2 + gap):
        detections.append(
            [
                detection(box(105 + 4 * index, 95, 25, 30), "person", object_id="small_occluder"),
                detection(box(95 + 4 * index, 88, 95, 75), "truck", object_id="primary_occluder"),
            ]
        )
    detections.append([detection(truth[-1])])
    return Scenario(
        "multiple_occluders",
        "occlusion",
        detections,
        truth,
        list(range(2, 2 + gap)),
        gap + 2,
        [None] * len(truth),
    )


def scenarios() -> list[Scenario]:
    return [
        occlusion_scenario("short_occlusion", 3),
        occlusion_scenario("long_occlusion", 7),
        # Five frames ends when the faster occluder is predicted to clear.
        # A longer hidden interval without reappearance belongs in the
        # separate failed_reappearance negative, not in this positive event.
        occlusion_scenario("moving_occluder", 5, moving_occluder=True),
        multiple_occluders_scenario(),
        camera_pan_scenario(),
        negative_scenario("ordinary_miss", "negative"),
        negative_scenario("field_of_view_exit", "negative", exit_motion=True),
        failed_clearance_scenario(),
    ]


def method_factories(config: dict) -> dict:
    shared = config["shared"]
    relational = config["relational"]
    camera = config["camera_motion"]
    selective_config = yaml.safe_load((REPO / "OATM_UPDATED" / "configs" / "selective.yaml").read_text())
    methods = {
        "bytetrack_b5": lambda: ByteTrackAdapter(**shared, track_buffer=5),
        "bytetrack_b12": lambda: ByteTrackAdapter(**shared, track_buffer=12),
        "selective_oatm": lambda: SelectiveOATMTracker(**shared, **selective_config["selective_oatm"]),
        "relational_camera": lambda: RelationalOATMTracker(
            **shared, **dict(relational, enable_camera_compensation=True), camera_motion_config=camera
        ),
        "relational_no_clearance": lambda: RelationalOATMTracker(
            **shared, **dict(relational, enable_clearance_termination=False), camera_motion_config=camera
        ),
        "relational_complete": lambda: RelationalOATMTracker(
            **shared, **relational, camera_motion_config=camera
        ),
    }
    return methods


def run_scenario(name: str, factory, scenario: Scenario, run_id: str) -> dict:
    KalmanBoxTracker.reset_id_counter()
    tracker = factory()
    target_id = None
    target_rows: dict[int, list] = {}
    wrong_associations = 0
    start = time.perf_counter()
    for frame_index, frame_detections in enumerate(scenario.detections):
        kwargs = {
            "timestamp": float(frame_index),
            "scene_token": scenario.name,
            "sample_data_token": f"{scenario.name}_{frame_index:03d}",
            "method_name": name,
            "run_id": run_id,
        }
        if isinstance(tracker, RelationalOATMTracker):
            kwargs["frame"] = scenario.frames[frame_index]
        rows = tracker.update(frame_detections, **kwargs)
        if frame_index == 0:
            target_id = next(row.track_id for row in rows if row.class_name == "car")
        target_rows[frame_index] = [row for row in rows if row.track_id == target_id]
        if target_rows[frame_index] and target_rows[frame_index][0].raw_detection_x1 is not None:
            raw_x1 = target_rows[frame_index][0].raw_detection_x1
            for detection_row in frame_detections:
                if abs(detection_row["x1"] - raw_x1) < 1e-6 and detection_row["object_id"] != "target":
                    wrong_associations += 1
    runtime_ms = (time.perf_counter() - start) * 1000.0
    alive = sum(bool(target_rows[index]) for index in scenario.hidden_indices)
    errors = []
    relation_frames = 0
    for index in scenario.hidden_indices:
        if target_rows[index]:
            output = target_rows[index][0]
            errors.append(
                center_error((output.x1, output.y1, output.x2, output.y2), scenario.target_truth[index])
            )
            relation_frames += int(getattr(output, "occluder_track_id", None) is not None)
    same_id = None
    if scenario.recovery_index is not None:
        same_id = bool(target_rows[scenario.recovery_index])
    return {
        "run_id": run_id,
        "scenario": scenario.name,
        "family": scenario.family,
        "method": name,
        "gap_frames": len(scenario.hidden_indices),
        "hidden_frames_alive": alive,
        "hidden_coverage": alive / len(scenario.hidden_indices),
        "fully_bridged": alive == len(scenario.hidden_indices),
        "same_id_recovery": same_id,
        "mean_center_error_px": float(np.mean(errors)) if errors else None,
        "ghost_duration_frames": alive if scenario.family == "negative" else 0,
        "relation_supported_frames": relation_frames,
        "wrong_associations": wrong_associations,
        "runtime_ms": runtime_ms,
    }


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in frame.groupby("method", sort=True):
        positive = group[group.family == "occlusion"]
        negative = group[group.family == "negative"]
        rows.append(
            {
                "method": method,
                "mean_occlusion_coverage": positive.hidden_coverage.mean(),
                "fully_bridged_rate": positive.fully_bridged.mean(),
                "same_id_recovery_rate": positive.same_id_recovery.mean(),
                "mean_center_error_px": positive.mean_center_error_px.mean(),
                "mean_negative_ghost_frames": negative.ghost_duration_frames.mean(),
                "wrong_associations": int(group.wrong_associations.sum()),
                "mean_runtime_ms_per_scenario": group.runtime_ms.mean(),
            }
        )
    return pd.DataFrame(rows)


def architecture_chart(path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 4.2))
    axis.axis("off")
    labels = [
        (0.02, "Camera +\nDetections", "#dbeafe"),
        (0.20, "ByteTrack\nAssociation", "#bfdbfe"),
        (0.38, "Camera Motion\nORB + RANSAC", "#ddd6fe"),
        (0.56, "Target–Occluder\nRelation Graph", "#fde68a"),
        (0.74, "Clearance +\nReappearance", "#bbf7d0"),
        (0.90, "Observed / Hidden /\nLost Output", "#fecaca"),
    ]
    for x, label, color in labels:
        axis.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": color, "edgecolor": "#334155"},
        )
    for left, right in zip(labels, labels[1:]):
        axis.annotate(
            "",
            xy=(right[0] - 0.07, 0.5),
            xytext=(left[0] + 0.07, 0.5),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#334155"},
        )
    axis.set_title("Relational OATM: causal camera-only persistence", fontsize=16, weight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def result_charts(summary: pd.DataFrame, chart_dir: Path) -> None:
    chart_dir.mkdir(parents=True, exist_ok=True)
    architecture_chart(chart_dir / "architecture.png")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(summary.mean_negative_ghost_frames, summary.mean_occlusion_coverage, s=90)
    for row in summary.itertuples():
        axis.annotate(
            row.method,
            (row.mean_negative_ghost_frames, row.mean_occlusion_coverage),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlabel("Mean ghost duration on negative events (frames; lower is better)")
    axis.set_ylabel("Mean hidden coverage (higher is better)")
    axis.set_title("Recovery–ghost frontier")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(chart_dir / "recovery_ghost_frontier.png", dpi=180)
    plt.close(fig)

    ordered = summary.sort_values("mean_center_error_px")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered.method, ordered.mean_center_error_px, color="#4f46e5")
    axis.set_xlabel("Mean hidden localization error (px; lower is better)")
    axis.set_title("Camera compensation and relational localization")
    fig.tight_layout()
    fig.savefig(chart_dir / "localization_error.png", dpi=180)
    plt.close(fig)


def report(summary: pd.DataFrame, run_id: str, n_scenarios: int) -> str:
    return f"""# Relational OATM Extended Synthetic Study

Run ID: `{run_id}`. Scenarios: {n_scenarios}. Seed: 42.

## Question

Does explicit target--occluder memory, clearance reasoning, and causal camera
compensation improve the recovery--ghost tradeoff over ByteTrack and Selective OATM?

## Results

{summary.to_markdown(index=False, floatfmt=".3f")}

## Interpretation boundary

These deterministic synthetic scenarios validate mechanisms and ablations;
they do not establish nuScenes or general driving superiority. Natural,
controlled-visual, and negative real-image studies remain separate required evidence.

## Included stress cases

Short and long occlusion, moving and multiple occluders, abrupt camera pan,
ordinary miss, field-of-view exit, and failed expected reappearance.
"""


def main() -> None:
    started = time.perf_counter()
    config = yaml.safe_load((ROOT / "configs" / "relational.yaml").read_text())
    run_id = uuid.uuid4().hex[:12]
    scenario_list = scenarios()
    rows = [
        run_scenario(method, factory, scenario, run_id)
        for scenario in scenario_list
        for method, factory in method_factories(config).items()
    ]
    event_frame = pd.DataFrame(rows)
    summary = aggregate(event_frame)
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    event_frame.to_csv(results / "synthetic_event_metrics.csv", index=False)
    summary.to_csv(results / "synthetic_summary.csv", index=False)
    result_charts(summary, results / "charts")
    (results / "synthetic_report.md").write_text(report(summary, run_id, len(scenario_list)))
    metadata = {
        "run_id": run_id,
        "experiment_family": "synthetic_development",
        "seed": config["random_seed"],
        "scenario_names": [item.name for item in scenario_list],
        "methods": list(method_factories(config)),
        "config": config,
        "runtime_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "opencv": importlib.metadata.version("opencv-python"),
            "pandas": importlib.metadata.version("pandas"),
        },
    }
    (results / "synthetic_run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(summary.to_string(index=False))
    print(f"Wrote run {run_id}")


if __name__ == "__main__":
    main()

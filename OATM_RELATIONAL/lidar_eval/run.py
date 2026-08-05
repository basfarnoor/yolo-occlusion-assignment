"""Run the separated CAM_FRONT tracking and privileged offline evaluation stages."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lidar_eval.common import (
    ARTIFACTS_ROOT,
    LIDAR_EVAL_ROOT,
    PROJECT_ROOT,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    git_provenance,
    load_config,
    runtime_provenance,
    sha256,
)
from lidar_eval.detector import ensure_detector_cache
from lidar_eval.matching import evaluate_method, validate_ground_truth
from lidar_eval.metrics import stratified_metrics, summarize_method
from lidar_eval.reporting import build_report
from lidar_eval.tracking import run_camera_trackers


def _default_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = os.environ.get("SLURM_JOB_ID", "local")
    return f"{timestamp}_{suffix}"


def _prepare_output_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path

def assign_scene_splits(
    scene_tokens: list[str], development_fraction: float, seed: int
) -> dict[str, str]:
    """Assign whole scenes by a stable seeded hash rank, never by neighboring frames."""
    unique = sorted(set(scene_tokens))
    if len(unique) < 2 or not 0.0 < development_fraction < 1.0:
        raise ValueError("scene split needs at least two scenes and a fraction in (0, 1)")
    ranked = sorted(
        unique,
        key=lambda token: hashlib.sha256(f"{seed}:{token}".encode()).hexdigest(),
    )
    development_count = round(len(ranked) * development_fraction)
    development_count = min(max(development_count, 1), len(ranked) - 1)
    development = set(ranked[:development_count])
    return {
        token: "development" if token in development else "validation"
        for token in unique
    }



def _validate_preparation(
    frames: pd.DataFrame, ground_truth: pd.DataFrame, config: dict[str, Any]
) -> dict[str, Any]:
    if len(frames) != 2342:
        raise ValueError(f"expected 2342 nuScenes-mini CAM_FRONT frames, found {len(frames)}")
    if frames.scene_token.nunique() != 10:
        raise ValueError("expected exactly 10 nuScenes-mini scenes")
    keyframes = frames[frames.is_keyframe.astype(bool)].copy()
    if len(keyframes) != 404:
        raise ValueError(f"expected 404 CAM_FRONT keyframes, found {len(keyframes)}")
    scoped_truth = validate_ground_truth(ground_truth, set(keyframes.sample_data_token))
    if scoped_truth.empty:
        raise ValueError("no projected car/pedestrian annotations are available")
    split_config = config["scene_split"]
    if split_config.get("strategy") != "stable_sha256_rank":
        raise ValueError("scene_split.strategy must be stable_sha256_rank")
    scene_splits = assign_scene_splits(
        frames.scene_token.tolist(),
        float(split_config["development_fraction"]),
        int(config["random_seed"]),
    )
    return {
        "dataset_version": "v1.0-mini",
        "scenes": int(frames.scene_token.nunique()),
        "camera_frames": len(frames),
        "keyframes": len(keyframes),
        "gt_annotations": len(scoped_truth),
        "instances": int(scoped_truth.instance_token.nunique()),
        "development_scenes": sum(value == "development" for value in scene_splits.values()),
        "validation_scenes": sum(value == "validation" for value in scene_splits.values()),
        "development_annotations": int(
            scoped_truth.scene_token.map(scene_splits).eq("development").sum()
        ),
        "validation_annotations": int(
            scoped_truth.scene_token.map(scene_splits).eq("validation").sum()
        ),
        "scene_splits": scene_splits,
        "keyframe_table": keyframes,
        "ground_truth_table": scoped_truth,
    }


def run(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config_path = args.config.resolve()
    config = load_config(config_path)
    seed = int(config["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    run_id = args.run_id or _default_run_id()
    output_dir = _prepare_output_directory(
        args.output_dir or (LIDAR_EVAL_ROOT / "results" / run_id)
    )

    frames_path = ARTIFACTS_ROOT / "frame_index.parquet"
    ground_truth_path = ARTIFACTS_ROOT / "projected_ground_truth.parquet"
    for required in (frames_path, ground_truth_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing {required}; run scripts/prepare_nuscenes.py first")
    frames = pd.read_parquet(frames_path)

    # Stage 1: validate or generate camera detections. No GT has been loaded.
    detector_audit = ensure_detector_cache(
        config,
        frames,
        force=args.refresh_detector,
        device_override=args.device,
    )
    detections_path = ARTIFACTS_ROOT / "detections.parquet"
    detections = pd.read_parquet(detections_path)

    # Stage 2: causal tracking. This API cannot receive privileged annotations.
    tracker_outputs, tracker_runtimes = run_camera_trackers(
        frames=frames,
        detections=detections,
        methods=list(config["methods"]),
        run_id=run_id,
    )
    atomic_write_parquet(tracker_outputs, output_dir / "tracker_outputs.parquet")
    del detections

    # Stage 3: only now load projected annotations for offline scoring.
    ground_truth = pd.read_parquet(ground_truth_path)
    dataset = _validate_preparation(frames, ground_truth, config)
    keyframes = dataset.pop("keyframe_table")
    scoped_truth = dataset.pop("ground_truth_table")
    scene_splits = dataset["scene_splits"]
    keyframes["split"] = keyframes.scene_token.map(scene_splits)
    scoped_truth["split"] = scoped_truth.scene_token.map(scene_splits)
    thresholds = [float(item) for item in config["matching"]["sensitivity_iou_thresholds"]]
    primary_threshold = float(config["matching"]["primary_iou_threshold"])
    summaries: list[dict[str, Any]] = []
    primary_matches = []
    primary_false_positives = []
    strata = []

    for method_name in config["methods"]:
        method_outputs = tracker_outputs[tracker_outputs.method_name == method_name]
        for threshold in thresholds:
            matches, false_positives = evaluate_method(
                method_outputs,
                scoped_truth,
                keyframes,
                threshold,
            )
            matches["split"] = matches.scene_token.map(scene_splits)
            false_positives["split"] = false_positives.scene_token.map(scene_splits)
            for population in ("all", "development", "validation"):
                population_matches = (
                    matches if population == "all" else matches[matches.split == population]
                )
                population_false_positives = (
                    false_positives
                    if population == "all"
                    else false_positives[false_positives.split == population]
                )
                population_summary = summarize_method(
                    population_matches, population_false_positives, threshold
                )
                population_summary["population"] = population
                summaries.append(population_summary)
            if threshold == primary_threshold:
                primary_matches.append(matches)
                primary_false_positives.append(false_positives)
                for population in ("all", "development", "validation"):
                    population_matches = (
                        matches if population == "all" else matches[matches.split == population]
                    )
                    population_strata = stratified_metrics(population_matches, config["strata"])
                    population_strata["population"] = population
                    strata.append(population_strata)

    summary_frame = pd.DataFrame(summaries).sort_values(
        ["population", "iou_threshold", "method_name"]
    )
    match_frame = pd.concat(primary_matches, ignore_index=True)
    false_positive_frame = pd.concat(primary_false_positives, ignore_index=True)
    strata_frame = pd.concat(strata, ignore_index=True)
    atomic_write_parquet(match_frame, output_dir / "ground_truth_matches.parquet")
    atomic_write_parquet(false_positive_frame, output_dir / "unmatched_outputs.parquet")
    atomic_write_parquet(strata_frame, output_dir / "stratified_metrics.parquet")
    atomic_write_csv(summary_frame, output_dir / "summary.csv")
    atomic_write_csv(strata_frame, output_dir / "stratified_metrics.csv")

    report = build_report(run_id, summary_frame, strata_frame, dataset, primary_threshold)
    atomic_write_text(output_dir / "report.md", report)
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "scientific_boundary": {
            "online_input": "causal CAM_FRONT images/detections and tracker history only",
            "privileged_evaluation_only": (
                "nuScenes projected 3D annotations, calibration, visibility, instance tokens, "
                "LiDAR/radar point counts, and ego poses"
            ),
            "future_frames_used_online": False,
        },
        "config_path": str(config_path),
        "config": config,
        "dataset": dataset,
        "input_hashes": {
            "frame_index.parquet": sha256(frames_path),
            "detections.parquet": sha256(detections_path),
            "projected_ground_truth.parquet": sha256(ground_truth_path),
            "relational.yaml": sha256(PROJECT_ROOT / "configs" / "relational.yaml"),
            "uv.lock": sha256(PROJECT_ROOT / "uv.lock"),
        },
        "detector": detector_audit,
        "tracker_runtime_seconds": tracker_runtimes,
        "row_counts": {
            "tracker_outputs": len(tracker_outputs),
            "ground_truth_matches": len(match_frame),
            "unmatched_outputs": len(false_positive_frame),
            "stratified_metrics": len(strata_frame),
        },
        "runtime_seconds": time.perf_counter() - started,
        "git": git_provenance(),
        "runtime": runtime_provenance(),
        "command": sys.argv,
    }
    atomic_write_json(output_dir / "run_metadata.json", metadata)
    atomic_write_json(output_dir / "detector_cache_audit.json", detector_audit)
    print(report)
    print(f"results: {output_dir}", flush=True)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=LIDAR_EVAL_ROOT / "config.yaml")
    parser.add_argument("--run-id", help="Stable run label; defaults to UTC timestamp and Slurm job ID")
    parser.add_argument("--output-dir", type=Path, help="Must be absent or empty; never overwritten")
    parser.add_argument(
        "--refresh-detector",
        action="store_true",
        help="Regenerate the frozen camera detector cache (uses the allocated GPU)",
    )
    parser.add_argument("--device", help="Detector device override: auto, cpu, 0, cuda, or cuda:0")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

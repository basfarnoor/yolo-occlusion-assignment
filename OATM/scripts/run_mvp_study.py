"""Task 11: the first complete OATM MVP study. Compares YOLO-only, static
memory (also serving as the "fixed-window memory" baseline -- its frozen
`track_buffer` IS a fixed window, so a sixth, redundant method was not
built), SORT, ByteTrack, and OATM MVP on IDENTICAL inputs across three
separate experiment families (natural / controlled_visual /
detector_intervention), never mixed.

Natural events need no special rerun: they occur inside the ordinary,
unmodified per-scene continuous run every method already gets (the same run
used for the global precision/recall/ghost-rate sanity metrics). Controlled
events (both families) require a fresh, event-scoped rerun of that ONE
event's scene with only that target's detections modified for the window --
every other object's detections, and every other event, are untouched.

Reproduction: `.venv/Scripts/python scripts/run_mvp_study.py` from OATM/
(assumes Tasks 5-10 have already been run at least once, so
`artifacts/detections.parquet`, `artifacts/projected_ground_truth.parquet`,
`artifacts/controlled_visual_frames/`, and `.detection_cache.json` exist).
"""
from __future__ import annotations

import copy
import csv
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_controlled_events as bce  # noqa: E402

from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.detection.cache import DetectionCache, cache_key, sha256_of_file  # noqa: E402
from oatm.evaluation.event_metrics import (  # noqa: E402
    EventMetricsInputs,
    compute_event_metrics,
    compute_yolo_only_event_metrics,
)
from oatm.evaluation.global_metrics import compute_ghost_rate, compute_precision_recall  # noqa: E402
from oatm.evaluation.ground_truth import index_by_frame, load_ground_truth  # noqa: E402
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402
from oatm.tracking.oatm_adapter import OATMTracker  # noqa: E402
from oatm.tracking.sort_adapter import SortAdapter  # noqa: E402
from oatm.tracking.static_memory import StaticMemoryTracker, _StaticTrack  # noqa: E402
from oatm.tracking.yolo_only import run_yolo_only_frame  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
METHODS = ("yolo_only", "static_memory", "sort", "bytetrack", "oatm_mvp")
RECOVERY_HORIZON_FRAMES = 10
EVAL_CLASS_TO_DETECTOR_CLASS = {"car": "car", "pedestrian": "person"}


def load_tracker_config() -> dict:
    with open(OATM_ROOT / "configs" / "tracker.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_methods_over_scene(
    scene_token: str, frames: list[dict], detections_by_frame: dict[str, list[dict]],
    tcfg: dict, run_id: str,
) -> dict[str, list[dict]]:
    """Fresh instance of every method, run continuously over `frames` (one
    scene's worth, in order), fed `detections_by_frame`. Returns
    method_name -> flat list of output rows (as plain dicts)."""
    KalmanBoxTracker.reset_id_counter()
    _StaticTrack.reset_id_counter()
    static_tracker = StaticMemoryTracker(**tcfg["static_memory"])
    sort_tracker = SortAdapter(**tcfg["sort"])
    byte_tracker = ByteTrackAdapter(**tcfg["bytetrack"])
    oatm_tracker = OATMTracker(**tcfg["oatm_mvp"])

    rows_by_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    for f in frames:
        raw_dets = detections_by_frame.get(f["sample_data_token"], [])
        ts = f["timestamp_us"] / 1_000_000.0
        common = dict(scene_token=scene_token, sample_data_token=f["sample_data_token"])

        rows_by_method["yolo_only"].extend(o.model_dump() for o in run_yolo_only_frame(
            raw_dets, tcfg["shared"]["high_score_threshold"], frame_index=f["frame_index"],
            method_name="yolo_only", run_id=run_id, **common))
        rows_by_method["static_memory"].extend(o.model_dump() for o in static_tracker.update(
            raw_dets, timestamp=ts, method_name="static_memory", run_id=run_id, **common))
        rows_by_method["sort"].extend(o.model_dump() for o in sort_tracker.update(
            raw_dets, timestamp=ts, method_name="sort", run_id=run_id, **common))
        rows_by_method["bytetrack"].extend(o.model_dump() for o in byte_tracker.update(
            raw_dets, timestamp=ts, method_name="bytetrack", run_id=run_id, **common))
        rows_by_method["oatm_mvp"].extend(o.model_dump() for o in oatm_tracker.update(
            raw_dets, timestamp=ts, method_name="oatm_mvp", run_id=run_id, **common))
    return rows_by_method


def index_rows_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame_index"]].append(r)
    return by_frame


def load_natural_events(results_dir: Path) -> list[dict]:
    with open(results_dir / "natural_event_manifest.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r["review_status"] == "accepted"]


def load_controlled_events(results_dir: Path) -> list[dict]:
    with open(results_dir / "controlled_event_manifest.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_controlled_visual_detections(
    event_id: str, window_frame_indices: list[int], frame_to_sdt: dict[int, str],
    config, cache: DetectionCache, weights_hash: str,
) -> dict[str, list[dict]]:
    """Reads the ALREADY-CACHED redetections (Task 7) for this event's masked
    frames -- never re-runs the detector, so results stay pinned to exactly
    what was committed. Raises if a frame's cache entry is missing (a silent
    re-inference here would break reproducibility, not just fail loudly)."""
    visual_dir = config.artifacts_dir / "controlled_visual_frames"
    pkg_versions = {"model": bce.MODEL_NAME}
    replacements: dict[str, list[dict]] = {}
    for frame_idx in window_frame_indices:
        image_path = visual_dir / f"{event_id}_f{frame_idx}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"missing masked frame for {event_id} at frame {frame_idx}: {image_path}")
        img_hash = sha256_of_file(image_path)
        key = cache_key(img_hash, bce.MODEL_NAME, weights_hash, bce.IMGSZ, bce.CONFIDENCE_FLOOR, pkg_versions)
        cached = cache.get(key)
        if cached is None:
            raise KeyError(f"no cached detection for {event_id} frame {frame_idx} (key={key[:12]}...)")
        sdt = frame_to_sdt[frame_idx]
        replacements[sdt] = cached["detections"]
    return replacements


def evaluate_event(
    event_id: str, method_name: str, target_class: str, reference_box: tuple,
    pre_frame_index: int, hidden_frame_indices: list[int], max_frame_index: int,
    rows_by_frame: dict[int, list[dict]], gt_box_by_frame: dict[int, tuple],
) -> dict:
    recovery_frames = list(range(
        max(hidden_frame_indices) + 1, min(max(hidden_frame_indices) + 1 + RECOVERY_HORIZON_FRAMES,
                                            max_frame_index + 1),
    ))
    inputs = EventMetricsInputs(
        event_id=event_id, method_name=method_name, reference_box=reference_box,
        reference_class=target_class, pre_frame_index=pre_frame_index,
        hidden_frame_indices=hidden_frame_indices, recovery_search_frame_indices=recovery_frames,
        outputs_by_frame=rows_by_frame, gt_box_by_frame=gt_box_by_frame,
    )
    metrics_fn = compute_yolo_only_event_metrics if method_name == "yolo_only" else compute_event_metrics
    result = metrics_fn(inputs)
    row = vars(result).copy()
    row["event_id"] = event_id
    return row


def main() -> None:
    t_start = time.time()
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    tcfg = load_tracker_config()
    run_id = uuid.uuid4().hex[:12]

    print("Loading cached detections, frame index, and ground truth...")
    detections_df = pd.read_parquet(config.artifacts_dir / "detections.parquet")
    frame_index_df = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")
    detections_by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in detections_df.to_dict("records"):
        detections_by_frame[row["sample_data_token"]].append({
            "class": row["detected_class"], "confidence": row["confidence"],
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        })
    frames_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in frame_index_df.sort_values(["scene_token", "frame_index"]).to_dict("records"):
        frames_by_scene[row["scene_token"]].append(row)
    frame_to_sdt_by_scene = {
        scene: {f["frame_index"]: f["sample_data_token"] for f in frames}
        for scene, frames in frames_by_scene.items()
    }
    max_frame_index_by_scene = {
        scene: max(f["frame_index"] for f in frames) for scene, frames in frames_by_scene.items()
    }

    gt = load_ground_truth(config.artifacts_dir)
    gt_by_frame = index_by_frame(gt)
    gt_by_frame_instance: dict[tuple[str, str], tuple] = {
        (r["sample_data_token"], r["instance_token"]): (r["x1"], r["y1"], r["x2"], r["y2"])
        for r in gt.to_dict("records")
    }
    # nuScenes only has 3D annotations at keyframes (2Hz); the far more
    # numerous sweep frames (12Hz) have no ground truth at all. Precision/
    # recall must be scored only where ground truth genuinely exists.
    keyframe_rows = frame_index_df[frame_index_df["is_keyframe"]].to_dict("records")
    keyframe_sdts = {row["sample_data_token"] for row in keyframe_rows}

    # ---------- Pass 1: one full, unmodified continuous run per scene, all 5 methods ----------
    # Serves natural-event evaluation (no rerun needed -- these events occur
    # inside this exact run) AND the global sanity metrics (precision/recall,
    # ghost rate) that check ordinary, non-occlusion tracking didn't regress.
    print(f"Running all {len(METHODS)} methods continuously over all {len(frames_by_scene)} scenes...")
    full_rows_by_method: dict[str, list[dict]] = {m: [] for m in METHODS}
    full_rows_by_scene_method: dict[tuple[str, str], dict[int, list[dict]]] = {}
    runtime_s = {m: 0.0 for m in METHODS}
    for scene_token, frames in frames_by_scene.items():
        t0 = time.perf_counter()
        rows_by_method = run_methods_over_scene(scene_token, frames, detections_by_frame, tcfg, run_id)
        elapsed = time.perf_counter() - t0
        for m in METHODS:
            full_rows_by_method[m].extend(rows_by_method[m])
            full_rows_by_scene_method[(scene_token, m)] = index_rows_by_frame(rows_by_method[m])
            runtime_s[m] += elapsed / len(METHODS)  # per-method share of this scene's wall time

    full_df = pd.DataFrame([r for rows in full_rows_by_method.values() for r in rows])
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(config.artifacts_dir / "mvp_full_outputs.parquet", index=False)
    print(f"Wrote mvp_full_outputs.parquet ({len(full_df)} rows).")

    global_metrics = {}
    for m in METHODS:
        global_metrics[m] = {
            "precision_recall": compute_precision_recall(
                full_rows_by_method[m], gt_by_frame, keyframe_sample_data_tokens=keyframe_sdts,
            ),
            "ghost": compute_ghost_rate(full_rows_by_method[m], gt_by_frame) if m != "yolo_only" else None,
            "runtime_s_10_scenes": runtime_s[m],
        }

    # ---------- Natural family: 6 accepted events, no rerun ----------
    print("Evaluating natural-occlusion events...")
    natural_events = load_natural_events(config.results_dir)
    natural_metrics = []
    for ev in natural_events:
        scene_token = ev["scene_token"]
        instance_token = ev["instance_token"]
        pre_idx = int(ev["pre_frame_index"])
        start_idx = int(ev["start_frame_index"])
        end_idx = int(ev["end_frame_index"])
        hidden_frames = list(range(start_idx, end_idx + 1))
        frame_to_sdt = frame_to_sdt_by_scene[scene_token]
        pre_sdt = frame_to_sdt[pre_idx]
        reference_box = gt_by_frame_instance.get((pre_sdt, instance_token))
        detector_class = EVAL_CLASS_TO_DETECTOR_CLASS.get(ev["evaluation_class"])
        last_frame = min(end_idx + 1 + RECOVERY_HORIZON_FRAMES, max_frame_index_by_scene[scene_token])
        event_gt_by_frame = {}
        for fi in range(pre_idx, last_frame + 1):
            box = gt_by_frame_instance.get((frame_to_sdt.get(fi), instance_token))
            if box is not None:
                event_gt_by_frame[fi] = box
        if reference_box is None or detector_class is None:
            for m in METHODS:
                natural_metrics.append({
                    "event_id": ev["event_id"], "method_name": m, "target_linked": False,
                    "n_hidden_frames": len(hidden_frames), "n_hidden_frames_alive": 0,
                    "hidden_frame_coverage": None, "fully_bridged": None, "n_hidden_frames_with_gt": 0,
                    "mean_center_error_px": None, "mean_iou": None,
                    "recovery_status": "n/a", "recovery_latency_frames": None,
                })
            continue
        for m in METHODS:
            rows_by_frame = full_rows_by_scene_method[(scene_token, m)]
            natural_metrics.append(evaluate_event(
                ev["event_id"], m, detector_class, reference_box, pre_idx, hidden_frames,
                max_frame_index_by_scene[scene_token], rows_by_frame, event_gt_by_frame,
            ))

    # ---------- Controlled families: 48 events, each gets its own event-scoped rerun ----------
    print("Rebuilding controlled-experiment target trajectories...")
    targets = bce.build_natural_targets(dict(detections_by_frame), dict(frames_by_scene), tcfg)
    targets_by_key = {(t.scene_token, t.track_id): t for t in targets}

    weights_path = repo_root / bce.MODEL_NAME
    weights_hash = sha256_of_file(weights_path)
    cache = DetectionCache(config.artifacts_dir / ".detection_cache.json")

    controlled_events = load_controlled_events(config.results_dir)
    controlled_metrics = []
    t0 = time.time()
    controlled_runtime_s = {m: 0.0 for m in METHODS}
    n_controlled_reruns = 0
    for ev in controlled_events:
        key = (ev["scene_token"], int(ev["track_id"]))
        target = targets_by_key.get(key)
        if target is None:
            print(f"  SKIP {ev['event_id']}: target {key} not reproducible from cached detections")
            continue
        window_frames = sorted(int(x) for x in ev["window_frame_indices"].split(";"))
        pos_start = target.frame_numbers.index(min(window_frames))
        pre_idx = target.frame_numbers[pos_start - 1]
        reference_box = target.raw_boxes[pos_start - 1]
        event_gt_by_frame = dict(zip(target.frame_numbers, target.raw_boxes))

        scene_token = ev["scene_token"]
        frames = frames_by_scene[scene_token]
        scene_sdts = {f["sample_data_token"] for f in frames}
        scene_detections = {sdt: detections_by_frame.get(sdt, []) for sdt in scene_sdts}

        if ev["event_source"] == "detector_intervention":
            modified = bce.build_detector_intervention_detections(
                scene_detections, target, set(window_frames), float(ev["coverage"]), bce.DEMOTED_CONFIDENCE,
            )
        else:
            frame_to_sdt = frame_to_sdt_by_scene[scene_token]
            replacements = build_controlled_visual_detections(
                ev["event_id"], window_frames, frame_to_sdt, config, cache, weights_hash,
            )
            modified = copy.deepcopy(scene_detections)
            modified.update(replacements)

        t_event0 = time.perf_counter()
        rows_by_method = run_methods_over_scene(scene_token, frames, modified, tcfg, run_id)
        elapsed = time.perf_counter() - t_event0
        n_controlled_reruns += 1
        for m in METHODS:
            controlled_runtime_s[m] += elapsed / len(METHODS)
            rows_by_frame = index_rows_by_frame(rows_by_method[m])
            metrics_row = evaluate_event(
                ev["event_id"], m, target.class_name, reference_box, pre_idx, window_frames,
                max_frame_index_by_scene[scene_token], rows_by_frame, event_gt_by_frame,
            )
            metrics_row["event_source"] = ev["event_source"]
            metrics_row["duration"] = ev["duration"]
            metrics_row["coverage"] = ev["coverage"]
            controlled_metrics.append(metrics_row)
    controlled_elapsed = time.time() - t0
    print(f"Controlled families: {n_controlled_reruns} event reruns in {controlled_elapsed:.1f}s.")

    for row in natural_metrics:
        row["event_source"] = "natural"
        row.setdefault("duration", None)
        row.setdefault("coverage", None)

    all_event_metrics = natural_metrics + controlled_metrics
    event_metrics_df = pd.DataFrame(all_event_metrics)
    event_metrics_path = config.results_dir / "mvp_event_metrics.csv"
    event_metrics_df.to_csv(event_metrics_path, index=False)
    print(f"Wrote {event_metrics_path} ({len(event_metrics_df)} rows).")

    total_elapsed = time.time() - t_start
    metadata = {
        "run_id": run_id,
        "methods": list(METHODS),
        "n_scenes": len(frames_by_scene),
        "n_natural_events": len(natural_events),
        "n_controlled_events": len(controlled_events),
        "n_controlled_reruns_completed": n_controlled_reruns,
        "n_unique_frames_all_scenes": sum(len(f) for f in frames_by_scene.values()),
        "n_keyframes_with_ground_truth": len(keyframe_sdts),
        "n_full_output_rows": len(full_df),
        "n_event_metric_rows": len(event_metrics_df),
        "tracker_config": tcfg,
        "total_elapsed_s": total_elapsed,
        "global_metrics": global_metrics,
    }
    with open(config.results_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Wrote run_metadata.json. Total elapsed: {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()

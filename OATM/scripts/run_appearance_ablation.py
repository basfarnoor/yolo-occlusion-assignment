"""Task 12: ablation comparing motion_only (Task 11's OATM MVP, unchanged),
appearance_only (reconnect purely by cosine similarity, ignoring location),
and dual (appearance + location-consistency gate) on the SAME 48 controlled
events (24 detector_intervention + 24 controlled_visual) Task 11 already
used -- identical inputs, per this project's own methodology throughout.

Embeddings are only computed for detections within a bounded window around
each event (from a lead-in margin before the pre-occlusion reference frame
through the recovery-search horizon) -- reconnection can only ever fire
inside that window anyway (see `run_mvp_study.py`'s RECOVERY_HORIZON_FRAMES),
so embedding the rest of each scene would cost real time for zero effect on
any measured event outcome.

Reproduction: `.venv/Scripts/python scripts/run_appearance_ablation.py` from
OATM/ (assumes Tasks 5-11 have already been run at least once).
"""
from __future__ import annotations

import copy
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_controlled_events as bce  # noqa: E402
from run_mvp_study import (  # noqa: E402
    build_controlled_visual_detections,
    evaluate_event,
    load_controlled_events,
    load_tracker_config,
)

from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.detection.cache import DetectionCache, sha256_of_file  # noqa: E402
from oatm.memory.embedder import AppearanceEmbedder  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402
from oatm.tracking.oatm_appearance_adapter import OATMAppearanceTracker  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
MODES = ("motion_only", "appearance_only", "dual")
EMBED_LEAD_IN_FRAMES = 15
RECOVERY_HORIZON_FRAMES = 10


def resolve_image_path(
    event_source: str, event_id: str, frame_idx: int, window_frames: set[int],
    image_path_by_frame: dict[int, str], config,
) -> Path:
    if event_source == "controlled_visual" and frame_idx in window_frames:
        return config.artifacts_dir / "controlled_visual_frames" / f"{event_id}_f{frame_idx}.jpg"
    return config.data_root / image_path_by_frame[frame_idx]


def embed_frame_detections(
    detections_by_frame: dict[str, list[dict]], frames: list[dict], frame_indices: set[int],
    high_score_threshold: float, embedder: AppearanceEmbedder,
    event_source: str, event_id: str, window_frames: set[int],
    image_path_by_frame: dict[int, str], config, embedding_cache: dict,
) -> None:
    """Mutates `detections_by_frame` in place, adding an `"embedding"` key to
    every high-confidence detection at the given frame indices."""
    for f in frames:
        if f["frame_index"] not in frame_indices:
            continue
        dets = detections_by_frame.get(f["sample_data_token"], [])
        high_dets = [d for d in dets if d.get("confidence", 0.0) >= high_score_threshold]
        if not high_dets:
            continue
        image_path = resolve_image_path(
            event_source, event_id, f["frame_index"], window_frames, image_path_by_frame, config,
        )
        image = None
        for det in high_dets:
            key = (str(image_path), round(det["x1"], 1), round(det["y1"], 1),
                   round(det["x2"], 1), round(det["y2"], 1))
            if key in embedding_cache:
                det["embedding"] = embedding_cache[key]
                continue
            if image is None:
                if not image_path.is_file():
                    continue
                image = Image.open(image_path).convert("RGB")
            x1, y1 = max(0.0, det["x1"]), max(0.0, det["y1"])
            x2, y2 = min(float(image.width), det["x2"]), min(float(image.height), det["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image.crop((x1, y1, x2, y2))
            embedding = embedder.embed_crop(crop)
            embedding_cache[key] = embedding
            det["embedding"] = embedding


def run_one_mode_over_scene(scene_token, frames, detections_by_frame, tcfg, run_id, mode) -> list[dict]:
    KalmanBoxTracker.reset_id_counter()
    tracker_cfg = dict(tcfg["oatm_mvp"])
    tracker = OATMAppearanceTracker(appearance_mode=mode, **tracker_cfg)
    rows: list[dict] = []
    for f in frames:
        raw_dets = detections_by_frame.get(f["sample_data_token"], [])
        ts = f["timestamp_us"] / 1_000_000.0
        outputs = tracker.update(
            raw_dets, timestamp=ts, scene_token=scene_token, sample_data_token=f["sample_data_token"],
            method_name=f"oatm_{mode}", run_id=run_id,
        )
        rows.extend(o.model_dump() for o in outputs)
    return rows


def index_rows_by_frame(rows: list[dict]) -> dict[int, list[dict]]:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_frame[r["frame_index"]].append(r)
    return by_frame


def main() -> None:
    t_start = time.time()
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    tcfg = load_tracker_config()
    run_id = uuid.uuid4().hex[:12]

    print("Loading cached detections and frame index...")
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
    image_path_by_frame_by_scene = {
        scene: {f["frame_index"]: f["image_path"] for f in frames}
        for scene, frames in frames_by_scene.items()
    }
    max_frame_index_by_scene = {
        scene: max(f["frame_index"] for f in frames) for scene, frames in frames_by_scene.items()
    }

    print("Rebuilding controlled-experiment target trajectories...")
    targets = bce.build_natural_targets(dict(detections_by_frame), dict(frames_by_scene), tcfg)
    targets_by_key = {(t.scene_token, t.track_id): t for t in targets}
    controlled_events = load_controlled_events(config.results_dir)

    weights_path = repo_root / bce.MODEL_NAME
    weights_hash = sha256_of_file(weights_path)
    cache = DetectionCache(config.artifacts_dir / ".detection_cache.json")

    print("Loading frozen appearance embedder (MobileNetV3-Small, ImageNet weights)...")
    embedder = AppearanceEmbedder()
    embedding_cache: dict = {}

    all_metrics = []
    t0 = time.time()
    n_events_run = 0
    for ev in controlled_events:
        key = (ev["scene_token"], int(ev["track_id"]))
        target = targets_by_key.get(key)
        if target is None:
            print(f"  SKIP {ev['event_id']}: target {key} not reproducible")
            continue
        scene_token = ev["scene_token"]
        frames = frames_by_scene[scene_token]
        window_frames = sorted(int(x) for x in ev["window_frame_indices"].split(";"))
        pos_start = target.frame_numbers.index(min(window_frames))
        pre_idx = target.frame_numbers[pos_start - 1]
        reference_box = target.raw_boxes[pos_start - 1]
        event_gt_by_frame = dict(zip(target.frame_numbers, target.raw_boxes))
        max_frame_index = max_frame_index_by_scene[scene_token]

        embed_frames = set(range(
            max(0, pre_idx - EMBED_LEAD_IN_FRAMES),
            min(max_frame_index, max(window_frames) + RECOVERY_HORIZON_FRAMES) + 1,
        ))

        scene_sdts = {f["sample_data_token"] for f in frames}
        scene_detections_real = {sdt: copy.deepcopy(detections_by_frame.get(sdt, [])) for sdt in scene_sdts}

        if ev["event_source"] == "detector_intervention":
            modified = bce.build_detector_intervention_detections(
                scene_detections_real, target, set(window_frames),
                float(ev["coverage"]), bce.DEMOTED_CONFIDENCE,
            )
        else:
            frame_to_sdt = frame_to_sdt_by_scene[scene_token]
            replacements = build_controlled_visual_detections(
                ev["event_id"], window_frames, frame_to_sdt, config, cache, weights_hash,
            )
            modified = copy.deepcopy(scene_detections_real)
            modified.update(replacements)

        embed_frame_detections(
            modified, frames, embed_frames, tcfg["shared"]["high_score_threshold"], embedder,
            ev["event_source"], ev["event_id"], set(window_frames),
            image_path_by_frame_by_scene[scene_token], config, embedding_cache,
        )

        for mode in MODES:
            rows = run_one_mode_over_scene(scene_token, frames, modified, tcfg, run_id, mode)
            rows_by_frame = index_rows_by_frame(rows)
            metrics_row = evaluate_event(
                ev["event_id"], f"oatm_{mode}", target.class_name, reference_box, pre_idx, window_frames,
                max_frame_index, rows_by_frame, event_gt_by_frame,
            )
            metrics_row["event_source"] = ev["event_source"]
            metrics_row["ablation_mode"] = mode
            all_metrics.append(metrics_row)
        n_events_run += 1
        if n_events_run % 10 == 0:
            print(f"  ...{n_events_run}/{len(controlled_events)} events done")

    elapsed = time.time() - t0
    print(f"Ran {n_events_run} events x {len(MODES)} modes in {elapsed:.1f}s "
          f"({len(embedding_cache)} unique embeddings computed).")

    df = pd.DataFrame(all_metrics)
    out_path = config.results_dir / "appearance_ablation_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows).")

    summary = {}
    for mode in MODES:
        sub = df[df.ablation_mode == mode]
        recovery_counts = sub["recovery_status"].value_counts().to_dict()
        summary[mode] = {
            "n_events": int(sub["event_id"].nunique()),
            "n_linked": int(sub["target_linked"].sum()),
            "mean_hidden_frame_coverage": sub["hidden_frame_coverage"].mean(),
            "fully_bridged_rate": sub["fully_bridged"].mean(),
            "mean_center_error_px": sub["mean_center_error_px"].mean(),
            "mean_iou": sub["mean_iou"].mean(),
            "recovery_status_counts": recovery_counts,
        }

    metadata = {
        "run_id": run_id, "n_events_run": n_events_run, "modes": list(MODES),
        "embed_lead_in_frames": EMBED_LEAD_IN_FRAMES, "n_unique_embeddings": len(embedding_cache),
        "total_elapsed_s": time.time() - t_start, "summary": summary,
    }
    with open(config.results_dir / "appearance_ablation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print("Wrote appearance_ablation_metadata.json")
    for mode in MODES:
        s = summary[mode]
        print(f"{mode}: linked {s['n_linked']}/{s['n_events'] * 1} rows, "
              f"coverage={s['mean_hidden_frame_coverage']:.3f}, "
              f"bridged={s['fully_bridged_rate']:.3f}, "
              f"recovery={s['recovery_status_counts']}")


if __name__ == "__main__":
    main()

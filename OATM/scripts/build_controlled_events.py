"""Phase 5 (Task 7): builds the two controlled-occlusion event families.
Selects real target tracks (via the ByteTrack baseline over full,
un-gapped detections), places deterministic windows, and for each:

- `detector_intervention`: demotes (coverage=0.5) or removes (coverage=1.0)
  the target's own detection row for the window -- pixels untouched.
- `controlled_visual`: paints a seeded mask over the target's box on a LOCAL
  COPY of the image, then reruns the frozen detector on that copy.

Writes the committed `results/controlled_event_manifest.csv` and
`results/controlled_protocol.md`. Modified images and the detector's rerun
output stay local-only (regenerable from the manifest + a fixed seed).

Reproduction: `.venv/Scripts/python scripts/build_controlled_events.py`.
"""
from __future__ import annotations

import hashlib
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.dataset.controlled_occlusion import (  # noqa: E402
    CONTROLLED_VISUAL,
    DETECTOR_INTERVENTION,
    NaturalTarget,
    apply_seeded_mask,
    build_controlled_windows,
    select_eligible_targets,
)
from oatm.detection.cache import DetectionCache, cache_key, sha256_of_file  # noqa: E402
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "yolo26n.pt"
CONFIDENCE_FLOOR = 0.05
IMGSZ = 640
DEVICE = "cpu"
SEED = 42
MIN_TRACK_LENGTH = 12
MIN_CONFIDENCE = 0.5
MAX_TARGETS = 6
DEMOTED_CONFIDENCE = 0.2
DURATIONS = [2, 5]
COVERAGES = [0.5, 1.0]


def build_natural_targets(detections_by_frame: dict, frames_by_scene: dict, cfg: dict) -> list[NaturalTarget]:
    targets_by_key: dict[tuple[str, int], NaturalTarget] = {}
    for scene_token, frames in frames_by_scene.items():
        KalmanBoxTracker.reset_id_counter()
        tracker = ByteTrackAdapter(**cfg["bytetrack"])
        for f in frames:
            dets = detections_by_frame.get(f["sample_data_token"], [])
            outputs = tracker.update(dets, timestamp=f["timestamp_us"] / 1e6, scene_token=scene_token)
            for o in outputs:
                if o.evidence_source not in ("strong_detection", "weak_detection"):
                    continue
                if o.raw_detection_x1 is None:
                    continue
                key = (scene_token, o.track_id)
                if key not in targets_by_key:
                    def _is_match(d):
                        return (abs(d["x1"] - o.raw_detection_x1) < 1e-6
                                and abs(d["y1"] - o.raw_detection_y1) < 1e-6)
                    class_name = next((d["class"] for d in dets if _is_match(d)), "unknown")
                    targets_by_key[key] = NaturalTarget(scene_token=scene_token, track_id=o.track_id,
                                                          class_name=class_name)
                nt = targets_by_key[key]
                nt.frame_numbers.append(f["frame_index"])
                nt.sample_data_tokens.append(f["sample_data_token"])
                nt.image_paths.append(f["image_path"])
                nt.raw_boxes.append((
                    o.raw_detection_x1, o.raw_detection_y1, o.raw_detection_x2, o.raw_detection_y2,
                ))
                nt.raw_confidences.append(o.detector_confidence or 0.0)
    return list(targets_by_key.values())


def build_detector_intervention_detections(
    detections_by_frame: dict, target: NaturalTarget, window_frame_indices: set[int],
    coverage: float, demoted_confidence: float,
) -> dict:
    """Returns a modified copy of detections_by_frame -- target's own row is
    demoted (coverage<1.0) or removed (coverage>=1.0) at exactly the window
    frames, nothing else touched."""
    import copy
    modified = {k: copy.deepcopy(v) for k, v in detections_by_frame.items()}
    box_by_frame_idx = dict(zip(target.frame_numbers, target.raw_boxes))

    for i, frame_idx in enumerate(target.frame_numbers):
        if frame_idx not in window_frame_indices:
            continue
        sd_token = target.sample_data_tokens[i]
        target_box = box_by_frame_idx[frame_idx]
        frame_dets = modified.get(sd_token, [])
        target_det_idx = None
        for j, d in enumerate(frame_dets):
            if (abs(d["x1"] - target_box[0]) < 1e-3 and abs(d["y1"] - target_box[1]) < 1e-3
                    and abs(d["x2"] - target_box[2]) < 1e-3 and abs(d["y2"] - target_box[3]) < 1e-3):
                target_det_idx = j
                break
        if target_det_idx is None:
            continue
        if coverage >= 1.0:
            frame_dets.pop(target_det_idx)
        else:
            frame_dets[target_det_idx]["confidence"] = demoted_confidence
    return modified


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    import yaml
    with open(OATM_ROOT / "configs" / "tracker.yaml", encoding="utf-8") as f:
        tcfg = yaml.safe_load(f)

    detections = pd.read_parquet(config.artifacts_dir / "detections.parquet")
    frame_index = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")

    detections_by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in detections.to_dict("records"):
        detections_by_frame[row["sample_data_token"]].append({
            "class": row["detected_class"], "confidence": row["confidence"],
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        })

    frames_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in frame_index.sort_values(["scene_token", "frame_index"]).to_dict("records"):
        frames_by_scene[row["scene_token"]].append(row)

    print("Building natural targets from cached detections + ByteTrack linking...")
    targets = build_natural_targets(dict(detections_by_frame), dict(frames_by_scene), tcfg)
    selected, selection_log = select_eligible_targets(
        targets, MIN_TRACK_LENGTH, MIN_CONFIDENCE, MAX_TARGETS, SEED
    )
    print(f"Selected {len(selected)} targets.")

    # --- Family A: detector intervention (cheap, no image work) ---
    intervention_events = []
    for target in selected:
        for w in build_controlled_windows(target, DURATIONS, COVERAGES):
            window_frame_indices = {target.frame_numbers[i] for i in w["window_indices"]}
            event_id = (f"{target.scene_token[:8]}_track{target.track_id}_"
                       f"{DETECTOR_INTERVENTION}_d{w['duration']}_c{w['coverage']}")
            intervention_events.append({
                "event_id": event_id, "event_source": DETECTOR_INTERVENTION,
                "scene_token": target.scene_token, "track_id": target.track_id,
                "class_name": target.class_name, "duration": w["duration"], "coverage": w["coverage"],
                "window_frame_indices": ";".join(str(x) for x in sorted(window_frame_indices)),
                "seed": SEED, "mask_box": "", "source_image_path": "", "cache_key": "",
            })

    # --- Family B: controlled visual (masks pixels, reruns the detector) ---
    weights_path = repo_root / MODEL_NAME
    model = YOLO(str(weights_path))
    weights_hash = sha256_of_file(weights_path)
    cache = DetectionCache(config.artifacts_dir / ".detection_cache.json")
    pkg_versions = {"model": MODEL_NAME}

    visual_dir = config.artifacts_dir / "controlled_visual_frames"
    visual_dir.mkdir(parents=True, exist_ok=True)
    visual_events = []
    t0 = time.time()

    for target in selected:
        box_by_frame_idx = dict(zip(target.frame_numbers, target.raw_boxes))
        path_by_frame_idx = dict(zip(target.frame_numbers, target.image_paths))
        for w in build_controlled_windows(target, DURATIONS, COVERAGES):
            window_frame_indices = [target.frame_numbers[i] for i in w["window_indices"]]
            event_id = (f"{target.scene_token[:8]}_track{target.track_id}_"
                       f"{CONTROLLED_VISUAL}_d{w['duration']}_c{w['coverage']}")
            # Python's built-in hash() is randomized per-process (PYTHONHASHSEED) --
            # using it here would make the mask non-reproducible across runs.
            # A deterministic digest keeps every event's mask exactly recreatable.
            digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()
            event_seed = SEED + int(digest[:8], 16) % 10_000

            per_frame_records = []
            for frame_idx in window_frame_indices:
                rel_path = path_by_frame_idx[frame_idx]
                target_box = box_by_frame_idx[frame_idx]
                original_image = Image.open(config.data_root / rel_path).convert("RGB")
                masked_image = original_image.copy()  # NEVER write to the original
                masked_image, mask_box = apply_seeded_mask(
                    masked_image, target_box, w["coverage"], event_seed
                )

                out_path = visual_dir / f"{event_id}_f{frame_idx}.jpg"
                masked_image.save(out_path, quality=90)

                img_hash = sha256_of_file(out_path)
                key = cache_key(img_hash, MODEL_NAME, weights_hash, IMGSZ, CONFIDENCE_FLOOR, pkg_versions)
                cached = cache.get(key)
                if cached is None:
                    preds = model.predict(source=str(out_path), imgsz=IMGSZ, conf=CONFIDENCE_FLOOR,
                                            device=DEVICE, verbose=False)
                    result = preds[0]
                    dets = []
                    if result.boxes is not None:
                        for box in result.boxes:
                            cls_id = int(box.cls.item())
                            dets.append({
                                "class": result.names.get(cls_id, str(cls_id)),
                                "confidence": float(box.conf.item()),
                                "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
                                "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3]),
                            })
                    cache.set(key, dets, 0.0)
                else:
                    dets = cached["detections"]

                modified_rel_path = str(out_path.relative_to(config.artifacts_dir))
                per_frame_records.append({
                    "event_id": event_id, "frame_index": frame_idx,
                    "source_image_path": rel_path, "modified_image_path": modified_rel_path,
                    "mask_box": str([round(v, 1) for v in mask_box]), "cache_key": key,
                    "n_redetections": len(dets),
                })

            visual_events.append({
                "event_id": event_id, "event_source": CONTROLLED_VISUAL,
                "scene_token": target.scene_token, "track_id": target.track_id,
                "class_name": target.class_name, "duration": w["duration"], "coverage": w["coverage"],
                "window_frame_indices": ";".join(str(x) for x in window_frame_indices),
                "seed": event_seed, "mask_box": per_frame_records[0]["mask_box"] if per_frame_records else "",
                "source_image_path": per_frame_records[0]["source_image_path"] if per_frame_records else "",
                "cache_key": per_frame_records[0]["cache_key"] if per_frame_records else "",
            })

    cache.save()
    elapsed_s = time.time() - t0

    all_events = intervention_events + visual_events
    manifest_path = config.results_dir / "controlled_event_manifest.csv"
    fieldnames = list(all_events[0].keys()) if all_events else []
    import csv
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_events)

    n_intervention = len(intervention_events)
    n_visual = len(visual_events)
    lines = ["# Controlled Experiment Protocol\n\n"]
    lines.append("## Target selection\n\n" + "\n".join(selection_log) + "\n\n")
    lines.append(
        "## Family A -- detector intervention\n\n"
        "The target's own detection row is demoted to confidence "
        f"**{DEMOTED_CONFIDENCE}** (coverage 0.5) or removed entirely (coverage 1.0) for exactly the "
        "window frames. Image pixels are never touched -- this isolates tracker behavior only and is "
        "never described as visual occlusion.\n\n"
        f"Events: **{n_intervention}** ({len(selected)} targets x {len(DURATIONS)} durations x "
        f"{len(COVERAGES)} coverage levels).\n\n"
    )
    lines.append(
        "## Family B -- controlled visual occlusion\n\n"
        "A seeded rectangular mask (gray, seeded per-event) is painted over the target's box on a "
        "**local copy** of the frame image -- the original nuScenes file is only ever opened for "
        "reading. The same frozen detector is then rerun on that copy (a genuine re-inference, not a "
        "confidence edit), and the resulting fresh detections replace that frame's detection list for "
        "this event only.\n\n"
        f"Events: **{n_visual}**. Rerun time: {elapsed_s:.1f}s.\n\n"
    )
    lines.append(
        "## Reproducibility\n\n"
        "Every event records its scene, target track, duration, coverage, RNG seed, the exact mask "
        "box, and (for Family B) the detector cache key -- any event can be recreated exactly by "
        "rerunning `scripts/build_controlled_events.py` with the same seed. Local-only outputs: "
        f"`OATM/artifacts/controlled_visual_frames/` ({n_visual} events' modified images, "
        "git-ignored, regenerable).\n\n"
        "**Families are never merged**: `event_source` (`detector_intervention` vs `controlled_visual`) "
        "is a required column in every downstream table that reads this manifest.\n"
    )

    with open(config.results_dir / "controlled_protocol.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {manifest_path} "
          f"({n_intervention} intervention + {n_visual} visual = {len(all_events)} events)")
    print("Wrote controlled_protocol.md")


if __name__ == "__main__":
    main()

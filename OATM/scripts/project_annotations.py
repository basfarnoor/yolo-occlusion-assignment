"""Phase 2 (Task 3): projects official nuScenes 3D annotations into CAM_FRONT
at every keyframe. Read-only against the dataset. Writes:

  - OATM/artifacts/projected_ground_truth.parquet (local-only, accepted rows only)
  - OATM/artifacts/projection_rejections.json     (local-only, full rejection detail)
  - OATM/results/projection_audit.md               (committed, compact summary)

Reproduction: `.venv/Scripts/python scripts/project_annotations.py` from OATM/.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.dataset.projection import map_evaluation_class, project_annotation  # noqa: E402
from oatm.records import ProjectedGroundTruthRecord  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
IMAGE_SIZE = (1600, 900)


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    meta_dir = config.data_root / config.dataset_version

    sample_data = {r["token"]: r for r in load_json(meta_dir / "sample_data.json")}
    samples = {s["token"]: s for s in load_json(meta_dir / "sample.json")}
    scenes = {s["token"]: s for s in load_json(meta_dir / "scene.json")}
    ego_poses = {e["token"]: e for e in load_json(meta_dir / "ego_pose.json")}
    calibrated_sensors = {c["token"]: c for c in load_json(meta_dir / "calibrated_sensor.json")}
    sensors = load_json(meta_dir / "sensor.json")
    instances = {r["token"]: r for r in load_json(meta_dir / "instance.json")}
    categories = {r["token"]: r for r in load_json(meta_dir / "category.json")}
    annotations = load_json(meta_dir / "sample_annotation.json")

    cam_front_sensor_token = next(s["token"] for s in sensors if s["channel"] == "CAM_FRONT")

    annotations_by_sample: dict[str, list[dict]] = defaultdict(list)
    for ann in annotations:
        annotations_by_sample[ann["sample_token"]].append(ann)

    # One CAM_FRONT sample_data per keyframe sample (there is exactly one,
    # since is_key_frame CAM_FRONT records align 1:1 with sample.json rows).
    cam_front_keyframes = [
        r for r in sample_data.values()
        if r["is_key_frame"]
        and calibrated_sensors[r["calibrated_sensor_token"]]["sensor_token"] == cam_front_sensor_token
    ]

    records: list[ProjectedGroundTruthRecord] = []
    rejections: list[dict] = []
    reject_reasons: dict[str, int] = defaultdict(int)
    n_considered = 0

    for sd_rec in cam_front_keyframes:
        sample = samples[sd_rec["sample_token"]]
        scene_token = sample["scene_token"]
        ego_pose = ego_poses[sd_rec["ego_pose_token"]]
        cs = calibrated_sensors[sd_rec["calibrated_sensor_token"]]
        if not cs.get("camera_intrinsic"):
            raise RuntimeError(
                f"No camera_intrinsic for sample_data {sd_rec['token']} -- required metadata missing."
            )

        for ann in annotations_by_sample.get(sample["token"], []):
            n_considered += 1
            instance = instances[ann["instance_token"]]
            original_category = categories[instance["category_token"]]["name"]
            vis_token = ann["visibility_token"]

            projected = project_annotation(ann, ego_pose, cs, IMAGE_SIZE)

            if projected.projection_status != "accepted":
                reject_reasons[projected.projection_status] += 1
                rejections.append({
                    "scene_token": scene_token, "sample_data_token": sd_rec["token"],
                    "instance_token": ann["instance_token"], "annotation_token": ann["token"],
                    "original_category": original_category, "reason": projected.projection_status,
                })
                continue

            records.append(ProjectedGroundTruthRecord(
                scene_token=scene_token,
                sample_data_token=sd_rec["token"],
                instance_token=ann["instance_token"],
                annotation_token=ann["token"],
                original_category=original_category,
                evaluation_class=map_evaluation_class(original_category) or "other",
                visibility_token=vis_token,
                x1=round(projected.x1, 2), y1=round(projected.y1, 2),
                x2=round(projected.x2, 2), y2=round(projected.y2, 2),
                center_depth_m=round(projected.center_depth_m, 3),
                num_lidar_pts=ann.get("num_lidar_pts", 0),
                num_radar_pts=ann.get("num_radar_pts", 0),
                truncation_fraction=round(projected.truncation_fraction, 4),
                projection_status=projected.projection_status,
            ))

    # Deterministic ordering.
    records.sort(key=lambda r: (r.scene_token, r.sample_data_token, r.instance_token))

    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump() for r in records])
    df.to_parquet(config.artifacts_dir / "projected_ground_truth.parquet", index=False)
    with open(config.artifacts_dir / "projection_rejections.json", "w", encoding="utf-8") as f:
        json.dump(rejections, f, indent=2)

    n_accepted = len(records)
    n_evaluation_class = sum(1 for r in records if r.evaluation_class in ("car", "pedestrian"))
    per_scene = defaultdict(lambda: {"keyframes": 0, "accepted": 0, "rejected": 0})
    for sd_rec in cam_front_keyframes:
        per_scene[samples[sd_rec["sample_token"]]["scene_token"]]["keyframes"] += 1
    for r in records:
        per_scene[r.scene_token]["accepted"] += 1
    for r in rejections:
        per_scene[r["scene_token"]]["rejected"] += 1

    lines = ["# Projection Audit\n\n"]
    lines.append(
        "These annotations are privileged offline evaluation evidence. They are not "
        "inputs to the online camera-only tracker.\n\n"
    )
    lines.append(f"Total annotation instances considered across all {len(cam_front_keyframes)} "
                 f"CAM_FRONT keyframes: **{n_considered}**.\n")
    lines.append(f"Accepted (projected successfully): **{n_accepted}**.\n")
    lines.append(
        f"Of which mapped to the MVP evaluation classes (car/pedestrian): **{n_evaluation_class}**.\n"
    )
    lines.append(f"Rejected: **{len(rejections)}**.\n\n")
    lines.append("## Rejection reasons\n\n")
    for reason, count in sorted(reject_reasons.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {count}x: {reason}\n")
    lines.append("\n## Per-scene keyframe coverage\n\n")
    lines.append("| Scene | Keyframes | Accepted boxes | Rejected boxes |\n|---|---:|---:|---:|\n")
    for scene_token, counts in sorted(per_scene.items(), key=lambda kv: scenes[kv[0]]["name"]):
        lines.append(f"| `{scenes[scene_token]['name']}` | {counts['keyframes']} | "
                     f"{counts['accepted']} | {counts['rejected']} |\n")

    n_clipped = sum(1 for r in records if r.truncation_fraction > 0)
    lines.append(f"\n**{n_clipped}** of the accepted boxes required clipping to the image boundary.\n")

    pgt_path = config.artifacts_dir / "projected_ground_truth.parquet"
    lines.append(
        f"\nLocal-only artifacts (git-ignored, regenerable with "
        f"`python scripts/project_annotations.py`):\n\n"
        f"- `OATM/artifacts/projected_ground_truth.parquet` -- {n_accepted} rows, "
        f"schema: `oatm.records.ProjectedGroundTruthRecord`, {pgt_path.stat().st_size / 1024:.0f} KB.\n"
        f"- `OATM/artifacts/projection_rejections.json` -- {len(rejections)} rejected annotations "
        f"with reasons, for audit.\n"
    )

    with open(config.results_dir / "projection_audit.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote projected_ground_truth.parquet ({n_accepted} accepted of {n_considered} considered)")
    print(f"Wrote projection_rejections.json ({len(rejections)} rejected)")
    print("Wrote projection_audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

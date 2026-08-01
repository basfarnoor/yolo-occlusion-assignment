"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 4 driver: projects official nuScenes 3D annotations into CAM_FRONT pixel
space at every annotated keyframe across the four clips, using
src/projection.py. Writes results/projected_ground_truth.csv (privileged
offline evaluation evidence -- never fed into the online tracker) and
results/projection_audit.md.

Reproduction: `python build_ground_truth.py` (needs clip_manifest.csv from
src/clip_builder.py; only touches the local nuScenes metadata, never the
original image/annotation files).
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from projection import project_annotation  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
REPO_ROOT = ASSIGNMENT_ROOT.parent.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
IMAGE_SIZE = (1600, 900)


def find_data_root() -> Path:
    for c in (REPO_ROOT / "data", REPO_ROOT / "data" / "nuscenes"):
        if (c / "v1.0-mini").is_dir():
            return c
    raise FileNotFoundError("nuScenes root not found")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    data_root = find_data_root()
    meta = data_root / "v1.0-mini"

    sample_data = {r["token"]: r for r in load_json(meta / "sample_data.json")}
    ego_pose = {r["token"]: r for r in load_json(meta / "ego_pose.json")}
    calibrated_sensor = {r["token"]: r for r in load_json(meta / "calibrated_sensor.json")}
    instances = {r["token"]: r for r in load_json(meta / "instance.json")}
    categories = {r["token"]: r for r in load_json(meta / "category.json")}
    visibility = {r["token"]: r for r in load_json(meta / "visibility.json")}

    annotations_by_sample: dict[str, list[dict]] = defaultdict(list)
    for ann in load_json(meta / "sample_annotation.json"):
        annotations_by_sample[ann["sample_token"]].append(ann)

    with open(OUT_ROOT / "clip_manifest.csv", newline="", encoding="utf-8") as f:
        clip_rows = list(csv.DictReader(f))

    keyframe_rows = [r for r in clip_rows if r["is_keyframe"] in ("True", "true", True)]

    out_rows = []
    reject_reasons: dict[str, int] = defaultdict(int)
    accepted = 0

    for row in keyframe_rows:
        sd_rec = sample_data[row["sample_data_token"]]
        sample_token = sd_rec["sample_token"]
        ep = ego_pose[sd_rec["ego_pose_token"]]
        cs = calibrated_sensor[sd_rec["calibrated_sensor_token"]]
        if not cs.get("camera_intrinsic"):
            raise RuntimeError(f"No camera_intrinsic for calibrated_sensor of {row['clip_name']} frame "
                               f"{row['frame_number']} -- required metadata missing.")

        for ann in annotations_by_sample.get(sample_token, []):
            instance = instances[ann["instance_token"]]
            category_name = categories[instance["category_token"]]["name"]
            vis_level = visibility[ann["visibility_token"]]["level"]

            pb = project_annotation(ann, ep, cs, category_name, vis_level, IMAGE_SIZE)
            out_rows.append({
                "clip_name": row["clip_name"],
                "frame_number": row["frame_number"],
                "sample_data_token": row["sample_data_token"],
                "sample_token": sample_token,
                "instance_token": pb.instance_token,
                "category": pb.category,
                "visibility_level": pb.visibility_level,
                "rejected": pb.rejected,
                "reject_reason": pb.reject_reason,
                "x1": round(pb.x1, 2), "y1": round(pb.y1, 2),
                "x2": round(pb.x2, 2), "y2": round(pb.y2, 2),
                "unclipped_x1": round(pb.unclipped_x1, 2), "unclipped_y1": round(pb.unclipped_y1, 2),
                "unclipped_x2": round(pb.unclipped_x2, 2), "unclipped_y2": round(pb.unclipped_y2, 2),
                "was_clipped": pb.was_clipped,
                "depth_m": round(pb.depth_m, 3),
                "num_lidar_pts": pb.num_lidar_pts,
                "num_radar_pts": pb.num_radar_pts,
            })
            if pb.rejected:
                reject_reasons[pb.reject_reason] += 1
            else:
                accepted += 1

    # Deterministic ordering for reproducibility.
    out_rows.sort(key=lambda r: (r["clip_name"], int(r["frame_number"]), r["instance_token"]))

    fieldnames = ["clip_name", "frame_number", "sample_data_token", "sample_token", "instance_token",
                  "category", "visibility_level", "rejected", "reject_reason",
                  "x1", "y1", "x2", "y2", "unclipped_x1", "unclipped_y1", "unclipped_x2", "unclipped_y2",
                  "was_clipped", "depth_m", "num_lidar_pts", "num_radar_pts"]
    gt_path = OUT_ROOT / "projected_ground_truth.csv"
    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    per_clip_counts = defaultdict(lambda: {"keyframes": 0, "accepted": 0, "rejected": 0})
    for row in keyframe_rows:
        per_clip_counts[row["clip_name"]]["keyframes"] += 1
    for row in out_rows:
        key = "accepted" if not row["rejected"] else "rejected"
        per_clip_counts[row["clip_name"]][key] += 1

    audit_lines = ["# Projection Audit\n\n"]
    audit_lines.append(
        "nuScenes labels are used to evaluate the camera tracker. They are not inputs "
        "to ByteTrack or SORT during online inference.\n\n")
    audit_lines.append(f"Total annotation instances considered across all keyframes: **{len(out_rows)}**.\n")
    audit_lines.append(f"Accepted (projected successfully): **{accepted}**.\n")
    audit_lines.append(f"Rejected: **{len(out_rows) - accepted}**.\n\n")
    audit_lines.append("## Rejection reasons\n\n")
    if reject_reasons:
        for reason, count in sorted(reject_reasons.items(), key=lambda kv: -kv[1]):
            audit_lines.append(f"- {count}x: {reason}\n")
    else:
        audit_lines.append("- None.\n")
    audit_lines.append("\n## Per-clip keyframe coverage\n\n")
    audit_lines.append("| Clip | Keyframes | Accepted boxes | Rejected boxes |\n|---|---:|---:|---:|\n")
    for clip_name, counts in sorted(per_clip_counts.items()):
        audit_lines.append(f"| `{clip_name}` | {counts['keyframes']} | {counts['accepted']} | {counts['rejected']} |\n")

    n_clipped = sum(1 for r in out_rows if not r["rejected"] and r["was_clipped"])
    audit_lines.append(f"\n**{n_clipped}** of the accepted boxes required clipping to the image boundary.\n")

    audit_path = OUT_ROOT / "projection_audit.md"
    with open(audit_path, "w", encoding="utf-8") as f:
        f.writelines(audit_lines)

    print(f"Wrote {gt_path} ({len(out_rows)} rows, {accepted} accepted)")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()

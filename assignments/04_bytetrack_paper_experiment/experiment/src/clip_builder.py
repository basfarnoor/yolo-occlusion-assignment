"""ByteTrack paper reference: Zhang et al., "ByteTrack: Multi-Object Tracking by
Associating Every Detection Box," ECCV 2022 (https://arxiv.org/abs/2110.06864).

Builds the scene-disjoint clip set for Assignment 4. Reuses the three
Assignment 3 clips (chain-walked from the same anchor frames, since their
copied frame files are not present in this checkout -- results/clips/ is
git-ignored local-only output) and adds one new clip from a fourth,
previously-unused scene so a development/evaluation split can be scene-disjoint.

ByteTrack is an *online* tracker: frame t may only use frame t and earlier
frames. This module's only job is to reconstruct correct chronological order
and scene identity around a few anchor frames, without touching the original
dataset.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
ASSIGNMENT_ROOT = EXPERIMENT_ROOT.parent
REPO_ROOT = ASSIGNMENT_ROOT.parent.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
CLIPS_ROOT = OUT_ROOT / "clips"
ASSIGNMENT3_ROOT = REPO_ROOT / "assignments" / "03_sort_paper_experiment"

MAX_FRAMES_PER_CLIP = 36
MAX_TOTAL_FRAMES = 144
MAX_CLIPS = 4
TARGET_SECONDS_EACH_SIDE = 1.5

# Reused from Assignment 3 (assignments/03_sort_paper_experiment/results/clip_manifest.csv).
REUSED_ANCHORS = ["sample_001", "sample_003", "sample_006"]
# New clip: a fourth, previously-unused nuScenes scene (scene-1094, "night, many
# peds, jaywalker, truck" -- also used as an anchor in Assignment 1, sample_011),
# needed so the split below is scene-disjoint rather than only frame-disjoint.
NEW_ANCHOR = "sample_011"
ANCHOR_STAGE_PREFERENCE = ["full_occlusion", "previous_no_occlusion", "full_appearance"]

# Scene-disjoint development / evaluation split, fixed before any threshold is
# chosen or any evaluation result is opened (Task 5 requirement).
DEV_SCENES = {"scene-0103", "scene-0757"}    # sample_001, sample_006
EVAL_SCENES = {"scene-0553", "scene-1094"}   # sample_003, sample_011 (new)


def find_data_root(project_root: Path) -> Path:
    candidates = [project_root / "data", project_root / "data" / "nuscenes"]
    for c in candidates:
        if (c / "v1.0-mini").is_dir() and (c / "samples").is_dir():
            return c
    raise FileNotFoundError("Could not find a nuScenes root (expected samples/, sweeps/, v1.0-mini/).")


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class NuscenesIndex:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        sd_records = load_json(data_root / "v1.0-mini" / "sample_data.json")
        self.sample_data_by_token = {r["token"]: r for r in sd_records}
        self.sample_data_by_filename = {r["filename"]: r for r in sd_records}
        self.sample_by_token = {s["token"]: s for s in load_json(data_root / "v1.0-mini" / "sample.json")}
        self.scene_by_token = {s["token"]: s for s in load_json(data_root / "v1.0-mini" / "scene.json")}

    def scene_for_sample_data(self, sd_rec: dict) -> tuple[str, str]:
        sample = self.sample_by_token[sd_rec["sample_token"]]
        scene = self.scene_by_token[sample["scene_token"]]
        return scene["name"], sample["scene_token"]


def anchor_source_path(anchor_sample: str) -> str | None:
    """Look up the original nuScenes relative path for an Assignment-1 anchor
    sample, preferring the full_occlusion frame as the clip's temporal center."""
    manifest_path = REPO_ROOT / "assignments" / "01_yolo_occlusion" / "occluded_samples" / "manifest.csv"
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    by_stage = {}
    for r in rows:
        if r["destination_path"].startswith(f"occluded_samples/{anchor_sample}/") and r["status"] in ("copied", "already_present"):
            by_stage[r["stage_name"]] = r["source_path"]
    for stage in ANCHOR_STAGE_PREFERENCE:
        if stage in by_stage:
            return by_stage[stage]
    return next(iter(by_stage.values()), None)


def walk_chain(sample_data_by_token: dict, start_token: str, direction: str, max_frames: int) -> list[dict]:
    out = []
    token = sample_data_by_token[start_token][direction]
    while token and len(out) < max_frames:
        rec = sample_data_by_token.get(token)
        if rec is None:
            break
        out.append(rec)
        token = rec[direction]
    return out


def build_chain_from_anchor_token(index: NuscenesIndex, anchor_token: str) -> list[dict]:
    anchor_rec = index.sample_data_by_token[anchor_token]
    before = walk_chain(index.sample_data_by_token, anchor_token, "prev", MAX_FRAMES_PER_CLIP)
    before.reverse()
    after = walk_chain(index.sample_data_by_token, anchor_token, "next", MAX_FRAMES_PER_CLIP)

    def trim_by_seconds(records: list[dict], ref_ts: int) -> list[dict]:
        limit_us = int(TARGET_SECONDS_EACH_SIDE * 1_000_000)
        return [r for r in records if abs(r["timestamp"] - ref_ts) <= limit_us]

    before = trim_by_seconds(before, anchor_rec["timestamp"])
    after = trim_by_seconds(after, anchor_rec["timestamp"])
    chain = before + [anchor_rec] + after
    if len(chain) > MAX_FRAMES_PER_CLIP:
        excess = len(chain) - MAX_FRAMES_PER_CLIP
        trim_front = excess // 2
        trim_back = excess - trim_front
        chain = chain[trim_front: len(chain) - trim_back]
    return chain


def copy_and_manifest(index: NuscenesIndex, clip_name: str, anchor_sample: str, chain: list[dict],
                       reused_from: str) -> list[dict]:
    clip_dir = CLIPS_ROOT / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_ts = None
    for i, rec in enumerate(chain, start=1):
        src = index.data_root / rec["filename"]
        if not src.is_file():
            raise FileNotFoundError(f"Missing source frame: {src}")
        ext = src.suffix
        dst_name = f"frame_{i:03d}{ext}"
        dst = clip_dir / dst_name
        shutil.copy2(src, dst)

        ts = rec["timestamp"]
        time_since_prev = (ts - prev_ts) / 1_000_000 if prev_ts is not None else 0.0
        prev_ts = ts

        scene_name, scene_token = index.scene_for_sample_data(rec)
        rows.append({
            "clip_name": clip_name,
            "frame_number": i,
            "sample_data_token": rec["token"],
            "scene_token": scene_token,
            "scene_name": scene_name,
            "timestamp": ts,
            "time_since_prev_s": round(time_since_prev, 4),
            "original_image_path": rec["filename"],
            "experiment_image_path": str(dst.relative_to(ASSIGNMENT_ROOT)).replace("\\", "/"),
            "is_keyframe": rec["is_key_frame"],
            "anchor_sample": anchor_sample,
            "reused_from_assignment": reused_from,
            "image_sha256": sha256_of_file(dst),
        })
    return rows


def main() -> None:
    data_root = find_data_root(REPO_ROOT)
    index = NuscenesIndex(data_root)

    manifest_rows: list[dict] = []
    check_lines: list[str] = ["# Data Check\n\n"]
    check_lines.append(f"nuScenes root discovered at: `{data_root.relative_to(REPO_ROOT)}`\n\n")

    all_anchors = [(a, "assignment_03") for a in REUSED_ANCHORS] + [(NEW_ANCHOR, "new_assignment_04")]
    scenes_used: dict[str, str] = {}

    for anchor, reused_from in all_anchors:
        if len(manifest_rows) >= MAX_TOTAL_FRAMES or len({r["clip_name"] for r in manifest_rows}) >= MAX_CLIPS:
            break
        clip_name = f"clip_{anchor}"
        rel_path = anchor_source_path(anchor)
        if rel_path is None:
            check_lines.append(f"- **{anchor}**: SKIPPED -- no source image found in Assignment 1 manifest.\n")
            continue
        anchor_rec = index.sample_data_by_filename.get(rel_path.replace("\\", "/"))
        if anchor_rec is None:
            check_lines.append(f"- **{anchor}**: SKIPPED -- no sample_data entry for `{rel_path}`.\n")
            continue

        chain = build_chain_from_anchor_token(index, anchor_rec["token"])
        rows = copy_and_manifest(index, clip_name, anchor, chain, reused_from)

        timestamps = [r["timestamp"] for r in rows]
        strictly_increasing = all(b > a for a, b in zip(timestamps, timestamps[1:]))
        all_exist = all((ASSIGNMENT_ROOT / r["experiment_image_path"]).is_file() for r in rows)
        one_scene = len({r["scene_token"] for r in rows}) == 1

        if not (strictly_increasing and all_exist and one_scene):
            check_lines.append(
                f"- **{anchor}** ({clip_name}): FAILED validation -- "
                f"increasing_timestamps={strictly_increasing}, all_files_exist={all_exist}, "
                f"single_scene={one_scene}\n")
            continue

        manifest_rows.extend(rows)
        scene_name = rows[0]["scene_name"]
        scenes_used[clip_name] = scene_name
        n_key = sum(1 for r in rows if r["is_keyframe"])
        check_lines.append(
            f"- **{anchor}** -> `{clip_name}`: {len(rows)} frames from **{scene_name}** "
            f"({n_key} keyframes, {len(rows) - n_key} sweeps), reused_from={reused_from}, "
            "timestamps strictly increasing, all files verified present, single scene confirmed.\n")

    manifest_path = OUT_ROOT / "clip_manifest.csv"
    fieldnames = ["clip_name", "frame_number", "sample_data_token", "scene_token", "scene_name",
                  "timestamp", "time_since_prev_s", "original_image_path", "experiment_image_path",
                  "is_keyframe", "anchor_sample", "reused_from_assignment", "image_sha256"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Scene-disjoint development / evaluation split.
    split_rows = []
    scene_to_split = {}
    for scene in DEV_SCENES:
        scene_to_split[scene] = "development"
    for scene in EVAL_SCENES:
        scene_to_split[scene] = "evaluation"

    overlap = DEV_SCENES & EVAL_SCENES
    for clip_name, scene_name in scenes_used.items():
        split = scene_to_split.get(scene_name, "UNASSIGNED")
        split_rows.append({"clip_name": clip_name, "scene_name": scene_name, "split": split})

    split_path = OUT_ROOT / "split_manifest.csv"
    with open(split_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_name", "scene_name", "split"])
        writer.writeheader()
        writer.writerows(split_rows)

    check_lines.append(f"\n**Clips built: {len(scenes_used)} / {MAX_CLIPS} max.**\n")
    check_lines.append(f"**Total frames across all clips: {len(manifest_rows)} / {MAX_TOTAL_FRAMES} max.**\n\n")
    check_lines.append("## Scene-disjoint split\n\n")
    for row in split_rows:
        check_lines.append(f"- `{row['clip_name']}` ({row['scene_name']}) -> **{row['split']}**\n")
    check_lines.append(f"\n**Scene overlap between development and evaluation: {sorted(overlap) or 'none'}.**\n")

    check_path = OUT_ROOT / "data_check.md"
    with open(check_path, "w", encoding="utf-8") as f:
        f.writelines(check_lines)

    print(f"Wrote {manifest_path} ({len(manifest_rows)} rows)")
    print(f"Wrote {split_path} ({len(split_rows)} clips)")
    print(f"Wrote {check_path}")
    print(f"Clips built: {len(scenes_used)}/{MAX_CLIPS}, scene overlap: {sorted(overlap) or 'none'}")


if __name__ == "__main__":
    main()

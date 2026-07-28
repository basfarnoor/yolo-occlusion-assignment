"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Builds short, time-ordered CAM_FRONT clips from the local nuScenes mini
dataset using the sample_data prev/next chain. SORT is an *online* tracker
-- it only ever sees frames in chronological order -- so this builder's job
is purely to reconstruct that chronological order around a few anchor
frames, without touching or altering the original dataset.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "sort_paper_experiment"
CLIPS_ROOT = OUT_ROOT / "clips"

MAX_FRAMES_PER_CLIP = 36
TARGET_SECONDS_EACH_SIDE = 1.5  # ~3s total, well under the 36-frame cap at ~12Hz

ANCHOR_SAMPLES = ["sample_001", "sample_003", "sample_006"]
ANCHOR_STAGE_PREFERENCE = ["full_occlusion", "previous_no_occlusion", "full_appearance"]


def find_data_root(project_root: Path) -> Path:
    """Discover the nuScenes root instead of hardcoding it."""
    candidates = [
        project_root / "data",
        project_root / "data" / "nuscenes",
    ]
    for c in candidates:
        if (c / "v1.0-mini").is_dir() and (c / "samples").is_dir():
            return c
    raise FileNotFoundError("Could not find a nuScenes root (expected samples/, sweeps/, v1.0-mini/).")


def load_sample_data(data_root: Path) -> dict[str, dict]:
    with open(data_root / "v1.0-mini" / "sample_data.json", encoding="utf-8") as f:
        records = json.load(f)
    return {r["token"]: r for r in records}


def anchor_source_path(sample: str) -> str | None:
    """Look up the original nuScenes relative path for one of our anchor samples,
    preferring the full_occlusion frame as the temporal center of the clip."""
    manifest_path = PROJECT_ROOT / "occluded_samples" / "manifest.csv"
    rows = list(csv.DictReader(open(manifest_path, encoding="utf-8")))
    by_stage = {}
    for r in rows:
        if r["destination_path"].startswith(f"occluded_samples/{sample}/") and r["status"] in ("copied", "already_present"):
            by_stage[r["stage_name"]] = r["source_path"]
    for stage in ANCHOR_STAGE_PREFERENCE:
        if stage in by_stage:
            return by_stage[stage]
    return next(iter(by_stage.values()), None)


def find_token_by_filename(sample_data: dict[str, dict], rel_path: str) -> str | None:
    rel_path_norm = rel_path.replace("\\", "/")
    for token, rec in sample_data.items():
        if rec["filename"] == rel_path_norm:
            return token
    return None


def walk_chain(sample_data: dict[str, dict], start_token: str, direction: str, max_frames: int) -> list[dict]:
    """direction: 'prev' or 'next'. Returns records in the order walked (not yet reversed)."""
    out = []
    token = sample_data[start_token][direction]
    while token and len(out) < max_frames:
        rec = sample_data.get(token)
        if rec is None:
            break
        out.append(rec)
        token = rec[direction]
    return out


def build_clip(data_root: Path, sample_data: dict[str, dict], anchor_sample: str, clip_name: str) -> dict:
    rel_path = anchor_source_path(anchor_sample)
    if rel_path is None:
        return {"anchor_sample": anchor_sample, "clip_name": clip_name, "ok": False,
                "reason": f"No copied source image found in manifest for {anchor_sample}."}

    anchor_token = find_token_by_filename(sample_data, rel_path)
    if anchor_token is None:
        return {"anchor_sample": anchor_sample, "clip_name": clip_name, "ok": False,
                "reason": f"Could not find sample_data entry for {rel_path}."}

    anchor_rec = sample_data[anchor_token]

    before = walk_chain(sample_data, anchor_token, "prev", MAX_FRAMES_PER_CLIP)
    before.reverse()
    after = walk_chain(sample_data, anchor_token, "next", MAX_FRAMES_PER_CLIP)

    # Trim by target seconds each side, then hard-cap total frames.
    def trim_by_seconds(records: list[dict], ref_ts: int) -> list[dict]:
        limit_us = int(TARGET_SECONDS_EACH_SIDE * 1_000_000)
        return [r for r in records if abs(r["timestamp"] - ref_ts) <= limit_us]

    before = trim_by_seconds(before, anchor_rec["timestamp"])
    after = trim_by_seconds(after, anchor_rec["timestamp"])

    chain = before + [anchor_rec] + after
    if len(chain) > MAX_FRAMES_PER_CLIP:
        # keep it centered: trim evenly from both ends
        excess = len(chain) - MAX_FRAMES_PER_CLIP
        trim_front = excess // 2
        trim_back = excess - trim_front
        chain = chain[trim_front: len(chain) - trim_back]

    return {"anchor_sample": anchor_sample, "clip_name": clip_name, "ok": True,
            "anchor_token": anchor_token, "chain": chain}


def copy_and_manifest(data_root: Path, build_result: dict) -> list[dict]:
    clip_name = build_result["clip_name"]
    clip_dir = CLIPS_ROOT / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    prev_ts = None
    for i, rec in enumerate(build_result["chain"], start=1):
        src = data_root / rec["filename"]
        if not src.is_file():
            raise FileNotFoundError(f"Missing source frame: {src}")
        ext = src.suffix
        dst_name = f"frame_{i:03d}{ext}"
        dst = clip_dir / dst_name
        if not dst.exists():
            shutil.copy2(src, dst)

        ts = rec["timestamp"]
        time_since_prev = (ts - prev_ts) / 1_000_000 if prev_ts is not None else 0.0
        prev_ts = ts

        rows.append({
            "clip_name": clip_name,
            "frame_number": i,
            "timestamp": ts,
            "time_since_prev_s": round(time_since_prev, 4),
            "original_image_path": rec["filename"],
            "experiment_image_path": str(dst.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "is_keyframe": rec["is_key_frame"],
            "anchor_sample": build_result["anchor_sample"],
        })
    return rows


def main() -> None:
    data_root = find_data_root(PROJECT_ROOT)
    sample_data = load_sample_data(data_root)

    manifest_rows: list[dict] = []
    check_lines: list[str] = ["# Data Check\n"]
    check_lines.append(f"nuScenes root discovered at: `{data_root.relative_to(PROJECT_ROOT)}`\n")

    used_clips = 0
    for anchor in ANCHOR_SAMPLES:
        if used_clips >= 3:
            break
        clip_name = f"clip_{anchor}"
        result = build_clip(data_root, sample_data, anchor, clip_name)
        if not result["ok"]:
            check_lines.append(f"- **{anchor}**: SKIPPED -- {result['reason']}\n")
            continue

        rows = copy_and_manifest(data_root, result)

        # Validate: timestamps strictly increasing, files exist.
        timestamps = [r["timestamp"] for r in rows]
        strictly_increasing = all(b > a for a, b in zip(timestamps, timestamps[1:]))
        all_exist = all((PROJECT_ROOT / r["experiment_image_path"]).is_file() for r in rows)

        if not strictly_increasing or not all_exist:
            check_lines.append(
                f"- **{anchor}** ({clip_name}): FAILED validation -- "
                f"increasing_timestamps={strictly_increasing}, all_files_exist={all_exist}\n")
            continue

        manifest_rows.extend(rows)
        used_clips += 1
        n_key = sum(1 for r in rows if r["is_keyframe"])
        check_lines.append(
            f"- **{anchor}** -> `{clip_name}`: {len(rows)} frames "
            f"({n_key} keyframes, {len(rows) - n_key} sweeps), "
            f"timestamps strictly increasing, all files verified present.\n")

    manifest_path = OUT_ROOT / "clip_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_name", "frame_number", "timestamp", "time_since_prev_s",
            "original_image_path", "experiment_image_path", "is_keyframe", "anchor_sample",
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    check_lines.append(f"\n**Clips built: {used_clips} / 3 requested.**\n")
    check_lines.append(f"**Total frames across all clips: {len(manifest_rows)}.**\n")
    check_path = OUT_ROOT / "data_check.md"
    with open(check_path, "w", encoding="utf-8") as f:
        f.writelines(check_lines)

    print(f"Wrote {manifest_path} ({len(manifest_rows)} rows)")
    print(f"Wrote {check_path}")
    print(f"Clips built: {used_clips}/3")


if __name__ == "__main__":
    main()

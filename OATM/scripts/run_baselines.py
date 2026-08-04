"""Phase 4 (Task 6): runs all four baselines (YOLO-only, static memory, SORT,
ByteTrack) over every scene, from one configuration, each receiving the
identical ordered frames and identical raw detections. A fresh tracker
instance is created per scene per method, so no track ID ever crosses a
scene boundary.

Reproduction: `.venv/Scripts/python scripts/run_baselines.py` from OATM/.
"""
from __future__ import annotations

import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.tracking.bytetrack_adapter import ByteTrackAdapter  # noqa: E402
from oatm.tracking.kalman import KalmanBoxTracker  # noqa: E402
from oatm.tracking.sort_adapter import SortAdapter  # noqa: E402
from oatm.tracking.static_memory import StaticMemoryTracker, _StaticTrack  # noqa: E402
from oatm.tracking.yolo_only import run_yolo_only_frame  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent


def load_tracker_config() -> dict:
    with open(OATM_ROOT / "configs" / "tracker.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    tcfg = load_tracker_config()

    detections = pd.read_parquet(config.artifacts_dir / "detections.parquet")
    frame_index = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")

    dets_by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in detections.to_dict("records"):
        dets_by_frame[row["sample_data_token"]].append({
            "class": row["detected_class"], "confidence": row["confidence"],
            "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"],
        })

    run_id = uuid.uuid4().hex[:12]
    all_outputs = []
    t0 = time.time()

    scenes = frame_index.sort_values(["scene_token", "frame_index"]).groupby("scene_token")
    n_scenes = 0
    for scene_token, scene_frames in scenes:
        n_scenes += 1
        frames = scene_frames.to_dict("records")

        KalmanBoxTracker.reset_id_counter()
        _StaticTrack.reset_id_counter()
        static_tracker = StaticMemoryTracker(**tcfg["static_memory"])
        sort_tracker = SortAdapter(**tcfg["sort"])
        byte_tracker = ByteTrackAdapter(**tcfg["bytetrack"])

        for f in frames:
            raw_dets = dets_by_frame.get(f["sample_data_token"], [])
            # The SAME raw detection list object is passed to all four methods.
            common_kwargs = dict(scene_token=scene_token, sample_data_token=f["sample_data_token"])
            ts = f["timestamp_us"] / 1_000_000.0

            all_outputs.extend(run_yolo_only_frame(
                raw_dets, tcfg["shared"]["high_score_threshold"], frame_index=f["frame_index"],
                method_name="yolo_only", run_id=run_id, **common_kwargs))
            all_outputs.extend(static_tracker.update(raw_dets, timestamp=ts, method_name="static_memory",
                                                       run_id=run_id, **common_kwargs))
            all_outputs.extend(sort_tracker.update(raw_dets, timestamp=ts, method_name="sort",
                                                     run_id=run_id, **common_kwargs))
            all_outputs.extend(byte_tracker.update(raw_dets, timestamp=ts, method_name="bytetrack",
                                                     run_id=run_id, **common_kwargs))

    elapsed_s = time.time() - t0

    df = pd.DataFrame([o.model_dump() for o in all_outputs])
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.artifacts_dir / "baseline_outputs.parquet", index=False)

    lines = ["# Baseline Summary (Task 6)\n\n"]
    lines.append(f"Run ID: `{run_id}`. Scenes: {n_scenes}. Elapsed: {elapsed_s:.1f}s.\n\n")
    lines.append("| Method | Output rows | Unique track IDs | Mean track length (rows) |\n"
                 "|---|---:|---:|---:|\n")
    for method in ("static_memory", "sort", "bytetrack"):
        sub = df[df["method_name"] == method]
        n_tracks = sub.groupby(["scene_token", "track_id"]).ngroups if len(sub) else 0
        mean_len = len(sub) / n_tracks if n_tracks else 0.0
        lines.append(f"| {method} | {len(sub)} | {n_tracks} | {mean_len:.2f} |\n")
    n_yolo_rows = len(df[df["method_name"] == "yolo_only"])
    lines.append(f"| yolo_only | {n_yolo_rows} | n/a | n/a |\n")
    lines.append(
        "\n`yolo_only`'s `track_id` is a fresh per-frame index with no real cross-frame identity "
        "(see reuse_audit.md) -- grouping by `(scene_token, track_id)` would silently merge "
        "unrelated detections from different frames into fake, meaningless \"tracks\" that just "
        "happen to share the same reused index number, so those two columns are intentionally "
        "reported as not applicable rather than computed and misread as real persistence.\n"
    )

    baseline_path = config.artifacts_dir / "baseline_outputs.parquet"
    lines.append(
        f"\nLocal-only artifact (git-ignored, regenerable with `python scripts/run_baselines.py`):\n\n"
        f"- `OATM/artifacts/baseline_outputs.parquet` -- {len(all_outputs)} rows, "
        f"schema: `oatm.records.TrackerOutputRecord`, {baseline_path.stat().st_size / 1024:.0f} KB.\n"
    )

    with open(config.results_dir / "baseline_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote baseline_outputs.parquet ({len(all_outputs)} rows across {n_scenes} scenes, "
          f"{elapsed_s:.1f}s)")
    print("Wrote baseline_summary.md")


if __name__ == "__main__":
    main()

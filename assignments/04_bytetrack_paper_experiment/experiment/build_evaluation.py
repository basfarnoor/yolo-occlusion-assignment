"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 11 driver: evaluates natural events against independent projected
ground truth, summarizes the controlled-experiment trials, times each method
SEPARATELY (required repair #5), and writes every required Task 11 artifact.
Natural and controlled results are kept in separate files and never pooled
together (required repair #6 / #8: state scene/clip/track/event/frame/row
counts separately).
"""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from natural_event_evaluation import evaluate_natural_events  # noqa: E402
from run_methods import new_bytetrack_tracker, new_sort_tracker  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": "", "median": "", "std": "", "n": 0}
    return {
        "mean": round(statistics.mean(values), 4),
        "median": round(statistics.median(values), 4),
        "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def measure_runtime_separately(clip_frame_numbers: dict, detections_by_frame: dict, timestamps_by_clip: dict,
                                 cfg: dict, repeats: int = 3) -> dict:
    """Times each method's per-frame update() call separately, over `repeats`
    independent passes per clip, discarding no warm-up (trackers are cheap and
    deterministic) -- reports median ms/frame per method. This directly
    repairs Assignment 3's shared-timer bug (reuse_audit.md, required repair #5)."""
    per_method_times_ms = defaultdict(list)
    for clip, frame_numbers in clip_frame_numbers.items():
        timestamps = timestamps_by_clip[clip]
        for _ in range(repeats):
            for method_name, factory in (("high_confidence_sort", new_sort_tracker),
                                          ("bytetrack", new_bytetrack_tracker)):
                tracker = factory(cfg)
                for frame_no in frame_numbers:
                    dets = detections_by_frame.get((clip, frame_no), [])
                    t0 = time.perf_counter()
                    tracker.update(dets, timestamp=timestamps[frame_no])
                    per_method_times_ms[method_name].append((time.perf_counter() - t0) * 1000)
    return {name: _stats(times) for name, times in per_method_times_ms.items()}


def summarize_controlled(controlled_trials: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (summary_by_event, summary_by_track) for the controlled trials."""
    by_event = defaultdict(list)
    for r in controlled_trials:
        by_event[(r["event_id"], r["method"])].append(r)

    event_summaries = []
    for (event_id, method), rows in sorted(by_event.items()):
        window_rows = [r for r in rows if r["in_window"] in ("True", "true")]
        after_rows = sorted((r for r in rows if r["in_window"] in ("False", "false")),
                             key=lambda r: int(r["frame_number"]))
        # "after" rows immediately following the window (frame_number greater
        # than every window frame).
        window_frame_nums = {int(r["frame_number"]) for r in window_rows}
        max_window_frame = max(window_frame_nums) if window_frame_nums else -1
        post_window_rows = [r for r in after_rows if int(r["frame_number"]) > max_window_frame]

        coverage = sum(1 for r in window_rows if r["has_output"] in ("True", "true")) / len(window_rows) if window_rows else ""
        window_ious = [float(r["iou"]) for r in window_rows if r["has_output"] in ("True", "true")]
        post_id_continuous = [r["id_continuous_from_before_window"] in ("True", "true") for r in post_window_rows]
        low_score_in_window = sum(1 for r in window_rows if r["evidence_source"] == "low_score_detection")

        event_summaries.append({
            "event_id": event_id, "method": method,
            "n_window_frames": len(window_rows),
            "n_post_window_frames": len(post_window_rows),
            "window_coverage_rate": round(coverage, 3) if coverage != "" else "",
            "window_mean_iou": round(sum(window_ious) / len(window_ious), 4) if window_ious else "",
            "window_low_score_match_count": low_score_in_window,
            "post_window_id_continuity_rate": (
                round(sum(post_id_continuous) / len(post_id_continuous), 3) if post_id_continuous else ""),
        })

    # Group by natural track (strip the "_confidence_demotion_wN" / "_complete_absence_wN" suffix).
    def track_key_of(event_id: str) -> str:
        for mode_suffix in ("_confidence_demotion_w", "_complete_absence_w"):
            if mode_suffix in event_id:
                return event_id.split(mode_suffix)[0]
        return event_id

    by_track = defaultdict(list)
    for s in event_summaries:
        by_track[(track_key_of(s["event_id"]), s["method"])].append(s)

    track_summaries = []
    for (track_key, method), summaries in sorted(by_track.items()):
        coverages = [s["window_coverage_rate"] for s in summaries if s["window_coverage_rate"] != ""]
        continuities = [s["post_window_id_continuity_rate"] for s in summaries if s["post_window_id_continuity_rate"] != ""]
        track_summaries.append({
            "track_key": track_key, "method": method,
            "n_events_for_this_track": len(summaries),
            "mean_window_coverage_rate": round(sum(coverages) / len(coverages), 3) if coverages else "",
            "mean_post_window_id_continuity_rate": round(sum(continuities) / len(continuities), 3) if continuities else "",
        })

    return event_summaries, track_summaries


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_hash = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")
    split_rows = load_csv(OUT_ROOT / "split_manifest.csv")
    gt_rows = load_csv(OUT_ROOT / "projected_ground_truth.csv")
    natural_events = load_csv(OUT_ROOT / "natural_event_manifest.csv")
    controlled_trials = load_csv(OUT_ROOT / "controlled_trials.csv")
    controlled_events = load_csv(OUT_ROOT / "controlled_event_manifest.csv")

    split_by_clip = {r["clip_name"]: r["split"] for r in split_rows}
    for e in natural_events:
        e["split"] = split_by_clip.get(e["clip_name"], "UNKNOWN")

    clip_frame_numbers = defaultdict(list)
    timestamps_by_clip = defaultdict(dict)
    for row in manifest_rows:
        clip, frame_no = row["clip_name"], int(row["frame_number"])
        clip_frame_numbers[clip].append(frame_no)
        timestamps_by_clip[clip][frame_no] = float(row["timestamp"]) / 1_000_000.0
    for clip in clip_frame_numbers:
        clip_frame_numbers[clip].sort()

    detections_by_frame = defaultdict(list)
    for d in detections:
        detections_by_frame[(d["clip"], int(d["frame_number"]))].append({
            "class": d["class"], "confidence": float(d["confidence"]),
            "x1": float(d["x1"]), "y1": float(d["y1"]), "x2": float(d["x2"]), "y2": float(d["y2"]),
        })

    gt_by_key = {}
    for g in gt_rows:
        if g["rejected"] in ("True", "true"):
            continue
        gt_by_key[(g["clip_name"], g["frame_number"], g["instance_token"])] = g

    print("Evaluating natural events against independent projected ground truth...")
    natural_trials = evaluate_natural_events(natural_events, gt_by_key, dict(clip_frame_numbers),
                                              dict(detections_by_frame), dict(timestamps_by_clip), cfg)
    write_csv(OUT_ROOT / "natural_trials.csv", natural_trials)
    print(f"Wrote natural_trials.csv ({len(natural_trials)} rows)")

    # --- summary_by_event.csv (natural events) ---
    natural_by_event_method = defaultdict(list)
    for r in natural_trials:
        natural_by_event_method[(r["instance_token"], r["clip_name"], r["method"])].append(r)

    natural_event_summary = []
    for (instance_token, clip_name, method), rows in sorted(natural_by_event_method.items()):
        event_row = rows[0]
        # natural_trials is the in-memory list from evaluate_natural_events --
        # "recovered" and "identity_preserved_before_to_after" are real Python
        # bools here (not CSV strings), so compare truthiness directly.
        recovered_at_event = next((bool(r["recovered"]) for r in rows if r["role"] == "event"), False)
        identity_preserved = bool(rows[0]["identity_preserved_before_to_after"])
        natural_event_summary.append({
            "instance_token": instance_token, "clip_name": clip_name, "split": event_row["split"],
            "category": event_row["category"], "method": method,
            "recovered_at_event_frame": recovered_at_event,
            "identity_preserved_before_to_after": identity_preserved,
        })
    write_csv(OUT_ROOT / "natural_event_summary.csv", natural_event_summary)

    controlled_event_summary, controlled_track_summary = summarize_controlled(controlled_trials)
    write_csv(OUT_ROOT / "summary_by_event.csv", controlled_event_summary)
    write_csv(OUT_ROOT / "summary_by_track.csv", controlled_track_summary)

    # --- summary_by_method.csv: pooled, natural and controlled kept in SEPARATE rows ---
    method_summary = []
    for method in ("high_confidence_sort", "bytetrack"):
        nat_rows = [r for r in natural_event_summary if r["method"] == method]
        n_natural_events = len(nat_rows)
        recovery_rate = (sum(1 for r in nat_rows if r["recovered_at_event_frame"]) / n_natural_events
                          if n_natural_events else "")
        identity_rate = (sum(1 for r in nat_rows if r["identity_preserved_before_to_after"]) / n_natural_events
                          if n_natural_events else "")
        method_summary.append({
            "method": method, "evidence_type": "natural_events",
            "n_events": n_natural_events, "n_rows": sum(1 for r in natural_trials if r["method"] == method),
            "recovery_rate_at_event_frame": round(recovery_rate, 3) if recovery_rate != "" else "",
            "identity_continuity_rate": round(identity_rate, 3) if identity_rate != "" else "",
        })

        ctrl_rows = [r for r in controlled_event_summary if r["method"] == method]
        n_ctrl_events = len(ctrl_rows)
        coverages = [r["window_coverage_rate"] for r in ctrl_rows if r["window_coverage_rate"] != ""]
        continuities = [r["post_window_id_continuity_rate"] for r in ctrl_rows if r["post_window_id_continuity_rate"] != ""]
        method_summary.append({
            "method": method, "evidence_type": "controlled_events",
            "n_events": n_ctrl_events, "n_rows": sum(1 for r in controlled_trials if r["method"] == method),
            "recovery_rate_at_event_frame": round(sum(coverages) / len(coverages), 3) if coverages else "",
            "identity_continuity_rate": round(sum(continuities) / len(continuities), 3) if continuities else "",
        })
    write_csv(OUT_ROOT / "summary_by_method.csv", method_summary)

    print("Timing each method separately (3 repeats per clip)...")
    runtime = measure_runtime_separately(dict(clip_frame_numbers), dict(detections_by_frame),
                                          dict(timestamps_by_clip), cfg)

    n_scenes = len({row["scene_name"] for row in manifest_rows})
    n_clips = len(clip_frame_numbers)
    n_unique_frames = len(manifest_rows)
    n_natural_events = len(natural_events)
    n_controlled_events = len(controlled_events)
    n_natural_instances = len({e["instance_token"] for e in natural_events})
    n_controlled_tracks = len({(e["clip"], e["natural_track_id"]) for e in controlled_events})

    run_metadata = {
        "config_hash": cfg_hash,
        "counts": {
            "scenes": n_scenes,
            "clips": n_clips,
            "unique_frames": n_unique_frames,
            "natural_events": n_natural_events,
            "natural_event_unique_instances": n_natural_instances,
            "controlled_events": n_controlled_events,
            "controlled_unique_target_tracks": n_controlled_tracks,
            "natural_trial_rows": len(natural_trials),
            "controlled_trial_rows": len(controlled_trials),
        },
        "note_on_repeated_rows": (
            f"{len(controlled_trials)} controlled trial rows come from only {n_controlled_events} events "
            f"on {n_controlled_tracks} unique target tracks -- rows are per-frame-per-method observations, "
            "not independent objects. Use the event or track as the experimental unit, never the row count."
        ),
        "runtime_ms_per_frame": runtime,
    }
    with open(OUT_ROOT / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Wrote run_metadata.json: {n_scenes} scenes, {n_clips} clips, {n_natural_events} natural events "
          f"({n_natural_instances} unique instances), {n_controlled_events} controlled events "
          f"({n_controlled_tracks} unique target tracks)")
    print(f"Runtime (median ms/frame): {json.dumps({k: v['median'] for k, v in runtime.items()})}")


if __name__ == "__main__":
    main()

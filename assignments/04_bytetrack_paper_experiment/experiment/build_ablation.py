"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 12: the required ablation (remove the second association) and the
required sensitivity check (vary high_score_threshold, development scenes
only, cheap cached tracking stage only -- no YOLO re-run).

Required ablation: SortTracker (Method B) already IS "ByteTrackTracker with
the second association removed" -- both share the identical Kalman filter,
first-association logic, new-track rule, and track_buffer (see
sort_tracker.py's docstring and reuse_audit.md). So the natural-event and
controlled-event SORT-vs-ByteTrack comparisons already computed in Task 11
directly answer this ablation; this script relabels and charts them as such,
rather than recomputing a third, redundant tracker.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from run_methods import new_bytetrack_tracker, new_sort_tracker  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
CHARTS_ROOT = OUT_ROOT / "charts"
DEV_CLIPS = {"clip_sample_001", "clip_sample_006"}
THRESHOLD_OFFSETS = [-0.15, 0.0, 0.15]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_ablation_csv_and_chart(cfg: dict) -> None:
    method_summary = load_csv(OUT_ROOT / "summary_by_method.csv")
    label_map = {"bytetrack": "bytetrack_full", "high_confidence_sort": "bytetrack_no_second_association"}

    rows = []
    for r in method_summary:
        rows.append({
            "variant": label_map[r["method"]],
            "evidence_type": r["evidence_type"],
            "n_events": r["n_events"],
            "n_rows": r["n_rows"],
            "recovery_or_coverage_rate": r["recovery_rate_at_event_frame"],
            "identity_continuity_rate": r["identity_continuity_rate"],
        })
    with open(OUT_ROOT / "ablation.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, evidence_type, title in (
        (axes[0], "natural_events", "Natural events (n=12)"),
        (axes[1], "controlled_events", "Controlled events (n=35)"),
    ):
        sub = [r for r in rows if r["evidence_type"] == evidence_type]
        variants = [r["variant"] for r in sub]
        recovery = [float(r["recovery_or_coverage_rate"]) for r in sub]
        identity = [float(r["identity_continuity_rate"]) for r in sub]
        x = range(len(sub))
        width = 0.35
        ax.bar([i - width / 2 for i in x], recovery, width, label="recovery/coverage rate")
        ax.bar([i + width / 2 for i in x], identity, width, label="identity continuity rate")
        ax.set_xticks(list(x))
        ax.set_xticklabels([v.replace("bytetrack_", "") for v in variants], rotation=15, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Ablation: full ByteTrack vs. second association removed\n"
                 "(higher is better; sample counts shown in ablation.csv)")
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "ablation.png", dpi=130)
    plt.close(fig)
    print("Wrote ablation.csv and ablation.png")


def run_threshold_sensitivity(cfg: dict) -> None:
    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")

    clip_frame_numbers = defaultdict(list)
    timestamps_by_clip = defaultdict(dict)
    for row in manifest_rows:
        if row["clip_name"] not in DEV_CLIPS:
            continue
        clip, frame_no = row["clip_name"], int(row["frame_number"])
        clip_frame_numbers[clip].append(frame_no)
        timestamps_by_clip[clip][frame_no] = float(row["timestamp"]) / 1_000_000.0
    for clip in clip_frame_numbers:
        clip_frame_numbers[clip].sort()

    detections_by_frame = defaultdict(list)
    for d in detections:
        if d["clip"] not in DEV_CLIPS:
            continue
        detections_by_frame[(d["clip"], int(d["frame_number"]))].append({
            "class": d["class"], "confidence": float(d["confidence"]),
            "x1": float(d["x1"]), "y1": float(d["y1"]), "x2": float(d["x2"]), "y2": float(d["y2"]),
        })

    base_threshold = cfg["tracker"]["high_score_threshold"]
    rows = []
    for offset in THRESHOLD_OFFSETS:
        threshold = round(base_threshold + offset, 3)
        cfg_variant = {**cfg, "tracker": {**cfg["tracker"], "high_score_threshold": threshold},
                       "sort_baseline": cfg["sort_baseline"]}

        for method_name, factory in (("high_confidence_sort", new_sort_tracker), ("bytetrack", new_bytetrack_tracker)):
            for clip in DEV_CLIPS:
                tracker = factory(cfg_variant)
                track_ids_seen = set()
                track_frame_counts = defaultdict(int)
                for frame_no in clip_frame_numbers[clip]:
                    dets = detections_by_frame.get((clip, frame_no), [])
                    outputs = tracker.update(dets, timestamp=timestamps_by_clip[clip][frame_no])
                    for o in outputs:
                        track_ids_seen.add(o.track_id)
                        track_frame_counts[o.track_id] += 1

                mean_track_length = (sum(track_frame_counts.values()) / len(track_frame_counts)
                                       if track_frame_counts else 0.0)
                rows.append({
                    "method": method_name, "clip": clip, "high_score_threshold": threshold,
                    "threshold_offset": offset, "n_tracks_born": len(track_ids_seen),
                    "mean_track_length_frames": round(mean_track_length, 2),
                })

    with open(OUT_ROOT / "threshold_sensitivity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for method_name, marker in (("high_confidence_sort", "o"), ("bytetrack", "s")):
        by_threshold = defaultdict(list)
        for r in rows:
            if r["method"] == method_name:
                by_threshold[r["threshold_offset"]].append(r)
        offsets_sorted = sorted(by_threshold.keys())
        mean_tracks = [sum(r["n_tracks_born"] for r in by_threshold[o]) / len(by_threshold[o]) for o in offsets_sorted]
        mean_lengths = [sum(r["mean_track_length_frames"] for r in by_threshold[o]) / len(by_threshold[o])
                         for o in offsets_sorted]
        axes[0].plot(offsets_sorted, mean_tracks, marker=marker, label=method_name)
        axes[1].plot(offsets_sorted, mean_lengths, marker=marker, label=method_name)

    axes[0].set_xlabel("high_score_threshold offset from frozen value (0.5)")
    axes[0].set_ylabel("mean tracks born per clip (lower = more stable identity)")
    axes[0].set_title("Track fragmentation vs. threshold")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("high_score_threshold offset from frozen value (0.5)")
    axes[1].set_ylabel("mean track length (frames)")
    axes[1].set_title("Track persistence vs. threshold")
    axes[1].legend(fontsize=8)
    fig.suptitle("Threshold sensitivity -- development scenes only, cached detections, no YOLO re-run")
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "threshold_sensitivity.png", dpi=130)
    plt.close(fig)
    print("Wrote threshold_sensitivity.csv and threshold_sensitivity.png")


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    build_ablation_csv_and_chart(cfg)
    run_threshold_sensitivity(cfg)


if __name__ == "__main__":
    main()

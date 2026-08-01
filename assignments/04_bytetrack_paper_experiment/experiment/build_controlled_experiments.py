"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 10 driver: selects controlled target tracks, builds the confidence-
demotion and complete-absence trials, and writes the manifest, per-frame
trial results, and methodology protocol.

Reproduction: `python build_controlled_experiments.py` (needs detections.csv
from run_detect.py).
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from controlled_trials import CONFIDENCE_DEMOTION, COMPLETE_ABSENCE, build_modified_detections, run_controlled_trial  # noqa: E402
from target_selection import build_natural_targets, select_eligible_targets  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"

MIN_TRACK_LENGTH = 14   # comfortably covers window<=3 (demotion) or <=7 (absence) plus lead-in/lookback
MIN_CONFIDENCE = 0.5
MAX_TARGETS = 12
SEED = 42
DEMOTION_WINDOWS = [1, 2, 3]
ABSENCE_WINDOWS = [1, 2, 3, 7]  # 7 > track_buffer (5): deliberately tests real expiry
DEMOTED_CONFIDENCE = 0.2  # inside [detection_floor, high_score_threshold)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")

    detections_by_frame = defaultdict(list)
    clip_frame_numbers = defaultdict(list)
    timestamps_by_clip: dict[str, dict[int, float]] = defaultdict(dict)
    for row in manifest_rows:
        clip, frame_no = row["clip_name"], int(row["frame_number"])
        clip_frame_numbers[clip].append(frame_no)
        timestamps_by_clip[clip][frame_no] = float(row["timestamp"]) / 1_000_000.0
    for clip in clip_frame_numbers:
        clip_frame_numbers[clip].sort()

    for d in detections:
        detections_by_frame[(d["clip"], int(d["frame_number"]))].append({
            "class": d["class"], "confidence": float(d["confidence"]),
            "x1": float(d["x1"]), "y1": float(d["y1"]), "x2": float(d["x2"]), "y2": float(d["y2"]),
        })

    print("Building natural targets from full clip detection streams...")
    targets = build_natural_targets(dict(detections_by_frame), dict(clip_frame_numbers), cfg)
    print(f"  {len(targets)} natural target segments found across {len(clip_frame_numbers)} clips.")

    selected, selection_log = select_eligible_targets(
        targets, min_track_length=MIN_TRACK_LENGTH, min_confidence=MIN_CONFIDENCE,
        max_targets=MAX_TARGETS, seed=SEED)

    if len(selected) < 3:
        print(f"ERROR: only {len(selected)} eligible targets found (need >= 3). Stopping.")
        with open(OUT_ROOT / "controlled_protocol.md", "w", encoding="utf-8") as f:
            f.write("# Controlled Experiment Protocol\n\nSTOPPED: fewer than 3 eligible targets found.\n\n")
            f.write("\n".join(selection_log))
        sys.exit(1)

    event_rows = []
    trial_rows = []

    for target in selected:
        n = len(target.frame_numbers)
        mid_idx = n // 2

        for mode, windows in ((CONFIDENCE_DEMOTION, DEMOTION_WINDOWS), (COMPLETE_ABSENCE, ABSENCE_WINDOWS)):
            for window_len in windows:
                start_idx = mid_idx - window_len // 2
                end_idx = start_idx + window_len  # exclusive
                if start_idx < 2 or end_idx > n - 2:
                    continue  # need at least 2 real frames of lead-in/lookback on each side
                window_frame_numbers = set(target.frame_numbers[start_idx:end_idx])

                modified = build_modified_detections(
                    dict(detections_by_frame), target.clip, target.frame_numbers, target.raw_boxes,
                    window_frame_numbers, mode, DEMOTED_CONFIDENCE)

                results = run_controlled_trial(
                    clip_frame_numbers[target.clip], modified, target.clip,
                    target.frame_numbers, target.raw_boxes, target.raw_confidences,
                    target.class_name, window_frame_numbers, cfg, timestamps_by_clip[target.clip])

                event_id = f"{target.clip}_track{target.track_id}_{mode}_w{window_len}"
                event_rows.append({
                    "event_id": event_id,
                    "clip": target.clip,
                    "natural_track_id": target.track_id,
                    "class_name": target.class_name,
                    "mode": mode,
                    "window_length": window_len,
                    "window_frame_numbers": ";".join(str(f) for f in sorted(window_frame_numbers)),
                    "target_total_frames": n,
                    "demoted_confidence": DEMOTED_CONFIDENCE if mode == CONFIDENCE_DEMOTION else "",
                })

                for method_name, rows in results.items():
                    for r in rows:
                        trial_rows.append({
                            "event_id": event_id,
                            "mode": mode,
                            "window_length": window_len,
                            "method": method_name,
                            **r,
                        })

    with open(OUT_ROOT / "controlled_event_manifest.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(event_rows[0].keys()) if event_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_rows)

    with open(OUT_ROOT / "controlled_trials.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(trial_rows[0].keys()) if trial_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trial_rows)

    n_demotion_events = sum(1 for e in event_rows if e["mode"] == CONFIDENCE_DEMOTION)
    n_absence_events = sum(1 for e in event_rows if e["mode"] == COMPLETE_ABSENCE)

    protocol_lines = [
        "# Controlled Experiment Protocol\n\n",
        "## Target selection\n\n",
        "\n".join(selection_log) + "\n\n",
        "## Experiment A -- confidence demotion\n\n",
        f"For each eligible target, a centered window of length in {DEMOTION_WINDOWS} frames is chosen "
        "(deterministically, from the middle of the target's natural detection span). The target's raw "
        "YOLO box is kept completely unchanged; only its confidence score is overwritten to "
        f"**{DEMOTED_CONFIDENCE}** (inside the low-confidence band) for exactly those frames. No other "
        "detection -- the target's own detections outside the window, or any other object's detections "
        "anywhere -- is touched. Both a fresh SortTracker and a fresh ByteTrackTracker are then run over "
        "the ENTIRE clip (every frame, in chronological order, real lifecycle enforced), and the tracker's "
        "output is compared against the target's ORIGINAL, undemoted raw YOLO box -- labeled "
        "**pseudo-ground-truth** throughout, never manually verified ground truth.\n\n"
        f"Windows tested: {n_demotion_events} demotion events across {len(selected)} targets.\n\n",
        "## Experiment B -- complete detection absence\n\n",
        f"Same target tracks, windows of length in {ABSENCE_WINDOWS} frames. The target's detection row is "
        "REMOVED ENTIRELY for those frames (not demoted) -- other objects' detections are left untouched, "
        "so ordinary false-association risk stays live. Window length 7 is deliberately longer than the "
        f"configured `track_buffer` ({cfg['tracker']['track_buffer']}) to test whether the track is "
        "genuinely allowed to expire, rather than surviving by construction as in Assignment 3.\n\n"
        f"Windows tested: {n_absence_events} absence events across {len(selected)} targets.\n\n",
        "## Identity measurement (repairs Assignment 3's hardcoded ID continuity)\n\n",
        "The target's track ID immediately before each window is found by matching tracker output boxes "
        "against the target's own known raw box (best IoU, same class) -- a legitimate, honest lookup, "
        "never a hardcoded assumption. `id_continuous_from_before_window` in controlled_trials.csv is only "
        "True when the SAME numeric track ID the real tracker assigned before the window is still present "
        "afterward -- if the track expired or a different track claims the location, this is correctly "
        "recorded as False, not silently assumed.\n\n",
        "## Repeated-row note\n\n",
        "Each (target, mode, window length) combination produces one row per evaluated frame per method. "
        "The main experimental unit for analysis is the (target, mode, window length) EVENT, not the frame "
        "row -- `summary_by_event.csv` and `summary_by_track.csv` (Task 11) report counts at both levels "
        "explicitly, never only the larger row count.\n",
    ]
    with open(OUT_ROOT / "controlled_protocol.md", "w", encoding="utf-8") as f:
        f.writelines(protocol_lines)

    print(f"Wrote controlled_event_manifest.csv ({len(event_rows)} events: "
          f"{n_demotion_events} demotion, {n_absence_events} absence)")
    print(f"Wrote controlled_trials.csv ({len(trial_rows)} rows)")
    print("Wrote controlled_protocol.md")


if __name__ == "__main__":
    main()

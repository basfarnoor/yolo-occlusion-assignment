"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 15: the student's chosen experiment -- "let lost tracks wait longer
before removal" -- raises track_buffer (and sort_baseline.max_age, kept equal
for a fair A/B comparison) from 5 to 10 frames. Everything else in
config.yaml stays fixed. Reruns only the cached-detections controlled-trial
and evaluation stage -- no YOLO re-run.
"""
from __future__ import annotations

import copy
import csv
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from controlled_trials import CONFIDENCE_DEMOTION, COMPLETE_ABSENCE, build_modified_detections, run_controlled_trial  # noqa: E402
from kalman_box_tracker import KalmanBoxTracker  # noqa: E402
from target_selection import build_natural_targets, select_eligible_targets  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"

MIN_TRACK_LENGTH = 14
MIN_CONFIDENCE = 0.5
MAX_TARGETS = 12
SEED = 42
NEW_TRACK_BUFFER = 10
OLD_TRACK_BUFFER = 5


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def post_window_id_continuity(trial_rows: list[dict], window_frame_numbers: set[int]) -> bool | None:
    post = sorted((r for r in trial_rows if r["frame_number"] not in window_frame_numbers
                   and r["frame_number"] > max(window_frame_numbers)), key=lambda r: r["frame_number"])
    if not post:
        return None
    return post[0]["id_continuous_from_before_window"]


def run_absence_w7_with_buffer(track_buffer: int, cfg_base: dict, detections_by_frame, clip_frame_numbers,
                                 timestamps_by_clip, selected_targets) -> dict:
    cfg = copy.deepcopy(cfg_base)
    cfg["tracker"]["track_buffer"] = track_buffer
    cfg["sort_baseline"]["max_age"] = track_buffer  # keep A/B comparison fair

    results = {}
    for target in selected_targets:
        n = len(target.frame_numbers)
        mid_idx = n // 2
        window_len = 7
        start_idx = mid_idx - window_len // 2
        end_idx = start_idx + window_len
        if start_idx < 2 or end_idx > n - 2:
            continue
        window_frame_numbers = set(target.frame_numbers[start_idx:end_idx])

        modified = build_modified_detections(
            dict(detections_by_frame), target.clip, target.frame_numbers, target.raw_boxes,
            window_frame_numbers, COMPLETE_ABSENCE, demoted_confidence=0.0)

        trial = run_controlled_trial(
            clip_frame_numbers[target.clip], modified, target.clip,
            target.frame_numbers, target.raw_boxes, target.raw_confidences,
            target.class_name, window_frame_numbers, cfg, timestamps_by_clip[target.clip])

        key = f"{target.clip}_track{target.track_id}"
        results[key] = {}
        for method, rows in trial.items():
            continuous = post_window_id_continuity(rows, window_frame_numbers)
            n_false_assoc = sum(1 for r in rows if r["frame_number"] in window_frame_numbers
                                  and r["evidence_source"] in ("high_score_detection", "low_score_detection")
                                  and isinstance(r["iou"], float) and r["iou"] < 0.3)
            results[key][method] = {"post_window_id_continuous": continuous, "false_associations_in_window": n_false_assoc}
    return results


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg_base = yaml.safe_load(f)

    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")

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

    KalmanBoxTracker.reset_id_counter()
    targets = build_natural_targets(dict(detections_by_frame), dict(clip_frame_numbers), cfg_base)
    selected, _ = select_eligible_targets(targets, MIN_TRACK_LENGTH, MIN_CONFIDENCE, MAX_TARGETS, SEED)

    print(f"Re-running window_length=7 complete-absence trials for {len(selected)} targets "
          f"at track_buffer={OLD_TRACK_BUFFER} (before) and track_buffer={NEW_TRACK_BUFFER} (after)...")

    before = run_absence_w7_with_buffer(OLD_TRACK_BUFFER, cfg_base, detections_by_frame, clip_frame_numbers,
                                          timestamps_by_clip, selected)
    after = run_absence_w7_with_buffer(NEW_TRACK_BUFFER, cfg_base, detections_by_frame, clip_frame_numbers,
                                         timestamps_by_clip, selected)

    lines = [
        "# Student Experiment\n\n",
        '**Student\'s exact choice:** "Let lost tracks wait longer before removal."\n\n',
        '**Student\'s prediction, in her own words:** "cost us" -- i.e. she predicted that letting '
        "tracks wait longer would cost more wrong reconnections/false associations, rather than being "
        "a clean improvement.\n\n",
        f"**Configuration change:** `track_buffer` raised from **{OLD_TRACK_BUFFER}** to "
        f"**{NEW_TRACK_BUFFER}** frames (and `sort_baseline.max_age` raised identically, to keep the "
        "SORT-vs-ByteTrack comparison fair -- both methods must share the same buffer).\n\n",
        "**What stayed fixed:** `detection_floor`, `high_score_threshold`, `new_track_threshold`, "
        "both IoU thresholds, the detector, the clips, and the 5 selected target tracks. Only the "
        "window-length-7 complete-absence trial was rerun (the case where the buffer boundary "
        "actually matters) -- cached detections were reused, no YOLO re-run.\n\n",
        "## Results before vs. after\n\n",
        "| Target track | Method | Reconnected (buffer=5) | Reconnected (buffer=10) | False associations in window (buffer=5) | False associations in window (buffer=10) |\n",
        "|---|---|---|---|---|---|\n",
    ]

    n_newly_reconnected = 0
    n_new_false_assoc = 0
    n_targets_compared = 0
    for key in sorted(before.keys()):
        for method in ("high_confidence_sort", "bytetrack"):
            b, a = before[key][method], after[key][method]
            lines.append(f"| `{key}` | {method} | {b['post_window_id_continuous']} | "
                           f"{a['post_window_id_continuous']} | {b['false_associations_in_window']} | "
                           f"{a['false_associations_in_window']} |\n")
            n_targets_compared += 1
            if (not b["post_window_id_continuous"]) and a["post_window_id_continuous"]:
                n_newly_reconnected += 1
            if a["false_associations_in_window"] > b["false_associations_in_window"]:
                n_new_false_assoc += 1

    lines.append(f"\n**{n_newly_reconnected}** (method, track) pairs that failed to reconnect at "
                 f"`track_buffer=5` succeeded at `track_buffer=10`.\n")
    lines.append(f"**{n_new_false_assoc}** (method, track) pairs picked up MORE in-window false "
                 f"associations at `track_buffer=10` than at `track_buffer=5`.\n\n")

    prediction_supported = n_new_false_assoc > 0
    lines.append("## Was the prediction supported?\n\n")
    if prediction_supported:
        lines.append(
            f"**Yes, partially.** The student predicted this change would \"cost us\" -- and "
            f"{n_new_false_assoc} case(s) did show more false associations at the longer buffer, "
            f"confirming her intuition that keeping a track alive longer on motion-guesswork alone "
            "creates more opportunity for it to grab the wrong object. At the same time, "
            f"{n_newly_reconnected} case(s) also gained a successful reconnection they didn't have "
            "before, so the change was not purely a cost -- it was a real trade-off, with her "
            "prediction correctly identifying the downside half of it.\n")
    else:
        lines.append(
            f"**Not in this sample.** No (method, track) pair picked up additional false associations "
            f"at `track_buffer=10` versus `track_buffer=5`, so the specific cost the student predicted "
            f"did not appear here -- though {n_newly_reconnected} case(s) did gain a successful "
            "reconnection. With only 5 target tracks, this should be read as 'no cost observed in this "
            "small sample,' not 'no cost exists.'\n")

    lines.append(
        "\n## Connection to the ByteTrack paper\n\n"
        "The paper's `track_buffer` (called the lost-track survival window) is exactly this "
        "parameter. The paper does not claim a longer buffer is free -- keeping a track alive longer "
        "without real evidence is a bet that the object will return before something else takes its "
        "place, and the student's prediction names precisely the risk the paper's track-state design "
        "(tracked/lost/removed) exists to manage.\n")

    with open(OUT_ROOT / "student_experiment.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote student_experiment.md ({n_targets_compared} (track, method) comparisons)")
    print(f"Newly reconnected: {n_newly_reconnected}, new false associations: {n_new_false_assoc}")


if __name__ == "__main__":
    main()

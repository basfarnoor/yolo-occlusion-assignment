"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Tasks 6-7: build natural tracks from the cached YOLO detections, select
eligible track segments deterministically (seed 42), artificially withhold
each one's detections for gap lengths 1/2/3/5, run all three baselines on
identical input, and save every trial's metrics.

Reproduction: `python run_experiment.py` (uses the cached detections.csv --
run_detect.py must have been run at least once first).
"""
from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from baselines import run_three_baselines  # noqa: E402
from geometry import center_error, iou  # noqa: E402
from track_selection import build_natural_tracks, select_eligible_tracks  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
IMAGE_WIDTH = 1600


def load_config() -> dict:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_detections() -> tuple[dict, dict]:
    detections_by_frame = defaultdict(list)
    clip_frame_numbers = defaultdict(set)
    with open(OUT_ROOT / "detections.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip = row["clip"]
            frame_no = int(row["frame_number"])
            clip_frame_numbers[clip].add(frame_no)
            detections_by_frame[(clip, frame_no)].append({
                "class": row["class"],
                "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })
    clip_frame_numbers = {clip: sorted(nums) for clip, nums in clip_frame_numbers.items()}
    return dict(detections_by_frame), clip_frame_numbers


ALLOWED_CLASSES = ("car", "truck", "bus", "person")


def run_trials(selected_tracks, gap_lengths: list[int]) -> list[dict]:
    trials = []
    for track in selected_tracks:
        n = len(track.frame_numbers)
        for gap_len in gap_lengths:
            # Need at least 1 real frame before and after the gap.
            if n < gap_len + 2:
                continue
            # Deterministic, centered gap placement.
            gap_start_idx = (n - gap_len) // 2
            if gap_start_idx <= 0 or gap_start_idx + gap_len >= n:
                continue

            withheld_boxes = track.boxes[gap_start_idx: gap_start_idx + gap_len]
            withheld_frame_numbers = track.frame_numbers[gap_start_idx: gap_start_idx + gap_len]

            t0 = time.perf_counter()
            results = run_three_baselines(track.boxes, track.class_name, gap_start_idx, gap_len)
            elapsed_ms = (time.perf_counter() - t0) * 1000 / max(1, gap_len)

            reappearance_track_id_before = track.track_id
            reappearance_track_id_after = track.track_id  # single-track simulation: ID is always continuous

            for offset, r in enumerate(results):
                pseudo_gt_box = withheld_boxes[offset]
                frame_no = withheld_frame_numbers[offset]

                for method, box in (("yolo_only", r.yolo_only_box),
                                      ("static_memory", r.static_memory_box),
                                      ("sort_motion", r.sort_box)):
                    has_box = box is not None
                    trials.append({
                        "clip": track.clip,
                        "track_id": track.track_id,
                        "class_name": track.class_name,
                        "gap_length": gap_len,
                        "frame_offset_in_gap": offset,
                        "frame_number": frame_no,
                        "method": method,
                        "has_box": has_box,
                        "center_error_px": center_error(box, pseudo_gt_box) if has_box else "",
                        "center_error_pct_width": (center_error(box, pseudo_gt_box) / IMAGE_WIDTH * 100) if has_box else "",
                        "iou": iou(box, pseudo_gt_box) if has_box else 0.0,
                        "iou_at_least_030": (iou(box, pseudo_gt_box) >= 0.30) if has_box else False,
                        "track_id_before_gap": reappearance_track_id_before,
                        "track_id_after_gap": reappearance_track_id_after,
                        "id_continuous": reappearance_track_id_before == reappearance_track_id_after,
                        "prediction_time_ms": round(elapsed_ms, 4),
                    })
    return trials


def main() -> None:
    cfg = load_config()
    detections_by_frame, clip_frame_numbers = load_detections()

    print("Building natural tracks from cached detections...")
    natural_tracks = build_natural_tracks(detections_by_frame, clip_frame_numbers)
    print(f"  {len(natural_tracks)} natural track segments found across "
          f"{len(clip_frame_numbers)} clips.")

    occ_cfg = cfg["occlusion_experiment"]
    selected, selection_log = select_eligible_tracks(
        natural_tracks,
        min_track_length=occ_cfg["min_track_length"],
        min_track_length_floor=occ_cfg["min_track_length_floor"],
        max_eligible_tracks=occ_cfg["max_eligible_tracks"],
        min_confidence=occ_cfg["min_confidence"],
        min_displacement_px=occ_cfg["min_displacement_px"],
        allowed_classes=ALLOWED_CLASSES,
        seed=occ_cfg["random_seed"],
    )

    log_path = OUT_ROOT / "track_selection_log.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# Automatic Track Selection Log\n\n")
        f.write("Rules applied (see `experiment/config.yaml`):\n\n")
        f.write(f"- Allowed classes: {ALLOWED_CLASSES}\n")
        f.write(f"- Minimum track length: {occ_cfg['min_track_length']} frames "
                f"(relaxable down to {occ_cfg['min_track_length_floor']})\n")
        f.write(f"- Minimum average confidence: {occ_cfg['min_confidence']}\n")
        f.write(f"- Minimum displacement: {occ_cfg['min_displacement_px']}px\n")
        f.write(f"- Must not begin/end within {25}px of the image boundary\n")
        f.write(f"- Maximum eligible tracks kept: {occ_cfg['max_eligible_tracks']} (seed {occ_cfg['random_seed']})\n\n")
        f.write("## Log\n\n")
        for line in selection_log:
            f.write(line + "\n")
    print(f"Wrote {log_path}")

    if len(selected) < 3:
        print(f"ERROR: only {len(selected)} eligible tracks found (need at least 3). "
              f"Stopping before running the experiment -- see {log_path} for details.")
        sys.exit(1)

    print(f"Selected {len(selected)} track segment(s) for the occlusion experiment.")

    gap_lengths = occ_cfg["gap_lengths"]
    trials = run_trials(selected, gap_lengths)

    trials_path = OUT_ROOT / "trials.csv"
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(trials[0].keys()) if trials else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)
    print(f"Wrote {trials_path} ({len(trials)} trial rows).")


if __name__ == "__main__":
    main()

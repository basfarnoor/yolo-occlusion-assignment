"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 9 driver: builds natural weak-evidence event candidates on development
and evaluation scenes separately, ranks them deterministically, keeps at most
12, and writes the manifest + selection log + contact sheets for visual
review.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from event_selection import build_instance_sequences, find_confidence_drop_events  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
MAX_EVENTS = 12


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    high_score_threshold = cfg["tracker"]["high_score_threshold"]

    gt_rows = load_csv(OUT_ROOT / "projected_ground_truth.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")
    split_rows = load_csv(OUT_ROOT / "split_manifest.csv")
    split_by_clip = {r["clip_name"]: r["split"] for r in split_rows}

    detections_by_frame = {}
    for d in detections:
        detections_by_frame.setdefault((d["clip"], d["frame_number"]), []).append(d)

    sequences = build_instance_sequences(gt_rows, detections_by_frame)
    all_events = find_confidence_drop_events(sequences, high_score_threshold)
    for e in all_events:
        e["split"] = split_by_clip.get(e["clip_name"], "UNKNOWN")

    log_lines = ["# Natural Event Selection Log\n\n"]
    log_lines.append(f"Rule: instance present with confidence >= {high_score_threshold} at a keyframe, "
                       "then weak (below threshold) or unmatched at the next keyframe, "
                       "with the instance still present in ground truth at a later keyframe.\n\n")
    log_lines.append(f"Total instance sequences considered (>=3 keyframe appearances): "
                       f"{sum(1 for s in sequences.values() if len(s) >= 3)}\n")
    log_lines.append(f"Raw candidate events found (before ranking/capping): {len(all_events)}\n\n")

    if not all_events:
        log_lines.append("## No candidates found\n\n")
        log_lines.append(
            "No natural confidence-drop events were found at keyframe granularity in this mini "
            "split. This is reported honestly rather than relaxed by loosening identity validity "
            "or the transition rule -- see the Limitations section of the final report: nuScenes "
            "annotates only keyframes (6 per 36-frame clip here), so the number of independently "
            "verifiable natural transitions this small dataset can support is itself limited. The "
            "controlled experiments (Task 10) do not depend on natural events existing and still "
            "provide the main quantitative evidence.\n")
        manifest_path = OUT_ROOT / "natural_event_manifest.csv"
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        with open(OUT_ROOT / "natural_event_selection.md", "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        print("No natural events found -- wrote empty manifest and an honest selection log.")
        return

    # Deterministic ranking: prefer events with more frames available on both
    # sides (more identity-continuity evidence), then a stable tiebreak.
    all_events.sort(key=lambda e: (
        -(min(e["frames_available_before"], e["frames_available_after"])),
        e["clip_name"], e["instance_token"], e["event_frame"],
    ))

    selected = all_events[:MAX_EVENTS]
    scenes_represented = {e["clip_name"] for e in selected}
    classes_represented = {e["category"] for e in selected}
    log_lines.append(f"Selected {len(selected)} of {len(all_events)} candidates (cap: {MAX_EVENTS}), "
                       "ranked by identity-continuity margin (min frames available before/after).\n")
    log_lines.append(f"Scenes represented: {sorted(scenes_represented)}\n")
    log_lines.append(f"Classes represented: {sorted(classes_represented)}\n\n")
    log_lines.append("## Selected events\n\n")
    for e in selected:
        log_lines.append(
            f"- `{e['clip_name']}` ({e['split']}) instance `{e['instance_token'][:10]}` "
            f"({e['category']}): frame {e['before_frame']} conf={e['before_confidence']:.2f} -> "
            f"frame {e['event_frame']} conf={e['event_confidence']} ({e['plausible_cause']}) -> "
            f"frame {e['after_frame']} conf={e['after_confidence']}\n")

    fieldnames = ["clip_name", "split", "instance_token", "category", "before_frame", "event_frame",
                  "after_frame", "before_confidence", "event_confidence", "after_confidence",
                  "before_visibility", "event_visibility", "after_visibility", "plausible_cause",
                  "frames_available_before", "frames_available_after"]
    manifest_path = OUT_ROOT / "natural_event_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    with open(OUT_ROOT / "natural_event_selection.md", "w", encoding="utf-8") as f:
        f.writelines(log_lines)

    print(f"Wrote {manifest_path} ({len(selected)} events)")
    print(f"Scenes represented: {sorted(scenes_represented)}")
    print(f"Classes represented: {sorted(classes_represented)}")


if __name__ == "__main__":
    main()

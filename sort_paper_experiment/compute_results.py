"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 7: turn trials.csv into summary.csv (grouped by method / gap length /
class / clip) and run_metadata.json (including the required cross-method
comparisons). Every number here is traceable back to a trials.csv row.
"""
from __future__ import annotations

import csv
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from evaluation import group_summary  # noqa: E402

import ultralytics  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "sort_paper_experiment"
TRIALS_CSV = OUT_ROOT / "trials.csv"


def load_trials() -> list[dict]:
    with open(TRIALS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gap_length"] = int(r["gap_length"])
        r["frame_offset_in_gap"] = int(r["frame_offset_in_gap"])
        r["has_box"] = r["has_box"] == "True"
        r["iou_at_least_030"] = r["iou_at_least_030"] == "True"
        r["id_continuous"] = r["id_continuous"] == "True"
    return rows


def write_summary(trials: list[dict]) -> None:
    rows = []
    for group_type, keys in (
        ("method_x_gap", ["method", "gap_length"]),
        ("method_x_class", ["method", "class_name"]),
        ("method_x_clip", ["method", "clip"]),
    ):
        for row in group_summary(trials, keys):
            row["group_type"] = group_type
            rows.append(row)

    all_keys = ["group_type", "method", "gap_length", "class_name", "clip", "n_trials",
                "coverage_rate", "iou_at_least_030_rate", "id_continuity_rate",
                "center_error_mean_px", "center_error_median_px", "center_error_std_px", "center_error_n",
                "iou_mean", "iou_median", "iou_std", "iou_n",
                "prediction_time_ms_mean", "prediction_time_ms_n"]
    summary_path = OUT_ROOT / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})
    print(f"Wrote {summary_path} ({len(rows)} grouped rows)")


def required_comparisons(trials: list[dict]) -> dict:
    by_pair_key = defaultdict(dict)  # (clip, track_id, gap_length, frame_offset) -> {method: trial}
    for t in trials:
        key = (t["clip"], t["track_id"], t["gap_length"], t["frame_offset_in_gap"])
        by_pair_key[key][t["method"]] = t

    iou_diff_by_gap = defaultdict(list)
    # Center-error % change uses ratio-of-means (aggregate mean center error for each
    # method, then compare those two means), not a mean of per-trial ratios: many of
    # these tracks barely move, so a per-trial percentage swings wildly around a
    # near-zero denominator and the average of those ratios is dominated by noise
    # rather than the real, large improvements on the tracks that do move a lot.
    static_ce_by_gap = defaultdict(list)
    sort_ce_by_gap = defaultdict(list)
    for key, methods in by_pair_key.items():
        gap_length = key[2]
        if "sort_motion" in methods and "static_memory" in methods:
            iou_diff = float(methods["sort_motion"]["iou"]) - float(methods["static_memory"]["iou"])
            iou_diff_by_gap[gap_length].append(iou_diff)

            static_ce = methods["static_memory"]["center_error_px"]
            sort_ce = methods["sort_motion"]["center_error_px"]
            if static_ce not in ("", None) and sort_ce not in ("", None):
                static_ce_by_gap[gap_length].append(float(static_ce))
                sort_ce_by_gap[gap_length].append(float(sort_ce))

    def mean(values):
        return round(sum(values) / len(values), 4) if values else None

    center_pct_change_by_gap = {}
    for gap_length in static_ce_by_gap:
        static_mean = mean(static_ce_by_gap[gap_length])
        sort_mean = mean(sort_ce_by_gap[gap_length])
        pct_change = ((sort_mean - static_mean) / static_mean * 100) if static_mean else None
        center_pct_change_by_gap[gap_length] = {
            "static_memory_mean_px": static_mean,
            "sort_motion_mean_px": sort_mean,
            "pct_change_ratio_of_means": round(pct_change, 2) if pct_change is not None else None,
            "n_pairs": len(static_ce_by_gap[gap_length]),
        }

    coverage_by_method = defaultdict(list)
    id_continuity_by_method = defaultdict(list)
    for t in trials:
        coverage_by_method[t["method"]].append(1 if t["has_box"] else 0)
        id_continuity_by_method[t["method"]].append(1 if t["id_continuous"] else 0)

    gap_trend = {}
    for gap_len in sorted({t["gap_length"] for t in trials}):
        subset = [t for t in trials if t["gap_length"] == gap_len]
        by_method = defaultdict(list)
        for t in subset:
            if t["has_box"]:
                by_method[t["method"]].append(float(t["center_error_px"]))
        gap_trend[gap_len] = {
            method: {"mean_center_error_px": mean(vals), "n": len(vals)}
            for method, vals in by_method.items()
        }

    return {
        "sort_iou_minus_static_iou_by_gap": {
            g: {"mean_diff": mean(vals), "n_pairs": len(vals)} for g, vals in iou_diff_by_gap.items()
        },
        "pct_change_center_error_sort_vs_static_by_gap": center_pct_change_by_gap,
        "prediction_coverage_by_method": {
            m: {"coverage_rate": mean(vals), "n": len(vals)} for m, vals in coverage_by_method.items()
        },
        "id_continuity_by_method": {
            m: {"continuity_rate": mean(vals), "n": len(vals)} for m, vals in id_continuity_by_method.items()
        },
        "metrics_by_gap_length": gap_trend,
    }


def main() -> None:
    trials = load_trials()
    write_summary(trials)

    comparisons = required_comparisons(trials)

    metadata = {
        "model_name": "yolo26n.pt",
        "ultralytics_version": ultralytics.__version__,
        "python_version": platform.python_version(),
        "device": "cpu",
        "n_trials": len(trials),
        "n_clips": len({t["clip"] for t in trials}),
        "n_tracks": len({(t["clip"], t["track_id"]) for t in trials}),
        "gap_lengths_tested": sorted({t["gap_length"] for t in trials}),
        "required_comparisons": comparisons,
    }
    metadata_path = OUT_ROOT / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()

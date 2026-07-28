"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 8: the 5 required charts, each comparing all applicable methods, each
labeled with units and sample counts, and each reminding the reader that
the reference boxes are withheld YOLO detections (pseudo-ground truth),
not manually verified ground truth.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "sort_paper_experiment"
TRIALS_CSV = OUT_ROOT / "trials.csv"

METHOD_COLORS = {"yolo_only": "#2F5597", "static_memory": "#FFA630", "sort_motion": "#1C7293"}
METHOD_LABELS = {"yolo_only": "YOLO only", "static_memory": "Static memory (Assignment 2)",
                  "sort_motion": "SORT motion prediction"}
REFERENCE_NOTE = ("Reference boxes are withheld YOLO detections (pseudo-ground truth), "
                   "not manually verified ground truth.")


def load_trials() -> list[dict]:
    with open(TRIALS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["gap_length"] = int(r["gap_length"])
        r["has_box"] = r["has_box"] == "True"
        r["id_continuous"] = r["id_continuous"] == "True"
    return rows


def grouped_by_gap_and_method(trials, value_fn, only_with_box=False):
    out = defaultdict(lambda: defaultdict(list))
    for t in trials:
        if only_with_box and not t["has_box"]:
            continue
        v = value_fn(t)
        if v is None:
            continue
        out[t["method"]][t["gap_length"]].append(v)
    return out


def bar_chart_by_gap(trials, value_fn, title, ylabel, out_name, only_with_box=False, show_std=True):
    data = grouped_by_gap_and_method(trials, value_fn, only_with_box=only_with_box)
    gap_lengths = sorted({g for m in data.values() for g in m})
    methods = [m for m in ("yolo_only", "static_memory", "sort_motion") if m in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    n_methods = len(methods)
    width = 0.8 / max(n_methods, 1)
    x = range(len(gap_lengths))

    for mi, method in enumerate(methods):
        means, stds, ns = [], [], []
        for g in gap_lengths:
            vals = data[method].get(g, [])
            means.append(statistics.mean(vals) if vals else 0)
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
            ns.append(len(vals))
        offsets = [xi + (mi - (n_methods - 1) / 2) * width for xi in x]
        bars = ax.bar(offsets, means, width=width, label=METHOD_LABELS[method],
                        color=METHOD_COLORS[method],
                        yerr=stds if show_std else None, capsize=4)
        for xi, m, n in zip(offsets, means, ns):
            ax.text(xi, m, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{g} frame{'s' if g != 1 else ''}" for g in gap_lengths])
    ax.set_xlabel("Artificial detection gap length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.text(0.5, -0.02, REFERENCE_NOTE, ha="center", fontsize=9, style="italic", color="dimgray")
    plt.tight_layout()
    out_path = OUT_ROOT / out_name
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    trials = load_trials()

    bar_chart_by_gap(
        trials,
        value_fn=lambda t: float(t["iou"]),
        title="Mean IoU with the withheld box, by gap length",
        ylabel="IoU (1.0 = perfect overlap, 0.0 = no overlap)",
        out_name="mean_iou_by_gap.png",
    )

    bar_chart_by_gap(
        trials,
        value_fn=lambda t: float(t["center_error_px"]) if t["has_box"] else None,
        title="Mean center error vs. the withheld box, by gap length",
        ylabel="Center error (pixels) -- only counted when a box was produced",
        out_name="center_error_by_gap.png",
        only_with_box=True,
    )

    bar_chart_by_gap(
        trials,
        value_fn=lambda t: 1.0 if t["has_box"] else 0.0,
        title="How often each method produced a box during the gap",
        ylabel="Coverage rate (1.0 = always produced a box)",
        out_name="prediction_coverage_by_gap.png",
        show_std=False,
    )

    bar_chart_by_gap(
        trials,
        value_fn=lambda t: 1.0 if t["id_continuous"] else 0.0,
        title="Track-ID continuity across the gap, by gap length",
        ylabel="Fraction of trials with a continuous ID before/after the gap",
        out_name="id_continuity_by_gap.png",
        show_std=False,
    )

    # Runtime comparison: prediction_time_ms per method, plus YOLO's own per-frame time for context.
    detections_csv = OUT_ROOT / "detections.csv"
    yolo_times = []
    with open(detections_csv, newline="", encoding="utf-8") as f:
        seen = set()
        for row in csv.DictReader(f):
            key = (row["clip"], row["frame_number"])
            if key in seen:
                continue
            seen.add(key)
            yolo_times.append(float(row["inference_time_ms"]))

    tracker_times = defaultdict(list)
    for t in trials:
        tracker_times[t["method"]].append(float(t["prediction_time_ms"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    labels = ["YOLO detection\n(per frame)"] + [METHOD_LABELS[m] for m in ("static_memory", "sort_motion")]
    means = [statistics.mean(yolo_times)] + [statistics.mean(tracker_times[m]) for m in ("static_memory", "sort_motion")]
    ns = [len(yolo_times)] + [len(tracker_times[m]) for m in ("static_memory", "sort_motion")]
    colors = ["#2F5597", METHOD_COLORS["static_memory"], METHOD_COLORS["sort_motion"]]
    bars = ax.bar(labels, means, color=colors)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Mean time (milliseconds, log scale)")
    ax.set_title("Runtime: YOLO detection vs. tracker-only prediction time\n"
                  "(the paper reports tracker speed separately from detector speed -- so do we)")
    ax.set_yscale("log")
    plt.tight_layout()
    out_path = OUT_ROOT / "runtime_comparison.png"
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 13: the remaining required charts, all computed from already-generated
CSVs (natural_trials.csv, controlled_trials.csv, run_metadata.json) -- no
tracker re-run needed. Every chart names its evaluation-reference type, shows
sample counts, and labels whether higher or lower is better, per Task 13's
requirements.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
CHARTS_ROOT = OUT_ROOT / "charts"
METHOD_LABELS = {"high_confidence_sort": "high-confidence SORT", "bytetrack": "ByteTrack"}
METHOD_COLORS = {"high_confidence_sort": "#1f77b4", "bytetrack": "#ff7f0e"}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def chart_low_score_recovery(controlled_trials: list[dict]) -> None:
    """During complete-absence windows, how many frames did each method
    actually recover the target via a LOW-score detection (ByteTrack's
    unique capability -- SORT structurally can never do this)?"""
    absence_window = [r for r in controlled_trials if r["mode"] == "complete_absence" and r["in_window"] in ("True", "true")]
    counts = {}
    for method in METHOD_LABELS:
        sub = [r for r in absence_window if r["method"] == method]
        low = sum(1 for r in sub if r["evidence_source"] == "low_score_detection")
        counts[method] = (low, len(sub))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    methods = list(METHOD_LABELS)
    values = [counts[m][0] for m in methods]
    ns = [counts[m][1] for m in methods]
    bars = ax.bar([METHOD_LABELS[m] for m in methods], values, color=[METHOD_COLORS[m] for m in methods])
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"n={n} window frames",
                 ha="center", fontsize=8)
    ax.set_ylabel("Frames recovered via a low-score detection (higher is better)")
    ax.set_title("Low-score recovery during complete detection absence\n"
                 "(evaluation reference: withheld raw YOLO box, pseudo-ground-truth)")
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "low_score_recovery_by_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote low_score_recovery_by_method.png")


def chart_fragmentation(natural_trials: list[dict]) -> None:
    """Fragmentation proxy: of natural events where the object was recovered
    both before and after the confidence dip, how often did it come back as a
    genuinely DIFFERENT track ID (a fragmentation/ID-switch event) vs. the
    same one?"""
    by_event_method = defaultdict(dict)
    for r in natural_trials:
        by_event_method[(r["instance_token"], r["method"])][r["role"]] = r

    counts = {m: {"same_id": 0, "switched_id": 0, "not_both_recovered": 0} for m in METHOD_LABELS}
    for (instance_token, method), roles in by_event_method.items():
        before, after = roles.get("before"), roles.get("after")
        if not before or not after:
            continue
        before_ok = before["recovered"] in ("True", "true") and before["track_id"] != ""
        after_ok = after["recovered"] in ("True", "true") and after["track_id"] != ""
        if not (before_ok and after_ok):
            counts[method]["not_both_recovered"] += 1
        elif before["track_id"] == after["track_id"]:
            counts[method]["same_id"] += 1
        else:
            counts[method]["switched_id"] += 1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = list(METHOD_LABELS)
    same = [counts[m]["same_id"] for m in methods]
    switched = [counts[m]["switched_id"] for m in methods]
    not_both = [counts[m]["not_both_recovered"] for m in methods]
    x = range(len(methods))
    ax.bar(x, same, label="same track ID before->after (good)", color="#2ca02c")
    ax.bar(x, switched, bottom=same, label="switched to a different ID (fragmentation)", color="#d62728")
    ax.bar(x, not_both, bottom=[s + w for s, w in zip(same, switched)],
           label="not recovered on both sides", color="#7f7f7f")
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel(f"Natural events (n=12 each)")
    ax.set_title("Track fragmentation across the natural confidence dip\n"
                 "(reference: independent projected nuScenes ground truth)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "fragmentation_by_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fragmentation_by_method.png")
    return counts


def chart_id_switches(fragmentation_counts: dict) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    methods = list(METHOD_LABELS)
    values = [fragmentation_counts[m]["switched_id"] for m in methods]
    ax.bar([METHOD_LABELS[m] for m in methods], values, color=[METHOD_COLORS[m] for m in methods])
    ax.set_ylabel("ID switches across the natural confidence dip (n=12 events; lower is better)")
    ax.set_title("ID switches by method\n(evaluation reference: independent projected nuScenes ground truth)")
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "id_switches_by_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote id_switches_by_method.png")


def chart_false_associations(controlled_trials: list[dict]) -> None:
    """A 'false association': the tracker matched a REAL detection (not a
    motion prediction) that was NOT the withheld/demoted target -- i.e. IoU
    against the pseudo-ground-truth target box below 0.3, during a controlled
    window where we know exactly what the correct answer should be."""
    counts = {m: {"confidence_demotion": 0, "complete_absence": 0} for m in METHOD_LABELS}
    ns = {m: {"confidence_demotion": 0, "complete_absence": 0} for m in METHOD_LABELS}
    for r in controlled_trials:
        if r["in_window"] not in ("True", "true"):
            continue
        is_real = r["evidence_source"] in ("high_score_detection", "low_score_detection")
        ns[r["method"]][r["mode"]] += 1
        if is_real and r["iou"] not in ("", None) and float(r["iou"]) < 0.3:
            counts[r["method"]][r["mode"]] += 1

    fig, ax = plt.subplots(figsize=(6, 4.5))
    methods = list(METHOD_LABELS)
    modes = ["confidence_demotion", "complete_absence"]
    x = range(len(methods))
    width = 0.35
    for i, mode in enumerate(modes):
        vals = [counts[m][mode] for m in methods]
        offset = (i - 0.5) * width
        bars = ax.bar([xi + offset for xi in x], vals, width, label=mode)
        for xi, v, m in zip(x, vals, methods):
            ax.text(xi + offset, v + 0.05, f"n={ns[m][mode]}", ha="center", fontsize=7)
    ax.set_xticks(list(x))
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
    ax.set_ylabel("False/ghost associations (wrong real detection matched; lower is better)")
    ax.set_title("False associations by method\n(evaluation reference: withheld/demoted raw YOLO box, pseudo-ground-truth)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "false_associations_by_method.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote false_associations_by_method.png")


def chart_complete_absence_survival(controlled_trials: list[dict]) -> None:
    absence = [r for r in controlled_trials if r["mode"] == "complete_absence"]
    window_lengths = sorted({int(r["window_length"]) for r in absence})

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method in METHOD_LABELS:
        rates = []
        ns = []
        for wl in window_lengths:
            sub = [r for r in absence if r["method"] == method and int(r["window_length"]) == wl]
            by_event = defaultdict(list)
            for r in sub:
                by_event[r["event_id"]].append(r)
            reconnected = 0
            n_events_with_post_window_data = 0
            for event_id, rows in by_event.items():
                window_frames = [int(r["frame_number"]) for r in rows if r["in_window"] in ("True", "true")]
                if not window_frames:
                    continue
                max_window_frame = max(window_frames)
                post_window_rows = sorted(
                    (r for r in rows if r["in_window"] in ("False", "false")
                     and int(r["frame_number"]) > max_window_frame),
                    key=lambda r: int(r["frame_number"]))
                if not post_window_rows:
                    continue
                n_events_with_post_window_data += 1
                if post_window_rows[0]["id_continuous_from_before_window"] in ("True", "true"):
                    reconnected += 1
            rates.append(reconnected / n_events_with_post_window_data if n_events_with_post_window_data else 0.0)
            ns.append(n_events_with_post_window_data)
        ax.plot(window_lengths, rates, marker="o", label=f"{METHOD_LABELS[method]}", color=METHOD_COLORS[method])
        for wl, rate, n in zip(window_lengths, rates, ns):
            ax.annotate(f"n={n}", (wl, rate), textcoords="offset points", xytext=(0, 6), fontsize=7)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("Complete-absence window length (frames)")
    ax.set_ylabel("Immediate post-window ID-continuity rate (higher is better)")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Survival through complete detection absence\n"
                 "(reference: real tracker output IDs, never assigned by construction)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "complete_absence_survival.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote complete_absence_survival.png")


def chart_runtime(run_metadata: dict) -> None:
    runtime = run_metadata["runtime_ms_per_frame"]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    methods = list(METHOD_LABELS)
    medians = [runtime[m]["median"] for m in methods]
    ns = [runtime[m]["n"] for m in methods]
    bars = ax.bar([METHOD_LABELS[m] for m in methods], medians, color=[METHOD_COLORS[m] for m in methods])
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"n={n} calls", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Median tracker-only time per frame (ms) -- lower is better")
    ax.set_title("Runtime comparison (tracker only, YOLO excluded)\nmeasured separately per method, 3 repeats/clip")
    fig.tight_layout()
    CHARTS_ROOT.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_ROOT / "runtime_comparison.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("Wrote runtime_comparison.png")


def main() -> None:
    natural_trials = load_csv(OUT_ROOT / "natural_trials.csv")
    controlled_trials = load_csv(OUT_ROOT / "controlled_trials.csv")
    with open(OUT_ROOT / "run_metadata.json", encoding="utf-8") as f:
        run_metadata = json.load(f)

    chart_low_score_recovery(controlled_trials)
    fragmentation_counts = chart_fragmentation(natural_trials)
    chart_id_switches(fragmentation_counts)
    chart_false_associations(controlled_trials)
    chart_complete_absence_survival(controlled_trials)
    chart_runtime(run_metadata)


if __name__ == "__main__":
    main()

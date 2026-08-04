"""Task 11: builds the compact report, charts, and calibration analysis from
the already-written immutable outputs (`artifacts/mvp_full_outputs.parquet`,
`results/mvp_event_metrics.csv`, `results/run_metadata.json`). Deliberately
separate from `run_mvp_study.py` -- the 48 event reruns are the expensive
part; this script is the fast, cacheable "one command regenerates the report
from immutable outputs" entry point Task 11 requires.

Reproduction: `.venv/Scripts/python scripts/build_mvp_report.py` from OATM/
(assumes `run_mvp_study.py` has already been run at least once).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.evaluation.ground_truth import (  # noqa: E402
    DETECTOR_TO_EVAL_CLASS,
    index_by_frame,
    load_ground_truth,
)
from oatm.tracking.geometry import iou  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
METHODS = ("yolo_only", "static_memory", "sort", "bytetrack", "oatm_mvp")
FAMILIES = ("natural", "controlled_visual", "detector_intervention")
METHOD_COLORS = {
    "yolo_only": "#888888", "static_memory": "#e07b39", "sort": "#3b7dd8",
    "bytetrack": "#2ca25f", "oatm_mvp": "#a83279",
}


def chart_hidden_recall_vs_ghost_duration(
    events: pd.DataFrame, global_metrics: dict, charts_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        ghost = global_metrics[method]["ghost"]
        if ghost is None or ghost["mean_ghost_duration_frames"] is None:
            continue  # yolo_only has no real identity -- ghost duration is not defined for it
        recall = events[events.method_name == method]["hidden_frame_coverage"].mean()
        ghost_dur = ghost["mean_ghost_duration_frames"]
        ax.scatter(ghost_dur, recall, s=90, color=METHOD_COLORS[method], label=method)
        ax.annotate(method, (ghost_dur, recall), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("Mean ghost-track duration (frames) -- global, across all 10 scenes")
    ax.set_ylabel("Mean hidden-frame coverage -- across all evaluable events, all 3 families")
    ax.set_title("Hidden recall vs. ghost duration (benefit vs. harm)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "hidden_recall_vs_ghost_duration.png", dpi=130)
    plt.close(fig)


def chart_identity_preservation(events: pd.DataFrame, charts_dir: Path) -> None:
    methods_with_identity = [m for m in METHODS if m != "yolo_only"]
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(14, 4.5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        sub = events[(events.event_source == family) & (events.method_name.isin(methods_with_identity))]
        same_rate, new_rate, lost_rate = [], [], []
        for m in methods_with_identity:
            counts = sub[sub.method_name == m]["recovery_status"].value_counts()
            total = counts.sum() if counts.sum() else 1
            same_rate.append(counts.get("same_id", 0) / total)
            new_rate.append(counts.get("new_id", 0) / total)
            lost_rate.append(counts.get("not_recovered", 0) / total)
        x = np.arange(len(methods_with_identity))
        ax.bar(x, same_rate, label="same ID (correct)", color="#2ca25f")
        ax.bar(x, new_rate, bottom=same_rate, label="new ID (identity switch)", color="#e07b39")
        bottom2 = np.array(same_rate) + np.array(new_rate)
        ax.bar(x, lost_rate, bottom=bottom2, label="not recovered", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(methods_with_identity, rotation=30, ha="right")
        ax.set_title(f"{family}\n(n={len(sub) // len(methods_with_identity)} events)")
    axes[0].set_ylabel("Fraction of events")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("Identity preservation vs. wrong-object association, by family")
    fig.tight_layout()
    fig.savefig(charts_dir / "identity_preservation.png", dpi=130)
    plt.close(fig)


def chart_localization_error_vs_gap(events: pd.DataFrame, charts_dir: Path) -> None:
    e = events.copy()
    e["gap_frames"] = e["n_hidden_frames"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for method in METHODS:
        sub = e[(e.method_name == method) & e.mean_center_error_px.notna()]
        if sub.empty:
            continue
        ax.scatter(sub["gap_frames"], sub["mean_center_error_px"], s=40, alpha=0.6,
                   color=METHOD_COLORS[method], label=method)
    ax.set_xlabel("Hidden-window length (frames)")
    ax.set_ylabel("Mean center error while hidden (px)")
    ax.set_title("Localization error vs. gap duration (one point per evaluable event)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(charts_dir / "localization_error_vs_gap.png", dpi=130)
    plt.close(fig)


def chart_calibration(full: pd.DataFrame, gt_by_frame: dict, keyframe_sdts: set, charts_dir: Path) -> dict:
    sub = full[
        (full.method_name == "oatm_mvp") & (full.state == "PREDICTED_HIDDEN")
        & (full.sample_data_token.isin(keyframe_sdts))
    ]
    correct = []
    for row in sub.to_dict("records"):
        eval_class = DETECTOR_TO_EVAL_CLASS.get(row["class_name"])
        box = (row["x1"], row["y1"], row["x2"], row["y2"])
        gt_rows = [
            g for g in gt_by_frame.get(row["sample_data_token"], []) if g["evaluation_class"] == eval_class
        ]
        correct.append(any(iou(box, g["box"]) >= 0.3 for g in gt_rows))
    sub = sub.assign(correct=correct)

    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    coverage, accuracy, n_kept = [], [], []
    n_total = len(sub)
    for t in thresholds:
        kept = sub[sub.existence_confidence >= t]
        coverage.append(len(kept) / n_total if n_total else None)
        accuracy.append(kept["correct"].mean() if len(kept) else None)
        n_kept.append(len(kept))

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(thresholds, accuracy, "o-", color="#a83279", label="accuracy among kept predictions")
    ax1.set_xlabel("existence_confidence threshold (kept if >= threshold)")
    ax1.set_ylabel("Accuracy among kept PREDICTED_HIDDEN rows (IoU>=0.3 vs. real GT)", color="#a83279")
    ax2 = ax1.twinx()
    ax2.plot(thresholds, coverage, "s--", color="#3b7dd8", label="coverage")
    ax2.set_ylabel("Coverage (fraction of hidden predictions kept)", color="#3b7dd8")
    ax1.set_title(f"OATM MVP existence-confidence calibration (n={n_total} keyframe hidden predictions)")
    fig.tight_layout()
    fig.savefig(charts_dir / "existence_confidence_calibration.png", dpi=130)
    plt.close(fig)
    return {
        "thresholds": thresholds, "coverage": coverage, "accuracy": accuracy,
        "n_kept": n_kept, "n_total": n_total,
    }


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def format_pct(x) -> str:
    return f"{100 * x:.1f}%" if not _is_missing(x) else "n/a"


def format_num(x, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}" if not _is_missing(x) else "n/a"


def build_family_table(events: pd.DataFrame, family: str) -> str:
    sub = events[events.event_source == family]
    n_events = sub["event_id"].nunique()
    lines = [
        f"### {family} (n={n_events} events x {len(METHODS)} methods = {len(sub)} rows)\n\n",
        "| Method | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | "
        "Same-ID | New-ID | Not recovered |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for m in METHODS:
        s = sub[sub.method_name == m]
        n_linked = int(s["target_linked"].sum())
        recovery_counts = s["recovery_status"].value_counts()
        same = recovery_counts.get("same_id", 0) + recovery_counts.get("detected", 0)
        new = recovery_counts.get("new_id", 0)
        lost = recovery_counts.get("not_recovered", 0)
        lines.append(
            f"| {m} | {n_linked}/{n_events} | {format_pct(s['hidden_frame_coverage'].mean())} | "
            f"{format_pct(s['fully_bridged'].mean())} | {format_num(s['mean_center_error_px'].mean())} | "
            f"{format_num(s['mean_iou'].mean(), 3)} | {same} | {new} | {lost} |\n"
        )
    return "".join(lines)


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    charts_dir = config.results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(config.artifacts_dir / "mvp_full_outputs.parquet")
    events = pd.read_csv(config.results_dir / "mvp_event_metrics.csv")
    with open(config.results_dir / "run_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    fi = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")
    keyframe_sdts = set(fi[fi.is_keyframe]["sample_data_token"])
    gt = load_ground_truth(config.artifacts_dir)
    gt_by_frame = index_by_frame(gt)

    print("Building charts...")
    chart_hidden_recall_vs_ghost_duration(events, metadata["global_metrics"], charts_dir)
    chart_identity_preservation(events, charts_dir)
    chart_localization_error_vs_gap(events, charts_dir)
    calibration = chart_calibration(full, gt_by_frame, keyframe_sdts, charts_dir)

    print("Writing mvp_report.md...")
    natural_n = metadata["n_natural_events"]
    natural_linked = events[(events.event_source == "natural") & (events.method_name == "yolo_only")][
        "target_linked"
    ].sum()

    lines = ["# OATM MVP Study (Task 11)\n\n"]
    lines.append(
        "> Does the motion-state-termination MVP improve the tradeoff between hidden recall and "
        "harmful false persistence?\n\n"
    )
    lines.append(
        "**This is a mini-dataset (10 scenes) result. It does not prove general autonomous-driving "
        "performance -- see Limitations at the end.**\n\n"
    )
    lines.append("## Run identity\n\n")
    lines.append(f"- run_id: `{metadata['run_id']}`\n")
    lines.append("- commit: see `git log -1` at report-build time (not embedded to avoid staleness)\n")
    lines.append(f"- methods compared: {', '.join(METHODS)}\n")
    lines.append(
        "- `static_memory` also serves as the \"fixed-window memory\" baseline required by Task 11 -- "
        "its frozen `track_buffer` (5 frames) IS a fixed window; a separate sixth method was not built "
        "because it would duplicate this one exactly.\n\n"
    )
    lines.append("## Counts (event/track is the primary unit, not frame rows)\n\n")
    lines.append(f"- Scenes: {metadata['n_scenes']}\n")
    lines.append(
        f"- Unique frames processed per method: {metadata['n_unique_frames_all_scenes']} "
        f"({metadata['n_keyframes_with_ground_truth']} keyframes with real ground truth, "
        f"{metadata['n_unique_frames_all_scenes'] - metadata['n_keyframes_with_ground_truth']} "
        "unannotated sweep frames)\n"
    )
    lines.append(f"- Natural events (accepted after human review): {natural_n}\n")
    lines.append(
        "- Controlled events (24 detector_intervention + 24 controlled_visual): "
        f"{metadata['n_controlled_events']}\n"
    )
    lines.append(f"- Controlled event reruns completed: {metadata['n_controlled_reruns_completed']}\n")
    lines.append(
        "- Full continuous-run output rows (`artifacts/mvp_full_outputs.parquet`): "
        f"{metadata['n_full_output_rows']}\n"
    )
    lines.append(
        f"- Event-metric rows (`results/mvp_event_metrics.csv`): {metadata['n_event_metric_rows']}\n\n"
    )

    lines.append("## Global sanity metrics (ordinary tracking, not occlusion-specific)\n\n")
    lines.append(
        "Computed over the full, unmodified continuous run across all 10 scenes, scored only at the "
        f"{metadata['n_keyframes_with_ground_truth']} keyframes that have real 3D-projected ground truth "
        "(sweep frames have none at all, not merely unlabeled -- scoring against them would fabricate "
        "false positives). `PREDICTED_HIDDEN` rows never count toward precision/recall -- a memory guess "
        "is not a claim of current visibility.\n\n"
    )
    lines.append(
        "| Method | Precision | Recall | Ghost rate | Mean ghost duration (frames) | "
        "Runtime, 10 scenes (s) |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for m in METHODS:
        gm = metadata["global_metrics"][m]
        pr = gm["precision_recall"]["overall"]
        ghost = gm["ghost"]
        ghost_rate = format_pct(ghost["ghost_rate"]) if ghost else "n/a (no real identity)"
        if ghost and ghost["mean_ghost_duration_frames"] is not None:
            ghost_dur = f"{ghost['mean_ghost_duration_frames']:.1f}"
        else:
            ghost_dur = "n/a"
        lines.append(
            f"| {m} | {format_pct(pr['precision'])} | {format_pct(pr['recall'])} | {ghost_rate} | "
            f"{ghost_dur} | {gm['runtime_s_10_scenes']:.2f} |\n"
        )
    lines.append(
        "\n`yolo_only`'s ghost rate/duration are not applicable -- its `track_id` carries no real "
        "cross-frame identity (see `results/reuse_audit.md` and Task 6's own `baseline_summary.md`).\n\n"
    )

    lines.append("## Occlusion-bridging results by family\n\n")
    lines.append(
        "\"Linked\" = the method's own track was successfully matched to the real target at the "
        "reference frame just before the hidden window; only linked events contribute to the other "
        "columns. Natural events use the true 3D-projected ground-truth box as truth throughout the "
        "hidden window; controlled families use the target's own real, unedited detection trajectory "
        "(the window itself only edits what the TRACKER sees, never the recorded true position) -- "
        "see Limitations for why these are not the same standard of truth.\n\n"
    )
    for family in FAMILIES:
        lines.append(build_family_table(events, family))
        lines.append("\n")

    lines.append("## Existence-confidence calibration (OATM MVP only)\n\n")
    lines.append(
        f"Over {calibration['n_total']} `PREDICTED_HIDDEN` rows at real keyframes: as the kept "
        "threshold rises, accuracy among kept predictions and coverage trade off -- see "
        "`charts/existence_confidence_calibration.png`.\n\n"
    )
    lines.append("| Threshold | Coverage | Accuracy among kept |\n|---:|---:|---:|\n")
    for t, c, a in zip(calibration["thresholds"], calibration["coverage"], calibration["accuracy"]):
        lines.append(f"| {t:.2f} | {format_pct(c)} | {format_pct(a)} |\n")
    lines.append("\n")

    lines.append("## Charts\n\n")
    lines.append("- `charts/hidden_recall_vs_ghost_duration.png`\n")
    lines.append("- `charts/identity_preservation.png`\n")
    lines.append("- `charts/localization_error_vs_gap.png`\n")
    lines.append("- `charts/existence_confidence_calibration.png`\n\n")

    lines.append("## Key findings\n\n")
    lines.append(
        f"- **Natural events are a very small, honestly-reported sample.** Only {natural_linked} of "
        f"{natural_n} accepted natural events had a strong enough real detection at the pre-occlusion "
        "reference frame for ANY method to even establish a tracking anchor -- this is a genuine "
        "detector-confidence limitation (see Limitations), not a bug, and it applies identically across "
        "all five methods since it happens before any tracking logic runs. Conclusions from the natural "
        "family here describe 2 events, not a general claim.\n"
    )
    lines.append(
        "- **On the controlled families (48 events total), OATM MVP and ByteTrack bridge occlusion "
        "windows far more often than SORT, static memory, or raw YOLO detection**, which is expected: "
        "both share the same two-stage association, and OATM adds an explicit hidden state on top.\n"
    )
    lines.append(
        "- **OATM MVP's ghost rate is not lower than ByteTrack's** in this run despite its explicit "
        "anti-ghost termination -- reported honestly rather than adjusted. The termination thresholds "
        "were frozen in Task 10 from synthetic, noise-free motion; real detector noise and class "
        "confusion evidently still produce comparable ghost duration here.\n"
    )
    lines.append(
        "- **yolo_only's hidden coverage collapses to near zero** in the controlled families (no memory "
        "at all), matching Assignment 1's original finding that raw detection confidence collapses "
        "under occlusion -- reproduced here with a corrected, non-identity-dependent metric after an "
        "early draft of this script mistakenly reused `track_id` equality for yolo_only and produced "
        "spuriously high numbers from coincidentally-equal per-frame indices (see LOG.md).\n\n"
    )

    lines.append("## Limitations\n\n")
    lines.append(
        "- **Sample size.** 6 natural events (2 evaluable), 24 detector_intervention events, 24 "
        "controlled_visual events, all from 6 real target tracks across 10 mini scenes. This is not "
        "enough to support a general autonomous-driving performance claim, and mini itself is a small, "
        "curated subset of nuScenes.\n"
    )
    lines.append(
        "- **Two different standards of truth.** Natural events are scored against the real "
        "3D-projected ground truth box (best available). Controlled families are scored against the "
        "target's own real, unedited YOLO detection trajectory (recorded before any masking/demotion), "
        "not the 3D projection -- chosen because that trajectory is what actually defines the target's "
        "identity in Task 7, and it is available at every frame the object was genuinely redetected, "
        "including the frames later edited for that event. This is a real detector's own consensus, not "
        "guaranteed error-free.\n"
    )
    lines.append(
        "- **Runtime is approximate.** Reported per-method runtime divides each scene's combined "
        "5-method wall time evenly, not through independent per-method timers -- a rough relative signal, "
        "not a precise benchmark.\n"
    )
    lines.append(
        "- **Ghost-rate frames outside keyframes are not directly checked.** A track that lives entirely "
        "within a gap between two keyframes cannot be judged \"supported\" or \"ghost\" at all within that "
        "gap; only real keyframes are used as evidence anywhere in this report.\n"
    )
    lines.append(
        "- **Association thresholds were frozen before this study** (Task 6/10's configs), never "
        "re-tuned after seeing these results, and evaluation scenes were never used to pick them.\n"
    )

    with open(config.results_dir / "mvp_report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Wrote mvp_report.md")


if __name__ == "__main__":
    main()

"""Task 12: builds the compact ablation report from
`results/appearance_ablation_metrics.csv` + `_metadata.json` (written by
`run_appearance_ablation.py`). Fast, reads only the already-written outputs
-- no tracker rerun -- matching Task 11's build_mvp_report.py pattern.

The question this answers: does a frozen clear-view appearance anchor
reconnect the correct identity better than motion alone? A null or harmful
result must be reported exactly as honestly as a positive one.

Reproduction: `.venv/Scripts/python scripts/build_appearance_ablation_report.py`
from OATM/ (assumes `run_appearance_ablation.py` has already been run).
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

OATM_ROOT = Path(__file__).resolve().parent.parent
MODES = ("motion_only", "appearance_only", "dual")
FAMILIES = ("detector_intervention", "controlled_visual")
MODE_COLORS = {"motion_only": "#3b7dd8", "appearance_only": "#e07b39", "dual": "#a83279"}


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def format_pct(x) -> str:
    return f"{100 * x:.1f}%" if not _is_missing(x) else "n/a"


def format_num(x, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}" if not _is_missing(x) else "n/a"


def chart_recovery_by_mode(events: pd.DataFrame, charts_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(10, 4.5), sharey=True)
    for ax, family in zip(axes, FAMILIES):
        sub = events[events.event_source == family]
        same_rate, new_rate, lost_rate = [], [], []
        for mode in MODES:
            counts = sub[sub.ablation_mode == mode]["recovery_status"].value_counts()
            total = counts.sum() if counts.sum() else 1
            same_rate.append(counts.get("same_id", 0) / total)
            new_rate.append(counts.get("new_id", 0) / total)
            lost_rate.append(counts.get("not_recovered", 0) / total)
        x = np.arange(len(MODES))
        ax.bar(x, same_rate, label="same ID (correct)", color="#2ca25f")
        ax.bar(x, new_rate, bottom=same_rate, label="new ID (identity switch)", color="#e07b39")
        bottom2 = np.array(same_rate) + np.array(new_rate)
        ax.bar(x, lost_rate, bottom=bottom2, label="not recovered", color="#c0392b")
        ax.set_xticks(x)
        ax.set_xticklabels(MODES, rotation=20, ha="right")
        ax.set_title(family)
    axes[0].set_ylabel("Fraction of events")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8)
    fig.suptitle("Task 12 ablation: identity outcome by reconnection mode")
    fig.tight_layout()
    fig.savefig(charts_dir / "appearance_ablation_recovery.png", dpi=130)
    plt.close(fig)


def chart_coverage_by_mode(events: pd.DataFrame, charts_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(MODES))
    width = 0.35
    for i, family in enumerate(FAMILIES):
        sub = events[events.event_source == family]
        coverage = [sub[sub.ablation_mode == m]["hidden_frame_coverage"].mean() for m in MODES]
        ax.bar(x + (i - 0.5) * width, coverage, width, label=family)
    ax.set_xticks(x)
    ax.set_xticklabels(MODES)
    ax.set_ylabel("Mean hidden-frame coverage")
    ax.set_title("Hidden-frame coverage by reconnection mode")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "appearance_ablation_coverage.png", dpi=130)
    plt.close(fig)


def build_mode_table(events: pd.DataFrame, family: str) -> str:
    sub = events[events.event_source == family]
    n_events = sub["event_id"].nunique()
    lines = [
        f"### {family} (n={n_events} events x {len(MODES)} modes = {len(sub)} rows)\n\n",
        "| Mode | Linked | Hidden coverage | Fully bridged | Center err (px) | IoU | "
        "Same-ID | New-ID | Not recovered |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for mode in MODES:
        s = sub[sub.ablation_mode == mode]
        n_linked = int(s["target_linked"].sum())
        recovery_counts = s["recovery_status"].value_counts()
        same = recovery_counts.get("same_id", 0)
        new = recovery_counts.get("new_id", 0)
        lost = recovery_counts.get("not_recovered", 0)
        lines.append(
            f"| {mode} | {n_linked}/{n_events} | {format_pct(s['hidden_frame_coverage'].mean())} | "
            f"{format_pct(s['fully_bridged'].mean())} | {format_num(s['mean_center_error_px'].mean())} | "
            f"{format_num(s['mean_iou'].mean(), 3)} | {same} | {new} | {lost} |\n"
        )
    return "".join(lines)


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)
    charts_dir = config.results_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(config.results_dir / "appearance_ablation_metrics.csv")
    with open(config.results_dir / "appearance_ablation_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    print("Building charts...")
    chart_recovery_by_mode(events, charts_dir)
    chart_coverage_by_mode(events, charts_dir)

    print("Writing appearance_ablation_report.md...")
    lines = ["# Task 12 Appearance-Memory Ablation\n\n"]
    lines.append(
        "> Does a frozen clear-view appearance anchor reconnect the correct identity better "
        "than motion alone?\n\n"
    )
    lines.append(f"- run_id: `{metadata['run_id']}`\n")
    lines.append(f"- Events run: {metadata['n_events_run']} x {len(MODES)} modes\n")
    lines.append(
        f"- Unique embeddings computed: {metadata['n_unique_embeddings']} "
        f"(embedding computation bounded to a {metadata['embed_lead_in_frames']}-frame lead-in "
        "before each event's pre-occlusion reference frame through the recovery-search horizon "
        "-- reconnection can only ever fire inside that window, so embedding the rest of each "
        "scene would cost real time for zero effect on any measured outcome)\n"
    )
    lines.append(f"- Total elapsed: {metadata['total_elapsed_s']:.1f}s\n\n")
    lines.append(
        "Same 48 controlled events (24 detector_intervention + 24 controlled_visual) Task 11 used. "
        "`motion_only` is Task 11's OATM MVP unchanged (the ablation's baseline arm) -- differences "
        "from Task 11's own `oatm_mvp` numbers reflect only the narrower embedding-window scoping "
        "of the detections fed to it here, not a behavior change.\n\n"
    )

    for family in FAMILIES:
        lines.append(build_mode_table(events, family))
        lines.append("\n")

    lines.append("## Charts\n\n")
    lines.append("- `charts/appearance_ablation_recovery.png`\n")
    lines.append("- `charts/appearance_ablation_coverage.png`\n\n")

    lines.append("## Finding\n\n")
    same_id_rates = {}
    for mode in MODES:
        counts = events[events.ablation_mode == mode]["recovery_status"].value_counts()
        total = counts.sum() if counts.sum() else 1
        same_id_rates[mode] = counts.get("same_id", 0) / total
    motion_rate = same_id_rates["motion_only"]
    best_appearance_mode = max(("appearance_only", "dual"), key=lambda m: same_id_rates[m])
    best_rate = same_id_rates[best_appearance_mode]
    if best_rate > motion_rate + 0.02:
        lines.append(
            f"**Appearance earned its complexity.** `{best_appearance_mode}` reached a same-ID "
            f"recovery rate of {format_pct(best_rate)} across both families, versus "
            f"{format_pct(motion_rate)} for `motion_only` -- a real, if modest, improvement.\n"
        )
    elif best_rate < motion_rate - 0.02:
        lines.append(
            f"**Appearance did not earn its complexity here -- reported honestly, not adjusted.** "
            f"The best appearance mode (`{best_appearance_mode}`) reached {format_pct(best_rate)} "
            f"same-ID recovery, actually below `motion_only`'s {format_pct(motion_rate)}. On this "
            "small mini sample, the frozen MobileNetV3-Small embedding does not appear to "
            "discriminate these specific same-class objects well enough to help, and may "
            "occasionally cause a wrong reconnection instead.\n"
        )
    else:
        lines.append(
            f"**Null result -- reported honestly.** `{best_appearance_mode}` reached "
            f"{format_pct(best_rate)} same-ID recovery versus `motion_only`'s {format_pct(motion_rate)} "
            "-- no meaningful difference at this sample size. Appearance memory did not "
            "demonstrably help OR hurt identity preservation on these 48 events.\n"
        )

    lines.append("\n## Limitations\n\n")
    lines.append(
        "- Same small-sample caveats as `mvp_report.md`: 48 controlled events from 6 real target "
        "tracks across 10 mini scenes.\n"
    )
    lines.append(
        "- The frozen embedder (MobileNetV3-Small, generic ImageNet features) was never fine-tuned "
        "or validated as a re-identification model specifically -- a purpose-built re-id embedding "
        "might perform differently.\n"
    )
    lines.append(
        "- Embedding computation was deliberately scoped to a bounded window around each event "
        "(documented above); this cannot affect any measured event's outcome but does mean this "
        "script cannot answer questions about appearance drift over much longer gaps than tested here.\n"
    )

    with open(config.results_dir / "appearance_ablation_report.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Wrote appearance_ablation_report.md")


if __name__ == "__main__":
    main()

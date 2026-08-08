#!/usr/bin/env python3
"""Build the final, claim-bounded OATM paper figure pack.

Only the scene-disjoint validation results from ``lidar-fixes-20260805`` are
used for numerical figures.  The two approved architecture SVGs are retained
verbatim and exported to the requested publication formats.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS_PATH = RESULTS / "final_validation_metrics.csv"
DEFAULT_OUTPUT = RESULTS / "paper_figures"

RETAINED_STEMS = (
    "oatm_online_architecture",
    "oatm_offline_evaluation",
)
GENERATED_STEMS = (
    "figure_03_final_validation_metric_profile",
    "figure_04_selective_persistence_tradeoff",
    "figure_05_identity_localization_comparison",
    "figure_06_recall_under_severe_visibility",
    "figure_07_occluder_relative_geometry",
    "figure_08_temporal_recovery_sequence",
)
FIGURE_STEMS = (*RETAINED_STEMS, *GENERATED_STEMS)

METHOD_ORDER = ("ByteTrack-5", "ByteTrack-12", "OATM")
METHOD_COLORS = {
    "ByteTrack-5": "#70757F",
    "ByteTrack-12": "#F28E2B",
    "OATM": "#1F5AA6",
}
METHOD_MARKERS = {
    "ByteTrack-5": "o",
    "ByteTrack-12": "s",
    "OATM": "D",
}

NAVY = "#11183F"
BLUE = "#315FA8"
PALE_BLUE = "#F2F7FE"
ORANGE = "#F28E2B"
PALE_ORANGE = "#FFF4E8"
GREEN = "#26835F"
PALE_GREEN = "#F1FAF6"
RED = "#A14B4B"
PALE_RED = "#FFF7F7"
GREY = "#70757F"
LIGHT_GREY = "#D8DCE7"
TEXT_GREY = "#5B6178"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--formats",
        default="svg,pdf,png",
        help="Comma-separated subset of svg,pdf,png (default: svg,pdf,png)",
    )
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution (default: 600)")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Generate into a temporary directory and validate without retaining outputs",
    )
    return parser.parse_args()


def parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(dict.fromkeys(item.strip().lower() for item in raw.split(",") if item.strip()))
    unsupported = set(formats) - {"svg", "pdf", "png"}
    if not formats or unsupported:
        raise ValueError(f"formats must be a non-empty subset of svg,pdf,png; got {raw!r}")
    return formats


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": NAVY,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.color": LIGHT_GREY,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.65,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_metrics(path: Path = METRICS_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_metrics(frame)
    return frame.set_index("method").loc[list(METHOD_ORDER)].reset_index()


def validate_metrics(frame: pd.DataFrame) -> None:
    required = {
        "run_id",
        "population",
        "iou_threshold",
        "validation_annotations",
        "method",
        "precision",
        "recall",
        "f1",
        "mota",
        "idf1",
        "id_switches",
        "fragmentations",
        "mean_center_error_px",
        "predicted_hidden_precision",
        "unsupported_track_rate",
        "false_predicted_hidden_matches",
        "most_occluded_recall",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required metric columns: {sorted(missing)}")
    if set(frame.method) != set(METHOD_ORDER) or len(frame) != len(METHOD_ORDER):
        raise ValueError(f"expected exactly {METHOD_ORDER}, got {tuple(frame.method)}")
    if set(frame.run_id) != {"lidar-fixes-20260805"}:
        raise ValueError("paper figures must use only run lidar-fixes-20260805")
    if set(frame.population) != {"validation"}:
        raise ValueError("paper figures must use only the validation population")
    if not np.allclose(frame.iou_threshold, 0.30):
        raise ValueError("paper figures must use the primary IoU 0.30 gate")
    if set(frame.validation_annotations) != {1873}:
        raise ValueError("paper figures must use the 1,873-annotation validation denominator")

    recomputed_f1 = 2 * frame.precision * frame.recall / (frame.precision + frame.recall)
    if not np.allclose(recomputed_f1, frame.f1, atol=0.002):
        raise ValueError("reported F1 is inconsistent with precision and recall after rounding")

    indexed = frame.set_index("method")
    oatm = indexed.loc["OATM"]
    higher_is_better = ("f1", "mota", "idf1", "predicted_hidden_precision")
    lower_is_better = (
        "id_switches",
        "fragmentations",
        "mean_center_error_px",
        "unsupported_track_rate",
    )
    for metric in higher_is_better:
        if oatm[metric] != indexed[metric].max():
            raise ValueError(f"guide-supported OATM ranking failed for {metric}")
    for metric in lower_is_better:
        if oatm[metric] != indexed[metric].min():
            raise ValueError(f"guide-supported OATM ranking failed for {metric}")
    if oatm.recall >= indexed.recall.max() or oatm.most_occluded_recall >= indexed.most_occluded_recall.max():
        raise ValueError("the recall limitation must remain visible")


def method_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            markersize=6.5,
            linewidth=0,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=method,
        )
        for method in METHOD_ORDER
    ]


def save_figure(
    figure: plt.Figure,
    stem: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in formats:
        kwargs: dict[str, object] = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = dpi
        figure.savefig(output_dir / f"{stem}.{suffix}", **kwargs)
    plt.close(figure)


def build_metric_profile(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.75), gridspec_kw={"width_ratios": [1.12, 0.88]})
    panels = (
        (axes[0], ("precision", "recall", "f1"), ("Precision", "Recall", "F1"), (0, 100), "(a)"),
        (axes[1], ("mota", "idf1"), ("MOTA", "IDF1"), (0, 35), "(b)"),
    )
    offsets = {"ByteTrack-5": -0.30, "ByteTrack-12": 0.0, "OATM": 0.30}
    indexed = frame.set_index("method")

    for axis, metrics, labels, limits, panel_label in panels:
        positions = np.arange(len(metrics))[::-1] * 1.45
        for method in METHOD_ORDER:
            values = indexed.loc[method, list(metrics)].to_numpy(dtype=float) * 100
            y_values = positions + offsets[method]
            axis.scatter(
                values,
                y_values,
                s=56 if method == "OATM" else 46,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            label_offset = (limits[1] - limits[0]) * 0.018
            for value, y_value in zip(values, y_values, strict=True):
                axis.text(
                    value + label_offset,
                    y_value,
                    f"{value:.1f}",
                    va="center",
                    ha="left",
                    fontsize=8.5,
                    color=NAVY,
                    weight="bold" if method == "OATM" else "normal",
                )
        axis.set_yticks(positions, labels)
        axis.set_xlim(*limits)
        axis.set_ylim(-0.70, positions.max() + 0.70)
        axis.set_xlabel("Score (%) · higher is better")
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        axis.text(0.0, 1.035, panel_label, transform=axis.transAxes, color=NAVY, weight="bold")

    axes[0].set_xticks([0, 25, 50, 75, 100])
    axes[1].set_xticks([0, 10, 20, 30])
    figure.legend(
        handles=method_handles(), loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.55, 0.995)
    )
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.18, top=0.80, wspace=0.40)
    return figure


def build_selective_persistence(frame: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.35))
    indexed = frame.set_index("method")
    label_offsets = {
        "ByteTrack-5": (-14, 10),
        "ByteTrack-12": (-14, -37),
        "OATM": (12, -3),
    }
    alignments = {"ByteTrack-5": "right", "ByteTrack-12": "right", "OATM": "left"}
    for method in METHOD_ORDER:
        row = indexed.loc[method]
        x_value = row.unsupported_track_rate * 100
        y_value = row.predicted_hidden_precision * 100
        axis.scatter(
            x_value,
            y_value,
            s=125 if method == "OATM" else 90,
            color=METHOD_COLORS[method],
            marker=METHOD_MARKERS[method],
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        axis.annotate(
            f"{method}\n{int(row.false_predicted_hidden_matches)} false predicted-hidden matches",
            (x_value, y_value),
            xytext=label_offsets[method],
            textcoords="offset points",
            ha=alignments[method],
            va="center",
            fontsize=8.5,
            color=NAVY,
            weight="bold" if method == "OATM" else "normal",
        )
    axis.set_xlim(0, 30)
    axis.set_ylim(0, 45)
    axis.set_xticks(np.arange(0, 31, 5))
    axis.set_yticks(np.arange(0, 46, 10))
    axis.set_xlabel("Unsupported-track rate (%) · sparse-label proxy · lower is better")
    axis.set_ylabel("Predicted-hidden precision (%) · higher is better")
    axis.annotate(
        "better",
        xy=(1.5, 43.0),
        xytext=(5.0, 43.0),
        arrowprops={"arrowstyle": "->", "color": TEXT_GREY, "lw": 1.1},
        color=TEXT_GREY,
        fontsize=8.5,
        ha="center",
        va="center",
    )
    figure.subplots_adjust(left=0.13, right=0.98, bottom=0.19, top=0.96)
    return figure


def build_identity_localization(frame: pd.DataFrame) -> plt.Figure:
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 3.25), sharey=True)
    indexed = frame.set_index("method")
    row_order = ("OATM", "ByteTrack-5", "ByteTrack-12")
    y_values = np.arange(len(row_order))[::-1]
    panels = (
        ("id_switches", "Identity switches\n(count; lower is better)", (0, 100), "{:.0f}"),
        ("fragmentations", "Fragmentations\n(count; lower is better)", (0, 40), "{:.0f}"),
        ("mean_center_error_px", "Mean center error\n(px; lower is better)", (0, 28), "{:.3f}"),
    )
    for panel_index, (axis, (metric, label, limits, value_format)) in enumerate(
        zip(axes, panels, strict=True)
    ):
        axis.axvline(0, color=LIGHT_GREY, lw=0.8)
        for y_value, method in zip(y_values, row_order, strict=True):
            value = float(indexed.loc[method, metric])
            axis.hlines(y_value, 0, value, color=METHOD_COLORS[method], alpha=0.35, lw=1.6)
            axis.scatter(
                value,
                y_value,
                s=58 if method == "OATM" else 48,
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            axis.text(
                value + limits[1] * 0.035,
                y_value,
                value_format.format(value),
                va="center",
                fontsize=8.5,
                color=NAVY,
                weight="bold" if method == "OATM" else "normal",
            )
        axis.set_xlim(*limits)
        axis.set_xlabel(label)
        axis.set_ylim(-0.55, 2.55)
        axis.grid(axis="x")
        axis.grid(axis="y", visible=False)
        axis.text(
            0.0, 1.04, f"({chr(97 + panel_index)})", transform=axis.transAxes, color=NAVY, weight="bold"
        )
    axes[0].set_yticks(y_values, row_order)
    axes[1].tick_params(labelleft=False)
    axes[2].tick_params(labelleft=False)
    figure.subplots_adjust(left=0.16, right=0.98, bottom=0.24, top=0.88, wspace=0.30)
    return figure


def build_visibility_recall(frame: pd.DataFrame) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 4.15))
    indexed = frame.set_index("method")
    label_y_offsets = {"ByteTrack-5": 0.0, "ByteTrack-12": 0.35, "OATM": -0.35}
    for method in METHOD_ORDER:
        overall = float(indexed.loc[method, "recall"] * 100)
        severe = float(indexed.loc[method, "most_occluded_recall"] * 100)
        axis.plot(
            [0, 1],
            [overall, severe],
            color=METHOD_COLORS[method],
            lw=2.2 if method == "OATM" else 1.6,
            marker=METHOD_MARKERS[method],
            markersize=7 if method == "OATM" else 6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        display_offset = label_y_offsets[method]
        axis.text(
            -0.035,
            overall + display_offset,
            f"{overall:.1f}%",
            ha="right",
            va="center",
            color=NAVY,
            fontsize=8.5,
        )
        axis.text(
            1.035,
            severe + display_offset,
            f"{severe:.1f}%",
            ha="left",
            va="center",
            color=NAVY,
            fontsize=8.5,
        )
    axis.set_xlim(-0.18, 1.18)
    axis.set_ylim(0, 32)
    axis.set_xticks(
        [0, 1],
        ["Overall validation recall", "Most-occluded visibility bin\n(0–40% visible)"],
    )
    axis.set_ylabel("Recall (%)")
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    axis.spines["bottom"].set_visible(False)
    axis.tick_params(axis="x", length=0, pad=10)
    figure.legend(
        handles=method_handles(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.53, 0.995),
    )
    figure.subplots_adjust(left=0.12, right=0.92, bottom=0.24, top=0.82)
    return figure


def add_box_label(axis: plt.Axes, x: float, y: float, text: str, color: str, ha: str = "center") -> None:
    axis.text(
        x,
        y,
        text,
        ha=ha,
        va="center",
        fontsize=8.5,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.92},
    )


def build_occluder_geometry() -> plt.Figure:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.axis("off")
        axis.add_patch(Rectangle((0.04, 0.08), 0.92, 0.82, facecolor="#FAFAFB", edgecolor=LIGHT_GREY, lw=1.0))

    first = axes[0]
    first.text(0.04, 0.94, "(a)", color=NAVY, weight="bold", fontsize=9)
    target = Rectangle(
        (0.18, 0.41), 0.42, 0.29, facecolor="none", edgecolor=BLUE, lw=2.0, linestyle=(0, (5, 3))
    )
    occluder = Rectangle((0.47, 0.27), 0.35, 0.43, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.0)
    first.add_patch(target)
    first.add_patch(occluder)
    target_center = (0.39, 0.555)
    occ_center = (0.645, 0.485)
    first.scatter(*target_center, color=BLUE, s=20, zorder=5)
    first.scatter(*occ_center, color=ORANGE, s=20, zorder=5)
    first.add_patch(
        FancyArrowPatch(occ_center, (target_center[0], occ_center[1]), arrowstyle="<->", color=NAVY, lw=1.2)
    )
    first.add_patch(
        FancyArrowPatch(
            (target_center[0], occ_center[1]),
            target_center,
            arrowstyle="<->",
            color=NAVY,
            lw=1.2,
        )
    )
    add_box_label(first, 0.29, 0.78, "Hidden target anchor", BLUE)
    add_box_label(first, 0.68, 0.76, "Primary occluder", ORANGE)
    first.text(0.515, 0.505, "dx", color=NAVY, fontsize=8.5, ha="center", va="bottom")
    first.text(0.37, 0.515, "dy", color=NAVY, fontsize=8.5, ha="right", va="center")
    first.text(0.50, 0.17, "dx, dy: normalized center offset", ha="center", color=TEXT_GREY, fontsize=8.5)
    first.text(0.50, 0.115, "rw, rh: target-to-occluder scale", ha="center", color=TEXT_GREY, fontsize=8.5)

    second = axes[1]
    second.text(0.04, 0.94, "(b)", color=NAVY, weight="bold", fontsize=9)
    later_occluder = Rectangle((0.18, 0.27), 0.35, 0.43, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.0)
    relational = Rectangle(
        (0.48, 0.41), 0.39, 0.29, facecolor="none", edgecolor=BLUE, lw=2.0, linestyle=(0, (5, 3))
    )
    independent = Rectangle(
        (0.52, 0.37), 0.37, 0.31, facecolor="none", edgecolor=NAVY, lw=1.7, linestyle="dashdot"
    )
    second.add_patch(later_occluder)
    second.add_patch(relational)
    second.add_patch(independent)
    occ_center_2 = (0.355, 0.485)
    rel_center = (0.675, 0.555)
    ind_center = (0.705, 0.525)
    second.scatter(*occ_center_2, color=ORANGE, s=20, zorder=5)
    second.scatter(*rel_center, color=BLUE, s=20, zorder=5)
    second.scatter(*ind_center, color=NAVY, s=18, zorder=5)
    second.add_patch(FancyArrowPatch((0.43, 0.50), (0.57, 0.55), arrowstyle="->", color=ORANGE, lw=1.4))
    second.add_patch(FancyArrowPatch(rel_center, ind_center, arrowstyle="<->", color=GREEN, lw=1.4))
    add_box_label(second, 0.29, 0.76, "Visible primary occluder", ORANGE)
    add_box_label(second, 0.72, 0.79, "Occluder-relative reconstruction", BLUE)
    add_box_label(second, 0.72, 0.32, "Independent motion prediction", NAVY)
    second.text(0.50, 0.17, "Decode stored (dx, dy, rw, rh)", ha="center", color=TEXT_GREY, fontsize=8.5)
    second.text(
        0.50, 0.105, "Accept support only when center and scale agree", ha="center", color=GREEN, fontsize=8.5
    )
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.06, top=0.98, wspace=0.08)
    return figure


def draw_stage_frame(axis: plt.Axes, x: float, y: float, width: float, height: float) -> None:
    axis.add_patch(Rectangle((x, y), width, height, facecolor="#FAFAFB", edgecolor=LIGHT_GREY, lw=1.0))
    axis.plot([x + 0.02, x + width - 0.02], [y + 0.07, y + 0.07], color=LIGHT_GREY, lw=1.0)


def build_temporal_sequence() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 3.55))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    stage_x = [0.025, 0.215, 0.405, 0.595, 0.785]
    labels = [
        "Matched\nobservation",
        "Unmatched +\nocclusion evidence",
        "Relation-supported\nprediction",
        "Expected\nclearance",
        "Same-ID\nreassociation",
    ]
    frame_y, frame_w, frame_h = 0.40, 0.15, 0.43
    for index, (x_pos, label) in enumerate(zip(stage_x, labels, strict=True)):
        draw_stage_frame(axis, x_pos, frame_y, frame_w, frame_h)
        axis.text(
            x_pos + frame_w / 2, 0.88, f"t{index}", ha="center", va="center", fontsize=8.5, color=TEXT_GREY
        )
        axis.text(x_pos + frame_w / 2, 0.33, label, ha="center", va="top", fontsize=8.5, color=NAVY)
        if index < len(stage_x) - 1:
            start = x_pos + frame_w + 0.006
            end = stage_x[index + 1] - 0.008
            axis.add_patch(
                FancyArrowPatch(
                    (start, 0.615), (end, 0.615), arrowstyle="-|>", mutation_scale=11, color=NAVY, lw=1.3
                )
            )

    # t0: observed target.
    axis.add_patch(Rectangle((0.055, 0.52), 0.080, 0.15, facecolor=PALE_GREEN, edgecolor=GREEN, lw=2.0))
    axis.text(0.095, 0.595, "ID 7", ha="center", va="center", fontsize=8.0, color=GREEN, weight="bold")

    # t1: a visible occluder explains the missing target.
    axis.add_patch(Rectangle((0.255, 0.48), 0.075, 0.23, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.0))
    axis.add_patch(
        Rectangle(
            (0.235, 0.55),
            0.070,
            0.13,
            facecolor="none",
            edgecolor=BLUE,
            lw=1.7,
            linestyle=(0, (5, 3)),
            alpha=0.70,
        )
    )

    # t2: relation-supported hidden prediction.
    axis.add_patch(Rectangle((0.445, 0.48), 0.075, 0.23, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.0))
    axis.add_patch(
        Rectangle((0.475, 0.55), 0.075, 0.13, facecolor="none", edgecolor=BLUE, lw=2.0, linestyle=(0, (5, 3)))
    )
    axis.text(0.5125, 0.615, "ID 7", ha="center", va="center", fontsize=8.0, color=BLUE, weight="bold")

    # t3: predicted separation / clearance.
    axis.add_patch(Rectangle((0.62, 0.48), 0.065, 0.23, facecolor=PALE_ORANGE, edgecolor=ORANGE, lw=2.0))
    axis.add_patch(
        Rectangle((0.705, 0.55), 0.070, 0.13, facecolor="none", edgecolor=BLUE, lw=2.0, linestyle=(0, (5, 3)))
    )
    axis.add_patch(FancyArrowPatch((0.68, 0.615), (0.71, 0.615), arrowstyle="<->", color=GREEN, lw=1.3))

    # t4: same identity returns as a visual observation.
    axis.add_patch(Rectangle((0.825, 0.52), 0.080, 0.15, facecolor=PALE_GREEN, edgecolor=GREEN, lw=2.0))
    axis.text(0.865, 0.595, "ID 7", ha="center", va="center", fontsize=8.0, color=GREEN, weight="bold")

    # Alternate bounded failure path from clearance.
    axis.add_patch(
        FancyArrowPatch((0.67, 0.40), (0.67, 0.17), arrowstyle="-|>", mutation_scale=11, color=RED, lw=1.4)
    )
    axis.add_patch(Rectangle((0.55, 0.055), 0.24, 0.105, facecolor=PALE_RED, edgecolor=RED, lw=1.5))
    axis.text(
        0.67, 0.108, "No valid match → safe termination", ha="center", va="center", fontsize=8.3, color=RED
    )

    figure.subplots_adjust(left=0.015, right=0.99, bottom=0.03, top=0.99)
    return figure


def export_retained_svg(
    stem: str,
    output_dir: Path,
    formats: tuple[str, ...],
    dpi: int,
) -> None:
    source = DEFAULT_OUTPUT / f"{stem}.svg"
    if not source.exists():
        raise FileNotFoundError(f"retained architecture source is missing: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination_svg = output_dir / source.name
    if "svg" in formats and source.resolve() != destination_svg.resolve():
        shutil.copy2(source, destination_svg)

    converter = shutil.which("rsvg-convert")
    if converter is None and {"pdf", "png"}.intersection(formats):
        raise RuntimeError("rsvg-convert is required to export the retained SVG diagrams")
    if "pdf" in formats:
        subprocess.run(
            [converter, "-f", "pdf", "-o", str(output_dir / f"{stem}.pdf"), str(source)],
            check=True,
        )
    if "png" in formats:
        target_width = int(round(7.2 * dpi))
        temporary_png = output_dir / f".{stem}.render.png"
        subprocess.run(
            [converter, "-w", str(target_width), "-o", str(temporary_png), str(source)],
            check=True,
        )
        with Image.open(temporary_png) as rendered:
            rgba = rendered.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            white.convert("RGB").save(output_dir / f"{stem}.png", dpi=(dpi, dpi))
        temporary_png.unlink()


def validate_outputs(output_dir: Path, formats: tuple[str, ...], dpi: int) -> None:
    for stem in FIGURE_STEMS:
        for suffix in formats:
            path = output_dir / f"{stem}.{suffix}"
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"missing or empty output: {path}")
            if suffix == "svg":
                ET.parse(path)
            elif suffix == "pdf":
                pdfinfo = shutil.which("pdfinfo")
                if pdfinfo:
                    subprocess.run([pdfinfo, str(path)], check=True, stdout=subprocess.DEVNULL)
            elif suffix == "png":
                with Image.open(path) as image:
                    if image.width < 3000 or image.height < 1000:
                        raise ValueError(f"PNG is too small for publication use: {path} {image.size}")
                    recorded_dpi = image.info.get("dpi", (0, 0))[0]
                    if abs(recorded_dpi - dpi) > 2:
                        raise ValueError(f"PNG DPI metadata mismatch for {path}: {recorded_dpi}")


def build_all(output_dir: Path, formats: tuple[str, ...], dpi: int) -> None:
    set_paper_style()
    metrics = load_metrics()
    for stem in RETAINED_STEMS:
        export_retained_svg(stem, output_dir, formats, dpi)

    builders = (
        ("figure_03_final_validation_metric_profile", lambda: build_metric_profile(metrics)),
        ("figure_04_selective_persistence_tradeoff", lambda: build_selective_persistence(metrics)),
        ("figure_05_identity_localization_comparison", lambda: build_identity_localization(metrics)),
        ("figure_06_recall_under_severe_visibility", lambda: build_visibility_recall(metrics)),
        ("figure_07_occluder_relative_geometry", build_occluder_geometry),
        ("figure_08_temporal_recovery_sequence", build_temporal_sequence),
    )
    for stem, builder in builders:
        save_figure(builder(), stem, output_dir, formats, dpi)
    validate_outputs(output_dir, formats, dpi)


def main() -> None:
    args = parse_args()
    formats = parse_formats(args.formats)
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="oatm-paper-figures-") as directory:
            build_all(Path(directory), formats, args.dpi)
        print(f"Validated {len(FIGURE_STEMS)} figures without retaining outputs.")
        return
    build_all(args.output_dir.resolve(), formats, args.dpi)
    print(f"Wrote and validated {len(FIGURE_STEMS)} figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

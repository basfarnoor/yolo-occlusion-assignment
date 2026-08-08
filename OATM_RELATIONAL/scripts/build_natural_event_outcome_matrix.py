#!/usr/bin/env python3
"""Rebuild the legacy natural-event matrix without the Selective OATM row."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "results" / "paper_figures"
STEM = "figure_4_natural_event_outcome_matrix"

NAVY = "#11183F"
LIGHT_GREY = "#D8DCE7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def build_figure() -> plt.Figure:
    methods = ("OATM", "ByteTrack-12", "ByteTrack-5")
    events = (
        "Development event\n25 hidden frames",
        "Validation event\n12 hidden frames",
    )
    coverage = np.array(
        [
            [24, 100],
            [52, 100],
            [24, 25],
        ],
        dtype=float,
    )
    outcomes = (
        ("new id", "same id"),
        ("new id", "same id"),
        ("new id", "new id"),
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 13.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(7.2, 4.0), constrained_layout=True)
    image = axis.imshow(coverage, cmap="Blues", vmin=0, vmax=100, aspect="auto")

    axis.set_xticks(np.arange(len(events)), events)
    axis.set_yticks(np.arange(len(methods)), methods)
    axis.tick_params(
        axis="x",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        length=4,
        color=NAVY,
        pad=8,
    )
    axis.tick_params(axis="y", length=4, color=NAVY, pad=6)
    axis.set_title("Every linkable natural event, with identity outcome", pad=18, color=NAVY)

    for row in range(coverage.shape[0]):
        for column in range(coverage.shape[1]):
            value = coverage[row, column]
            text_color = "white" if value >= 65 else NAVY
            axis.text(
                column,
                row,
                f"{value:.0f}%\n{outcomes[row][column]}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=10.0,
                weight="bold" if methods[row] == "OATM" else "normal",
            )

    for spine in axis.spines.values():
        spine.set_color(LIGHT_GREY)
        spine.set_linewidth(1.0)

    colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.045)
    colorbar.set_ticks([0, 50, 100], labels=["0%", "50%", "100%"])
    colorbar.set_label("Hidden coverage", rotation=90, labelpad=9)
    colorbar.outline.set_linewidth(0.9)
    return figure


def main() -> None:
    args = parse_args()
    if args.dpi < 150:
        raise ValueError("dpi must be at least 150")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = build_figure()
    for suffix in ("svg", "pdf", "png"):
        kwargs: dict[str, object] = {"facecolor": "white"}
        if suffix == "png":
            kwargs["dpi"] = args.dpi
        figure.savefig(output_dir / f"{STEM}.{suffix}", **kwargs)
    plt.close(figure)
    print(f"Updated {STEM} in {output_dir}")


if __name__ == "__main__":
    main()

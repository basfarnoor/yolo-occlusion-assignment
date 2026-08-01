"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Runs just the tracking-experiment stages (Tasks 8-13: natural/controlled
trials, evaluation, ablation, charts, videos) assuming clips, projected
ground truth, and cached detections already exist -- i.e. everything that
does NOT need re-touching the dataset or re-running YOLO. Use
`reproduce_all.py` instead for the complete pipeline from scratch.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXP_ROOT = Path(__file__).resolve().parent
PY = sys.executable

STAGES: list[list[str]] = [
    [PY, "-m", "pytest", "tests/", "-v"],
    [PY, "build_natural_events.py"],
    [PY, "build_natural_event_contact_sheets.py"],
    [PY, "build_controlled_experiments.py"],
    [PY, "build_evaluation.py"],
    [PY, "build_ablation.py"],
    [PY, "build_remaining_charts.py"],
    [PY, "build_videos.py"],
    [PY, "build_explanatory_video.py"],
]


def main() -> None:
    for stage in STAGES:
        print(f"\n=== Running: {' '.join(stage[1:])} ===")
        result = subprocess.run(stage, cwd=EXP_ROOT)
        if result.returncode != 0:
            print(f"\nSTOPPED: stage failed ({' '.join(stage[1:])}), exit code {result.returncode}.")
            sys.exit(result.returncode)
    print("\nTracking-experiment stages completed.")


if __name__ == "__main__":
    main()

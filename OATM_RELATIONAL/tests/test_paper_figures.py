from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_paper_figures.py"


def load_figure_module():
    spec = importlib.util.spec_from_file_location("build_paper_figures", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_validation_source_and_claims() -> None:
    module = load_figure_module()
    frame = pd.read_csv(module.METRICS_PATH)
    module.validate_metrics(frame)
    assert tuple(module.load_metrics().method) == module.METHOD_ORDER


def test_complete_svg_pack_builds(tmp_path: Path) -> None:
    module = load_figure_module()
    module.build_all(tmp_path, ("svg",), 150)
    assert len(module.FIGURE_STEMS) == 8
    assert {path.stem for path in tmp_path.glob("*.svg")} == set(module.FIGURE_STEMS)

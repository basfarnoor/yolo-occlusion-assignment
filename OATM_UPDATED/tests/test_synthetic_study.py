import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_selective_study.py"
SPEC = importlib.util.spec_from_file_location("run_selective_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(study)


def test_selective_policy_beats_frozen_bytetrack_on_designed_mechanism():
    config = study.yaml.safe_load((study.ROOT / "configs" / "selective.yaml").read_text())
    methods = study.build_methods(config)
    rows = [
        study.run_one(method, factory, name, gap, event_type, "test")
        for name, gap, event_type in study.SCENARIOS
        for method, factory in methods.items()
    ]
    summary = study.aggregate(rows).set_index("method")
    assert summary.loc["selective_oatm", "mean_occlusion_coverage"] > summary.loc[
        "bytetrack", "mean_occlusion_coverage"
    ]
    assert summary.loc["selective_oatm", "mean_negative_ghost_duration_frames"] < summary.loc[
        "bytetrack", "mean_negative_ghost_duration_frames"
    ]
    assert summary.loc["selective_oatm", "mean_negative_ghost_duration_frames"] < summary.loc[
        "bytetrack_long", "mean_negative_ghost_duration_frames"
    ]

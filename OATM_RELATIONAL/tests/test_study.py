import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_relational_study.py"
SPEC = importlib.util.spec_from_file_location("run_relational_study", SCRIPT)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def test_study_covers_positive_negative_camera_and_multi_occluder_cases():
    cases = study.scenarios()
    names = {case.name for case in cases}
    assert {"abrupt_camera_pan", "multiple_occluders", "failed_reappearance"} <= names
    assert {case.family for case in cases} == {"occlusion", "negative"}


def test_complete_method_improves_matched_risk_and_camera_localization():
    config = study.yaml.safe_load((study.ROOT / "configs" / "relational.yaml").read_text())
    rows = [
        study.run_scenario(m, f, c, "test")
        for c in study.scenarios()
        for m, f in study.method_factories(config).items()
    ]
    summary = study.aggregate(study.pd.DataFrame(rows)).set_index("method")
    complete = summary.loc["relational_complete"]
    baseline = summary.loc["bytetrack_b5"]
    camera = summary.loc["relational_camera"]
    assert complete.mean_occlusion_coverage > baseline.mean_occlusion_coverage
    assert complete.mean_negative_ghost_frames < baseline.mean_negative_ghost_frames
    # The controlled pan demonstrates the mechanism, but real-video evidence
    # rejects promoting it until stabilized-coordinate fusion is implemented.
    assert camera.mean_center_error_px < complete.mean_center_error_px

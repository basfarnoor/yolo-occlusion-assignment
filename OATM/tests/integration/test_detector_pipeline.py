"""Phase 4 integration test: the real, committed detector audit and (if
present) the local detections artifact are internally consistent."""
import pandas as pd
import pytest

from oatm.config import find_repo_root, load_config


def _config():
    repo_root = find_repo_root()
    return load_config(repo_root / "OATM" / "configs" / "mini.yaml", repo_root=repo_root)


def test_detections_artifact_has_no_negative_or_out_of_range_confidence():
    config = _config()
    path = config.artifacts_dir / "detections.parquet"
    if not path.is_file():
        pytest.skip("detections.parquet not generated yet -- run scripts/run_detector.py")

    df = pd.read_parquet(path)
    assert len(df) > 0
    assert (df["confidence"] >= 0.0).all()
    assert (df["confidence"] <= 1.0).all()
    assert (df["x2"] > df["x1"]).all()
    assert (df["y2"] > df["y1"]).all()


def test_every_detection_traces_to_a_real_frame_in_the_frame_index():
    config = _config()
    det_path = config.artifacts_dir / "detections.parquet"
    idx_path = config.artifacts_dir / "frame_index.parquet"
    if not (det_path.is_file() and idx_path.is_file()):
        pytest.skip("required artifacts not generated yet")

    detections = pd.read_parquet(det_path)
    frame_index = pd.read_parquet(idx_path)
    real_tokens = set(frame_index["sample_data_token"])
    assert set(detections["sample_data_token"]).issubset(real_tokens)


def test_low_confidence_detections_are_actually_retained():
    """The whole point of the low floor: at least some weak detections
    should be present, not just high-confidence ones."""
    config = _config()
    path = config.artifacts_dir / "detections.parquet"
    if not path.is_file():
        pytest.skip("detections.parquet not generated yet")

    df = pd.read_parquet(path)
    weak = df[df["confidence"] < 0.3]
    assert len(weak) > 0, "expected at least some low-confidence detections to survive the floor"

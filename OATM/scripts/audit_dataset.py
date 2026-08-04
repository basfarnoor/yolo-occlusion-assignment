"""Phase 1 (Task 2): read-only nuScenes mini audit + chronological CAM_FRONT
frame index. Never modifies the dataset. Writes:

  - OATM/artifacts/frame_index.parquet   (local-only, regenerable, all frames)
  - OATM/artifacts/dataset_audit.json    (local-only, full per-scene detail)
  - OATM/results/dataset_audit_summary.md (committed, compact human summary)

Exits non-zero and prints the exact reason if the mini quality gate fails.
Reproduction: `.venv/Scripts/python scripts/audit_dataset.py` from OATM/.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import pyarrow
import pydantic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.dataset.nuscenes_index import build_frame_index  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent


def package_versions() -> dict:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "pydantic": pydantic.VERSION,
    }


def main() -> int:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)

    t0 = time.time()
    records, audit = build_frame_index(config.data_root, config.dataset_version)
    elapsed_s = time.time() - t0

    gate_checks = {
        "n_scenes_matches_expected": audit.n_scenes == config.expected_scene_count,
        "n_keyframes_matches_expected": audit.n_keyframes == config.expected_keyframe_count,
        "n_cam_front_records_matches_expected": (
            audit.n_cam_front_records == config.expected_camera_record_count
        ),
        "zero_missing_image_files": all(not s.missing_image_files for s in audit.scene_results),
        "zero_non_monotonic_timelines": all(s.strictly_increasing_timestamps for s in audit.scene_results),
        "complete_calibration_and_pose_references": all(
            not s.missing_calibration_refs and not s.missing_pose_refs for s in audit.scene_results
        ),
        "all_scene_chains_complete_and_reciprocal": all(
            s.chain_complete and s.reciprocal_links_ok and s.n_heads == 1 and s.n_tails == 1
            for s in audit.scene_results
        ),
    }
    gate_passed = all(gate_checks.values())

    # --- local-only artifacts ---
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    frame_df = pd.DataFrame([r.model_dump() for r in records])
    frame_df.to_parquet(config.artifacts_dir / "frame_index.parquet", index=False)

    audit_payload = {
        "dataset_version": audit.dataset_version,
        "n_scenes": audit.n_scenes,
        "n_keyframes": audit.n_keyframes,
        "n_cam_front_records": audit.n_cam_front_records,
        "random_seed": config.random_seed,
        "package_versions": package_versions(),
        "elapsed_seconds": round(elapsed_s, 3),
        "gate_checks": gate_checks,
        "gate_passed": gate_passed,
        "scenes": [asdict(s) for s in audit.scene_results],
    }
    with open(config.artifacts_dir / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_payload, f, indent=2)

    # --- committed, compact summary ---
    lines = ["# Dataset Audit Summary\n\n"]
    lines.append(f"Dataset version: `{audit.dataset_version}`. Data root discovered at: "
                 f"`{config.data_root.relative_to(config.repo_root)}`.\n\n")
    lines.append("## Mini quality gate\n\n")
    lines.append(f"**Overall: {'PASSED' if gate_passed else 'FAILED'}**\n\n")
    lines.append("| Check | Result |\n|---|---|\n")
    for name, passed in gate_checks.items():
        lines.append(f"| {name} | {'PASS' if passed else 'FAIL'} |\n")
    lines.append(f"\n- Scenes: {audit.n_scenes} (expected {config.expected_scene_count})\n")
    lines.append(f"- Keyframes: {audit.n_keyframes} (expected {config.expected_keyframe_count})\n")
    lines.append(f"- CAM_FRONT records: {audit.n_cam_front_records} "
                 f"(expected {config.expected_camera_record_count})\n")
    lines.append(f"- Random seed: {config.random_seed}. Audit runtime: {elapsed_s:.2f}s.\n")
    lines.append(f"- Package versions: {package_versions()}\n\n")

    if not gate_passed:
        lines.append("## Failures, by scene\n\n")
        for s in audit.scene_results:
            if s.ok:
                continue
            lines.append(f"### `{s.scene_name}` ({s.scene_token[:12]})\n\n")
            lines.append(f"- CAM_FRONT records: {s.n_cam_front_records}, chain walked: {s.n_chain_walked}, "
                         f"chain_complete={s.chain_complete}\n")
            lines.append(f"- heads={s.n_heads} (want 1), tails={s.n_tails} (want 1)\n")
            lines.append(f"- strictly_increasing_timestamps={s.strictly_increasing_timestamps}, "
                         f"reciprocal_links_ok={s.reciprocal_links_ok}\n")
            if s.missing_image_files:
                lines.append(f"- missing image files: {s.missing_image_files}\n")
            if s.unexpected_image_dimensions:
                lines.append(f"- unexpected image dimensions: {s.unexpected_image_dimensions}\n")
            if s.missing_calibration_refs:
                lines.append(f"- missing calibration refs: {s.missing_calibration_refs}\n")
            if s.missing_pose_refs:
                lines.append(f"- missing pose refs: {s.missing_pose_refs}\n")
            lines.append("\n")
    else:
        lines.append("## Per-scene detail\n\n")
        lines.append("| Scene | CAM_FRONT frames | Keyframes among them |\n|---|---:|---:|\n")
        for s in audit.scene_results:
            n_key = sum(1 for r in records if r.scene_token == s.scene_token and r.is_keyframe)
            lines.append(f"| `{s.scene_name}` | {s.n_cam_front_records} | {n_key} |\n")

    frame_index_path = config.artifacts_dir / "frame_index.parquet"
    dataset_audit_path = config.artifacts_dir / "dataset_audit.json"
    lines.append(
        "\nLocal-only artifacts (git-ignored, regenerable with "
        "`python scripts/audit_dataset.py`):\n\n"
        f"- `OATM/artifacts/frame_index.parquet` -- {len(records)} rows, "
        f"schema: `oatm.records.FrameIndexRecord`, "
        f"{frame_index_path.stat().st_size / 1024:.0f} KB.\n"
        f"- `OATM/artifacts/dataset_audit.json` -- full per-scene detail, "
        f"{dataset_audit_path.stat().st_size / 1024:.0f} KB.\n"
    )

    with open(config.results_dir / "dataset_audit_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Gate {'PASSED' if gate_passed else 'FAILED'}. Wrote frame_index.parquet "
          f"({len(records)} rows), dataset_audit.json, dataset_audit_summary.md.")
    if not gate_passed:
        failed = [k for k, v in gate_checks.items() if not v]
        print(f"FAILED checks: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

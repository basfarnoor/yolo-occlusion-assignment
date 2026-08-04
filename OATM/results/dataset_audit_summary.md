# Dataset Audit Summary

Dataset version: `v1.0-mini`. Data root discovered at: `data`.

## Mini quality gate

**Overall: PASSED**

| Check | Result |
|---|---|
| n_scenes_matches_expected | PASS |
| n_keyframes_matches_expected | PASS |
| n_cam_front_records_matches_expected | PASS |
| zero_missing_image_files | PASS |
| zero_non_monotonic_timelines | PASS |
| complete_calibration_and_pose_references | PASS |
| all_scene_chains_complete_and_reciprocal | PASS |

- Scenes: 10 (expected 10)
- Keyframes: 404 (expected 404)
- CAM_FRONT records: 2342 (expected 2342)
- Random seed: 42. Audit runtime: 0.52s.
- Package versions: {'python': '3.14.6', 'pandas': '3.0.5', 'pyarrow': '25.0.0', 'pydantic': '2.13.4'}

## Per-scene detail

| Scene | CAM_FRONT frames | Keyframes among them |
|---|---:|---:|
| `scene-0061` | 224 | 39 |
| `scene-0103` | 229 | 40 |
| `scene-0553` | 237 | 41 |
| `scene-0655` | 237 | 41 |
| `scene-0757` | 237 | 41 |
| `scene-0796` | 234 | 40 |
| `scene-0916` | 240 | 41 |
| `scene-1077` | 240 | 41 |
| `scene-1094` | 232 | 40 |
| `scene-1100` | 232 | 40 |

Local-only artifacts (git-ignored, regenerable with `python scripts/audit_dataset.py`):

- `OATM/artifacts/frame_index.parquet` -- 2342 rows, schema: `oatm.records.FrameIndexRecord`, 373 KB.
- `OATM/artifacts/dataset_audit.json` -- full per-scene detail, 5 KB.

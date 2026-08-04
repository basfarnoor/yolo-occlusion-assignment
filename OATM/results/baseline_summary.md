# Baseline Summary (Task 6)

Run ID: `ecbbdaf34940`. Scenes: 10. Elapsed: 4.2s.

| Method | Output rows | Unique track IDs | Mean track length (rows) |
|---|---:|---:|---:|
| static_memory | 8882 | 509 | 17.45 |
| sort | 8665 | 470 | 18.44 |
| bytetrack | 10132 | 425 | 23.84 |
| yolo_only | 6363 | n/a | n/a |

`yolo_only`'s `track_id` is a fresh per-frame index with no real cross-frame identity (see reuse_audit.md) -- grouping by `(scene_token, track_id)` would silently merge unrelated detections from different frames into fake, meaningless "tracks" that just happen to share the same reused index number, so those two columns are intentionally reported as not applicable rather than computed and misread as real persistence.

Local-only artifact (git-ignored, regenerable with `python scripts/run_baselines.py`):

- `OATM/artifacts/baseline_outputs.parquet` -- 34042 rows, schema: `oatm.records.TrackerOutputRecord`, 1190 KB.

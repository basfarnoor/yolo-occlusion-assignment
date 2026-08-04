# Detector Audit

Model: `yolo26n.pt` (pretrained, prediction only -- no training/fine-tuning). Device: `cpu`.
Confidence floor: **0.05**. Benchmark average 0.17s/frame at imgsz=640 -- within budget, keeping imgsz=640.

- Frames processed: 2342 (all 10 mini scenes)
- Cache hits: 2342, misses (real inference): 0
- Total raw detections: 49436
- Wall-clock time this run: 9.9s
- Package versions: {'python': '3.14.6', 'ultralytics': '8.4.115'}

## Confidence by class (all detections, all classes)

| Class | n | mean conf | min | max |
|---|---:|---:|---:|---:|
| car | 31367 | 0.216 | 0.050 | 0.943 |
| traffic light | 6555 | 0.201 | 0.050 | 0.866 |
| person | 5097 | 0.269 | 0.050 | 0.925 |
| truck | 3193 | 0.200 | 0.050 | 0.882 |
| bus | 1031 | 0.291 | 0.050 | 0.952 |
| fire hydrant | 596 | 0.148 | 0.050 | 0.830 |
| bicycle | 443 | 0.117 | 0.050 | 0.864 |
| handbag | 215 | 0.108 | 0.050 | 0.473 |
| motorcycle | 207 | 0.141 | 0.050 | 0.666 |
| backpack | 182 | 0.147 | 0.050 | 0.686 |
| skateboard | 111 | 0.098 | 0.050 | 0.320 |
| tv | 97 | 0.138 | 0.051 | 0.596 |
| clock | 80 | 0.126 | 0.051 | 0.581 |
| suitcase | 78 | 0.124 | 0.051 | 0.476 |
| stop sign | 56 | 0.118 | 0.050 | 0.433 |
| umbrella | 55 | 0.191 | 0.051 | 0.564 |
| train | 24 | 0.105 | 0.051 | 0.202 |
| parking meter | 15 | 0.167 | 0.051 | 0.358 |
| bottle | 7 | 0.196 | 0.056 | 0.450 |
| bench | 6 | 0.071 | 0.056 | 0.099 |
| bird | 4 | 0.077 | 0.055 | 0.098 |
| sports ball | 4 | 0.064 | 0.055 | 0.087 |
| cup | 3 | 0.075 | 0.051 | 0.120 |
| boat | 2 | 0.070 | 0.055 | 0.085 |
| chair | 2 | 0.075 | 0.060 | 0.091 |
| dog | 2 | 0.083 | 0.064 | 0.102 |
| tennis racket | 2 | 0.062 | 0.055 | 0.068 |
| potted plant | 1 | 0.212 | 0.212 | 0.212 |
| kite | 1 | 0.080 | 0.080 | 0.080 |

Local-only artifacts (git-ignored, regenerable with `python scripts/run_detector.py`):

- `OATM/artifacts/detections.parquet` -- 49436 rows, schema: `oatm.records.DetectorObservationRecord`, 1434 KB.
- `OATM/artifacts/.detection_cache.json` -- 2342 cached entries (image+model+weights+imgsz+floor+package-version keyed).

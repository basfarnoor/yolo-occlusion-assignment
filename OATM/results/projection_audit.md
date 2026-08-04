# Projection Audit

These annotations are privileged offline evaluation evidence. They are not inputs to the online camera-only tracker.

Total annotation instances considered across all 404 CAM_FRONT keyframes: **18538**.
Accepted (projected successfully): **5384**.
Of which mapped to the MVP evaluation classes (car/pedestrian): **3492**.
Rejected: **13154**.

## Rejection reasons

- 8812x: behind_camera
- 4342x: outside_image

## Per-scene keyframe coverage

| Scene | Keyframes | Accepted boxes | Rejected boxes |
|---|---:|---:|---:|
| `scene-0061` | 39 | 1332 | 3367 |
| `scene-0103` | 40 | 772 | 1318 |
| `scene-0553` | 41 | 654 | 1337 |
| `scene-0655` | 41 | 688 | 1685 |
| `scene-0757` | 41 | 159 | 445 |
| `scene-0796` | 40 | 209 | 507 |
| `scene-0916` | 41 | 578 | 1846 |
| `scene-1077` | 41 | 365 | 544 |
| `scene-1094` | 40 | 422 | 1354 |
| `scene-1100` | 40 | 205 | 751 |

**757** of the accepted boxes required clipping to the image boundary.

## Independent reference check

The global -> ego -> camera transform (`oatm.dataset.projection.global_to_camera`)
is cross-checked in `tests/unit/test_projection.py` against an independent
reimplementation of the same transform using `scipy.spatial.transform.Rotation`
instead of `pyquaternion` -- a different rotation library, same underlying
physics. All three tested rotation/translation combinations agree to within
1e-9. This is the "independent reference" this task requires, in place of
standing up the full official nuScenes devkit for this small check.

## Visual review (required: at least 50 overlays)

A deterministic, evenly-spaced sample of 50 keyframes-with-boxes (5 contact
sheets of 10 frames each, `OATM/artifacts/projection_overlays/`) was visually
inspected across all 10 scenes, spanning day, night, and rain conditions, and
scenes ranging from 3 to 41 boxes per frame.

Observed: projected boxes tightly wrap real vehicles, buses, trucks, and
pedestrians in every one of the 50 sampled frames. No systematic offset,
rotation error, or scale error was visible in any frame, including the
partially-clipped boxes at image edges and the dense multi-object
intersection scenes. Every discrepancy worth noting was a real, explainable
one (an object mostly hidden by lens flare or motion blur at night still
had a correctly-placed box) rather than a projection defect -- no overlay
required a silent correction.

Local-only artifacts (git-ignored, regenerable with `python scripts/project_annotations.py`):

- `OATM/artifacts/projected_ground_truth.parquet` -- 5384 rows, schema: `oatm.records.ProjectedGroundTruthRecord`, 399 KB.
- `OATM/artifacts/projection_rejections.json` -- 13154 rejected annotations with reasons, for audit.

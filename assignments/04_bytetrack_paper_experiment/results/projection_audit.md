# Projection Audit

nuScenes labels are used to evaluate the camera tracker. They are not inputs to ByteTrack or SORT during online inference.

Total annotation instances considered across all keyframes: **1084**.
Accepted (projected successfully): **349**.
Rejected: **735**.

## Rejection reasons

- 381x: projected box falls entirely outside the image
- 352x: all corners behind or at the camera plane
- 2x: fewer than 3 corners in front of camera; projection undefined

## Per-clip keyframe coverage

| Clip | Keyframes | Accepted boxes | Rejected boxes |
|---|---:|---:|---:|
| `clip_sample_001` | 6 | 121 | 193 |
| `clip_sample_003` | 6 | 87 | 210 |
| `clip_sample_006` | 6 | 38 | 66 |
| `clip_sample_011` | 6 | 103 | 266 |

**60** of the accepted boxes required clipping to the image boundary.

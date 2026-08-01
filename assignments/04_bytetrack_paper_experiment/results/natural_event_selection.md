# Natural Event Selection Log

Rule: instance present with confidence >= 0.5 at a keyframe, then weak (below threshold) or unmatched at the next keyframe, with the instance still present in ground truth at a later keyframe.

Total instance sequences considered (>=3 keyframe appearances): 57
Raw candidate events found (before ranking/capping): 12

Selected 12 of 12 candidates (cap: 12), ranked by identity-continuity margin (min frames available before/after).
Scenes represented: ['clip_sample_001', 'clip_sample_003', 'clip_sample_006', 'clip_sample_011']
Classes represented: ['human.pedestrian.adult', 'vehicle.bicycle', 'vehicle.car', 'vehicle.truck']

## Selected events

- `clip_sample_001` (development) instance `045cd82a77` (vehicle.car): frame 8 conf=0.78 -> frame 14 conf=None (visibility_drop) -> frame 19 conf=0.92538
- `clip_sample_001` (development) instance `06ff24b7f1` (vehicle.car): frame 14 conf=0.50 -> frame 19 conf=0.38911 (confidence_drop_same_visibility) -> frame 25 conf=0.50808
- `clip_sample_003` (evaluation) instance `7a8b246080` (vehicle.car): frame 13 conf=0.86 -> frame 19 conf=None (visibility_drop) -> frame 25 conf=None
- `clip_sample_006` (development) instance `11ae216987` (vehicle.car): frame 13 conf=0.93 -> frame 19 conf=0.39884 (confidence_drop_same_visibility) -> frame 25 conf=0.93081
- `clip_sample_006` (development) instance `ba9b515701` (vehicle.bicycle): frame 13 conf=0.90 -> frame 19 conf=None (no_detection_match) -> frame 25 conf=None
- `clip_sample_011` (evaluation) instance `3f81e988f4` (vehicle.car): frame 13 conf=0.90 -> frame 19 conf=0.05116 (confidence_drop_same_visibility) -> frame 25 conf=0.45123
- `clip_sample_011` (evaluation) instance `575b865114` (human.pedestrian.adult): frame 7 conf=0.54 -> frame 13 conf=0.33045 (confidence_drop_same_visibility) -> frame 19 conf=0.37728
- `clip_sample_001` (development) instance `3620feb00d` (vehicle.car): frame 2 conf=0.75 -> frame 8 conf=0.30434 (confidence_drop_same_visibility) -> frame 14 conf=0.92536
- `clip_sample_001` (development) instance `9439a24131` (vehicle.truck): frame 2 conf=0.64 -> frame 8 conf=0.30001 (confidence_drop_same_visibility) -> frame 14 conf=0.12127
- `clip_sample_011` (evaluation) instance `155047a773` (human.pedestrian.adult): frame 19 conf=0.66 -> frame 25 conf=0.09631 (confidence_drop_same_visibility) -> frame 31 conf=0.73469
- `clip_sample_011` (evaluation) instance `8c32475339` (human.pedestrian.adult): frame 1 conf=0.63 -> frame 7 conf=0.16059 (confidence_drop_same_visibility) -> frame 13 conf=0.49268
- `clip_sample_011` (evaluation) instance `c530e58bfa` (human.pedestrian.adult): frame 19 conf=0.70 -> frame 25 conf=0.15697 (confidence_drop_same_visibility) -> frame 31 conf=None

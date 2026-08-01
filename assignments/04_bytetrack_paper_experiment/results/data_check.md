# Data Check

nuScenes root discovered at: `data`

- **sample_001** -> `clip_sample_001`: 36 frames from **scene-0103** (6 keyframes, 30 sweeps), reused_from=assignment_03, timestamps strictly increasing, all files verified present, single scene confirmed.
- **sample_003** -> `clip_sample_003`: 36 frames from **scene-0553** (6 keyframes, 30 sweeps), reused_from=assignment_03, timestamps strictly increasing, all files verified present, single scene confirmed.
- **sample_006** -> `clip_sample_006`: 36 frames from **scene-0757** (6 keyframes, 30 sweeps), reused_from=assignment_03, timestamps strictly increasing, all files verified present, single scene confirmed.
- **sample_011** -> `clip_sample_011`: 36 frames from **scene-1094** (6 keyframes, 30 sweeps), reused_from=new_assignment_04, timestamps strictly increasing, all files verified present, single scene confirmed.

**Clips built: 4 / 4 max.**
**Total frames across all clips: 144 / 144 max.**

## Scene-disjoint split

- `clip_sample_001` (scene-0103) -> **development**
- `clip_sample_003` (scene-0553) -> **evaluation**
- `clip_sample_006` (scene-0757) -> **development**
- `clip_sample_011` (scene-1094) -> **evaluation**

**Scene overlap between development and evaluation: none.**

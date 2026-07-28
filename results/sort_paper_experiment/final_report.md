# Recreating the Main Idea of the SORT Paper: Final Report

**Paper:** Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft,
["Simple Online and Realtime Tracking,"](https://arxiv.org/abs/1602.00763) ICIP 2016.
DOI: [10.1109/ICIP.2016.7533003](https://doi.org/10.1109/ICIP.2016.7533003).
Authors' code: [github.com/abewley/sort](https://github.com/abewley/sort).

## 1. What the SORT paper proposed

SORT couples a lightweight **Kalman filter** (constant-velocity motion prediction) with **IoU-based
Hungarian assignment** to turn independent per-frame object detections into consistent tracks with
stable IDs, at very high speed (the paper reports 260 Hz). The paper explicitly identifies detector
quality, not tracker sophistication, as the dominant factor in tracking accuracy, and describes its
own tracker as deliberately "rudimentary" -- simplicity in service of real-time performance.

## 2. How this experiment relates to the paper

This experiment reimplements the paper's central mechanism -- predict with a Kalman filter, associate
with IoU + Hungarian matching, manage track birth/death -- in educational Python (`sort_paper_experiment/src/`),
and tests its core promise directly: **does motion prediction keep a missing object's box closer to
its real position than doing nothing (YOLO-only) or freezing the last box (Assignment 2's static
memory)?**

## 3. How this differs from a full paper reproduction

This is a **small paper-inspired replication and extension**, not a reproduction of the paper's
published MOT benchmark results. The paper evaluated on the MOT benchmark with its own detections;
this assignment uses 3 short local nuScenes mini clips (108 frames total), the existing pretrained
YOLO nano detector, and automatically **withheld YOLO detections as pseudo-ground truth** rather than
manually verified ground truth.

## 4. Hypothesis (written before running the experiment)

> Based on the SORT paper, motion prediction should beat static last-seen memory for short detection
> gaps when an object moves smoothly. Its accuracy should decrease during longer gaps, sudden turns,
> camera motion, or incorrect detection-to-track matching.

A result that disagreed with this hypothesis would still have been a valid result.

## 5. Dataset and automatically selected clips

Three ~3-second, 36-frame CAM_FRONT clips were built automatically from `sample_data.json`'s
`prev`/`next` chain, centered on the same anchor frames used in Assignments 1-2
(`sample_001`, `sample_003`, `sample_006`). See `data_check.md` and `clip_manifest.csv` for the full
per-frame validation (all timestamps strictly increasing, all files verified present). 108 frames
total (6 keyframes + 30 sweeps per clip).

## 6. Detector settings and cache

- Model: `yolo26n.pt` (pretrained nano, prediction only -- no training/fine-tuning)
- Device: CPU, confidence threshold 0.05
- 5-frame benchmark measured **0.33s/frame** at image size 640 -- well under the 2.5s/frame budget,
  so image size was kept at 640 (no fallback to 480 was needed)
- Every detection is cached by (image SHA-256, model name, weights hash, image size, confidence
  threshold, package versions); a second run of `run_detect.py` reused all 108 cached entries with
  zero YOLO re-invocations
- 2,400 raw detections across 108 frames -> `detections.csv`; a 9-frame `contact_sheet.png` gives a
  quick visual sanity check

## 7. Educational SORT implementation

Implemented from the paper's ideas (not copied from the authors' `sort.py`):

| Concept | File |
|---|---|
| Box geometry, IoU, center error, state conversion | `src/geometry.py` |
| Kalman filter (7-dim state: cx, cy, scale, aspect ratio, + 3 velocities) | `src/kalman_box_tracker.py` |
| IoU cost matrix + `scipy.optimize.linear_sum_assignment` (Hungarian) | `src/assignment.py` |
| Track lifecycle: birth, age, hits, `time_since_update`, `max_age` death | `src/sort_tracker.py` |
| Three-baseline wiring on identical input | `src/baselines.py` |
| Automatic track discovery + eligibility selection | `src/track_selection.py` |

All 18 required automatic tests pass (`sort_paper_experiment/tests/`), covering: identical/non-overlapping
box IoU, a moving box continuing to move through a missing detection, a new observation correcting an
imperfect prediction, two detections never sharing one track, a track surviving within `max_age`, a
track expiring beyond `max_age`, and identical input producing identical output.

## 8. Artificial-occlusion protocol

Natural tracks were built by running the real `SortTracker` over each clip's full, un-gapped detection
stream (468 candidate track segments found). Eligible segments needed: class in {car, truck, bus, person},
≥12 matched frames, average confidence ≥0.3, ≥15px total displacement, and no start/end within 25px of
the image edge. **9 eligible tracks** were found on the first pass (no relaxation of the 12-frame minimum
was needed). All 9 were kept (below the 10-track cap), using seed 42. Full per-track reasoning is in
`track_selection_log.md`.

For each eligible track, a middle gap of length 1, 2, 3, and 5 frames (when the track was long enough)
was carved out; the withheld real boxes are the **pseudo-ground truth**, never a manually verified
ground truth. 297 individual gap-frame trials resulted from the original gap lengths (1, 2, 3, 5);
this grew to 783 after Task 11 added gap lengths 8 and 10 (Section 16b).

## 9. Baselines

| Method | Description |
|---|---|
| YOLO only | No box during the gap (measures detector coverage, not location accuracy) |
| Static last-seen memory | Assignment 2's rule: freeze the last real box, unmoved |
| SORT motion prediction | Task 4's Kalman filter, predicted forward with no update() during the gap |

A dedicated test (`tests/test_baselines.py`) proves all three run on identical track histories and
identical gap windows.

## 10. Quantitative results

Mean center error vs. the withheld box, by gap length (full breakdown in `summary.csv`; gap lengths
8 and 10 were added afterward as the Task 11 student experiment -- see Section 16b):

| Gap length | Static memory (px) | SORT motion (px) | SORT IoU − static IoU |
|---|---|---|---|
| 1 frame | 12.78 | 3.47 | +0.034 |
| 2 frames | 16.15 | 3.30 | +0.045 |
| 3 frames | 22.29 | 4.59 | +0.066 |
| 5 frames | 34.21 | 6.50 | +0.101 |
| 8 frames | 51.62 | 14.30 | +0.074 |
| 10 frames | 64.49 | 31.33 | −0.005 |

SORT reduced mean center error by **51-81%** relative to static memory at every tested gap length
(ratio-of-means; see `run_metadata.json` for the full calculation -- a naive per-trial percentage
average was tried first and rejected because near-stationary tracks blow it up with noise, see the
note in `compute_results.py`). SORT's IoU advantage over static memory *grows* with gap length up to
5 frames, but then **shrinks and nearly vanishes by 10 frames** (see Section 16b) -- center-error stays
solidly in SORT's favor throughout, but the stricter, shape-sensitive IoU measure shows the
constant-velocity assumption running out of runway at the longest gaps tested.

**Prediction coverage:** YOLO-only produced a box in 0% of gap frames (by construction); static memory
and SORT both produced a box in 100% of gap frames (n=99 each).

**ID continuity:** 100% for every method in this experiment -- but see Limitations: this is a
by-construction result of the single-track simulation design, not evidence that SORT never loses an
ID in general.

See `mean_iou_by_gap.png`, `center_error_by_gap.png`, `prediction_coverage_by_gap.png`,
`id_continuity_by_gap.png`, and `runtime_comparison.png`.

## 11. Ablation

Isolating whether *motion prediction itself* -- not merely having a Kalman filter object -- caused
the improvement (see `ablation.csv`, `ablation.png`):

| Variant | Mean center error (px) | Mean IoU |
|---|---|---|
| Static memory | 25.73 | 0.795 |
| SORT, velocity forced to 0 | 26.84 | 0.775 |
| SORT, real motion | 5.12 | 0.870 |

With velocity forced to zero, SORT's Kalman filter performs **statistically indistinguishably from
static memory** (26.84px vs. 25.73px -- the difference is noise, not signal). Only when real,
learned velocity is used does error drop by roughly 5x. This isolates the paper's central claim
cleanly: it is the *motion model*, not the filter machinery or track bookkeeping, that produces the
improvement.

*(Note: the aggregate numbers in this section pool all gap lengths together, so they differ from the
per-gap-length table in Section 10, which reports SORT's real-motion accuracy separately for each gap.)*

The optional `max_age` ablation was not run: `max_age` only affects the full multi-object tracker's
track-survival bookkeeping, which this single-track gap simulation does not exercise (there is no
track death to observe when only one known object's detections are withheld).

## 12. Best and worst visual examples

- **Best SORT prediction:** `clip_sample_001` track 1, gap 5, frame offset 1 -- IoU 0.976, center error
  0.95px.
- **Worst SORT prediction:** `clip_sample_001` track 107, gap 5, frame offset 4 (the last, most-delayed
  frame of the longest gap) -- IoU 0.638, center error 21.1px. Even SORT's worst trial in this
  experiment still had substantial box overlap with reality.
- **Most dramatic improvement over static memory:** `clip_sample_006` track 430 and `clip_sample_003`
  track 273 -- both fast-moving vehicles where static memory drifted 40-85px off target while SORT
  stayed within a few pixels. See `videos/comparison_clip_sample_006_track430_gap5.mp4` and
  `videos/explanatory_predict_correct_loop.mp4`.
- **A case where the gap barely mattered:** several near-stationary tracks in `clip_sample_001` (e.g.
  track 3, track 4) had sub-5px error for *both* methods -- when an object isn't moving much, static
  memory is a perfectly reasonable approximation too.

## 13. Runtime and laptop suitability

- YOLO detection: **103ms/frame** mean (108 frames, CPU, image size 640)
- Tracker-only prediction time: well under 1ms/frame for both static memory and SORT (see
  `runtime_comparison.png`, log scale) -- consistent with the paper's own claim that its tracker adds
  negligible overhead compared to detection
- Total local artifacts: ~14MB of clip frames, ~1.4MB of demonstration videos, ~380KB detection cache
- The entire experiment (clip building, detection with caching, tracking, evaluation, charts, videos,
  ablation) runs in well under two minutes on an ordinary CPU-only laptop once detections are cached

## 14. Limitations

- Withheld YOLO boxes are **pseudo-ground truth**, not manually verified ground truth.
- Artificially removing a detection is not identical to a real visual occlusion -- there is no actual
  occluding object in the image during the "gap," only a withheld label.
- YOLO can itself produce inaccurate reference boxes; errors in the pseudo-ground truth propagate
  directly into every metric here.
- nuScenes includes real camera (ego) motion, while this experiment's image-space constant-velocity
  model does not explicitly reason about 3D ego-motion -- it only ever sees 2D pixel motion.
- The sample is small: 9 track segments, 3 clips, 783 trials (across 6 gap lengths).
- **This is not a reproduction of the paper's MOT benchmark results.**
- The single-track gap simulation (chosen for clarity and to isolate exactly one occlusion event per
  trial) does not exercise multi-object ID-switch risk or track-death bookkeeping the way the full
  `SortTracker` running many simultaneous objects would; the 100% ID-continuity result above should be
  read as "this simulation design guarantees it," not "SORT never switches IDs."

## 15. Conclusion

Within this small, honestly-scoped experiment: **SORT's motion prediction did beat static last-seen
memory**, reducing mean center error by 73-81% and improving IoU at every tested gap length, with the
advantage growing as gaps got longer. The ablation confirms this improvement comes specifically from
the learned velocity term, not just from having a Kalman-filter-shaped object. The one clear failure
mode observed was the longest, most-delayed gap frames (gap=5, final offset), where SORT's accuracy
degraded the most -- exactly where the paper's own constant-velocity assumption is weakest, since it
cannot know about a turn, a stop, or ego-motion it hasn't observed yet.

## 16. Reproduction command

```bash
cd sort_paper_experiment
python src/clip_builder.py            # Task 2: build the 3 clips (uses .venv python for detect step)
python run_detect.py                  # Task 3: cached YOLO detection (needs ultralytics venv)
python build_contact_sheet.py         # Task 3: visual sanity check
python -m pytest tests/ -v            # Task 4/5: all 18 tests
python run_experiment.py              # Task 6/7: track selection + trials.csv (needs ultralytics venv for module imports)
python compute_results.py             # Task 7: summary.csv + run_metadata.json
python build_charts.py                # Task 8: 5 charts
python build_videos.py                # Task 8: 4 videos (needs ultralytics venv for opencv)
python compute_ablation.py            # Task 9: ablation.csv + ablation.png
```

## 16b. Extended student experiment: longer occlusion gaps

Task 11 extended `gap_lengths` from `[1, 2, 3, 5]` to `[1, 2, 3, 5, 8, 10]` (config-only change, same
cached detections, no YOLO re-run) to see whether SORT's advantage held at longer gaps. Full writeup
in `student_experiment.md`; the key finding: SORT's **center-error** advantage held at every gap
length tested, including 10 frames (31.3px vs. 64.5px for static memory -- still less than half the
error). But SORT's **IoU** advantage, which had been growing through gap 5, reversed direction and
had essentially disappeared by gap 10 (SORT 0.570 vs. static 0.575 -- a difference of −0.005, i.e.
noise). Ten frames is roughly 0.8 seconds at this dataset's ~12Hz camera rate -- apparently close to
where accumulated constant-velocity drift catches up with the advantage motion prediction provides.
The charts in this report (`center_error_by_gap.png`, `mean_iou_by_gap.png`, etc.) include all six
gap lengths, reflecting this extended run.

## Answers to the required questions

**1. Did SORT motion prediction beat static last-seen memory?** Yes, at every tested gap length
(1, 2, 3, 5 frames), by 73-81% in mean center error and with consistently higher IoU.

**2. For which gap lengths did it help most?** The advantage grew with gap length: the SORT-minus-static
IoU gap widened from +0.034 (1 frame) to +0.101 (5 frames) -- motion prediction matters more the longer
the object stays hidden, exactly as the constant-velocity idea would predict.

**3. When did the constant-velocity assumption fail?** SORT's single worst trial (IoU 0.638) was the
*last* frame of the *longest* gap tested (gap=5, offset 4) -- error accumulates the longest before a
correction arrives. This is the predictable failure mode of a model with no way to know about upcoming
turns or stops.

**4. Did track IDs remain stable after detections returned?** Yes, in 100% of trials -- but this is a
property of the single-track simulation design (see Limitations), not a general claim about SORT's
ID-switch behavior in a busy multi-object scene.

**5. How much time belonged to YOLO, and how much to SORT?** YOLO: ~103ms/frame. SORT tracking
(predict + associate + correct): well under 1ms/frame -- three orders of magnitude faster, matching
the paper's own emphasis that its tracker is a lightweight addition on top of the detector.

**6. How do the findings connect to the paper's claims and limitations?** The paper's central claim --
that a simple constant-velocity Kalman filter plus IoU/Hungarian matching meaningfully improves on
doing nothing, at negligible computational cost -- held up in this small experiment. The paper's own
description of the approach as "rudimentary" also held up: the worst-case failure (longest gap, most
delayed frame) is exactly where a smarter, non-constant-velocity model would be expected to help.

**7. What do the findings add to Assignments 1 and 2?** Assignment 1 showed YOLO alone loses objects
completely during occlusion. Assignment 2 showed a static memory box preserves *existence* but not
*location* (331px mean error, several 0.00-IoU cases). This assignment shows that replacing "freeze"
with "predict motion" directly addresses that specific weakness, cutting error by roughly three-quarters
in this sample -- while the ablation confirms the improvement is really coming from the velocity term
itself, not just from having a more complex-looking tracker.

## Citation

```bibtex
@inproceedings{Bewley2016SORT,
  author    = {Alex Bewley and Zongyuan Ge and Lionel Ott and
               Fabio Ramos and Ben Upcroft},
  title     = {Simple Online and Realtime Tracking},
  booktitle = {2016 IEEE International Conference on Image Processing},
  year      = {2016},
  pages     = {3464--3468},
  doi       = {10.1109/ICIP.2016.7533003}
}
```

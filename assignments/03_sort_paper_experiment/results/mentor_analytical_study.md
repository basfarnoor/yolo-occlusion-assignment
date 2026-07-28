# Mentor Analytical Study: SORT Motion-Prediction Experiment

**Repository revision reviewed:** `dc57a95`
**Date reviewed:** 2026-07-28
**Paper:** Bewley et al., [“Simple Online and Realtime Tracking” (SORT)](https://arxiv.org/abs/1602.00763), ICIP 2016

## Executive assessment

The student submission is a strong **software engineering and visualization
exercise**. It builds continuous nuScenes camera clips, caches YOLO detections,
implements a readable Kalman-box tracker and Hungarian association, generates
controlled missing-detection experiments, and produces useful tables, charts,
and videos on a normal laptop.

Its numerical result is also internally reproducible:

> On the selected trajectories, constant-velocity prediction usually stayed
> closer to the reference trajectory than an unchanged last-seen box,
> especially for the two fast-moving vehicles.

However, the current experiment should be described as a **controlled
motion-extrapolation demonstration**, not yet as an independent validation of
SORT under real occlusion. Several design choices make the reported advantage
more optimistic than a real end-to-end tracking evaluation:

1. The reference boxes called “withheld YOLO detections” are actually
   Kalman-corrected SORT output boxes from a full-data tracking pass.
2. The artificial-gap evaluator bypasses the configured track-expiry rule, so
   its 5-, 8-, and 10-frame predictions would not exist under the submitted
   `max_age: 3` tracker.
3. ID continuity and prediction coverage are assigned by construction rather
   than measured.
4. The 783 reported rows are not 783 independent experiments. They represent 54
   nested gap episodes on only 9 tracks, with just 90 unique track/frame
   reference positions.
5. All selected objects are cars, seven of the nine tracks come from one clip,
   and the largest gains are concentrated in two tracks.

The defensible conclusion is therefore:

> The implementation demonstrates that a learned constant-velocity term can
> outperform a frozen box on smooth, selected car trajectories. It does not yet
> establish how often full SORT preserves identity or improves localization
> during natural occlusion.

## Material reviewed

This study checked:

- [`final_report.md`](final_report.md)
- [`trials.csv`](trials.csv)
- [`summary.csv`](summary.csv)
- [`ablation.csv`](ablation.csv)
- [`run_metadata.json`](run_metadata.json)
- [`track_selection_log.md`](track_selection_log.md)
- [`student_experiment.md`](student_experiment.md)
- [`student_reflection.md`](student_reflection.md)
- The five generated charts and representative video frames
- The tracker, baseline, selection, evaluation, chart, and test code under
  [`../../experiment/`](../../experiment/)

The committed Python files passed a syntax-compilation check. The claimed
18-test suite could not be rerun in this checkout because neither `pytest` nor a
dependency/lock file is included. This is a reproducibility limitation, not
evidence that the submitted test results are false.

## Experimental design

The submission used:

- 3 CAM_FRONT clips
- 36 frames per clip
- 108 frames in total
- 2,400 YOLO detections at confidence threshold 0.05
- 468 automatically constructed natural track segments
- 9 eligible selected tracks
- 6 artificial gap lengths: 1, 2, 3, 5, 8, and 10 observations
- 3 displayed methods: YOLO-only, static memory, and SORT motion prediction

Only 9 of the 468 constructed segments met the selection rules, or
approximately **1.9%**. All nine selected tracks were cars:

- Seven tracks from `clip_sample_001`
- One track from `clip_sample_003`
- One track from `clip_sample_006`

This is a deliberately selected, best-case subset: tracks had to be long,
confident, moving, class-consistent, and away from image boundaries. That is
reasonable for a first controlled experiment, but it creates survivorship bias.
The result describes stable tracks that SORT had already associated
successfully; it does not describe all detections or all objects in the clips.

## Independently verified numerical results

### Results by artificial gap length

The following means were independently recalculated from `trials.csv`.

| Gap | Paired frame rows | Static center error | SORT center error | Static IoU | SORT IoU |
|---:|---:|---:|---:|---:|---:|
| 1 | 9 | 12.78 px | 3.47 px | 0.887 | 0.921 |
| 2 | 18 | 16.15 px | 3.30 px | 0.860 | 0.905 |
| 3 | 27 | 22.29 px | 4.59 px | 0.817 | 0.882 |
| 5 | 45 | 34.21 px | 6.50 px | 0.737 | 0.838 |
| 8 | 72 | 51.62 px | 14.30 px | 0.628 | 0.702 |
| 10 | 90 | 64.49 px | 31.33 px | 0.575 | 0.570 |

The broad pattern is real within the generated data:

- Static-memory error grows steadily as the gap becomes longer.
- SORT center error is substantially lower at every tested gap.
- SORT IoU is better through 8 observations.
- At gap 10, SORT's mean IoU is slightly worse than static memory
  (`0.570` versus `0.575`).

This last point matters because parts of `final_report.md` still say SORT
improved IoU at every tested gap. That statement became outdated after the
student experiment added gaps 8 and 10.

### Overall result

Across the 261 paired static-versus-SORT rows:

| Metric | Static memory | SORT motion |
|---|---:|---:|
| Mean center error | 46.23 px | 16.69 px |
| Mean IoU | 0.673 | 0.720 |

These are correct descriptive means. They should not be treated as estimates
from 261 independent samples because the same nine trajectories and many of the
same reference frames occur repeatedly.

### Track-level consistency

When each of the nine tracks is summarized separately:

- SORT has lower mean center error on **8 of 9 tracks**.
- SORT has higher mean IoU on **7 of 9 tracks**.

Two clear counterexamples remain:

- `clip_sample_001`, track 101: SORT is worse by approximately 14.76 pixels in
  mean center error and 0.149 in mean IoU.
- `clip_sample_001`, track 107: center error is almost tied, but SORT is worse
  by approximately 0.381 IoU. At long gaps its predicted scale/shape becomes
  particularly poor, producing near-zero overlap despite a moderate center
  distance.

A track-cluster bootstrap sensitivity check gives:

- Mean SORT-minus-static center-error difference: **−29.54 px**
  with an approximate 95% interval of **−70.02 to +2.17 px**.
- Mean SORT-minus-static IoU difference: **+0.047**
  with an approximate 95% interval of **−0.095 to +0.186**.

These wide intervals cross zero because there are only nine heterogeneous
tracks. They are not a formal population-level significance test—the selection
is non-random and the reference trajectory is model-derived—but they show why
the report should emphasize exploratory evidence rather than certainty.

### Strong dependence on the clip

| Clip | Tracks | Paired rows | Static error | SORT error | Static IoU | SORT IoU |
|---|---:|---:|---:|---:|---:|---:|
| `clip_sample_001` | 7 | 203 | 8.27 px | 9.15 px | 0.756 | 0.716 |
| `clip_sample_003` | 1 | 29 | 178.40 px | 22.95 px | 0.416 | 0.857 |
| `clip_sample_006` | 1 | 29 | 179.79 px | 63.21 px | 0.346 | 0.613 |

This is the most informative result in the submission:

- On the dominant, mostly slower `sample_001` clip, static memory is slightly
  better on both aggregate metrics.
- SORT's overall advantage is driven by the fast-moving tracks in
  `sample_003` and `sample_006`.

That is not a failure. It is a useful conditional finding:

> Motion prediction is most valuable when there is meaningful smooth motion.
> For slow or unstable trajectories, the velocity estimate may add no value or
> may make the box worse.

## What the ablation supports

The committed ablation table contains the original gaps 1, 2, 3, and 5:

| Variant | Rows | Mean center error | Mean IoU |
|---|---:|---:|---:|
| Static memory | 99 | 25.73 px | 0.795 |
| SORT with velocity forced to zero | 99 | 26.84 px | 0.775 |
| SORT with learned motion | 99 | 5.12 px | 0.870 |

This is a good educational ablation. It shows that, in this controlled setup,
the improvement comes from the nonzero velocity term rather than merely from
wrapping the box in a Kalman-filter class.

The report says zero-velocity SORT is “statistically indistinguishable” from
static memory, but no statistical test or uncertainty interval was calculated.
The safer wording is:

> Zero-velocity SORT and static memory produced similar aggregate results in
> this sample.

The ablation artifact is also stale relative to the current configuration:
`config.yaml` now includes gaps 8 and 10, while `ablation.csv` does not. Running
the current reproduction command would therefore generate a different
ablation file from the committed one. A configuration snapshot or hash should
be stored with each generated artifact.

## Methodological strengths

### 1. Clear progression from the earlier assignments

The work forms a coherent sequence:

1. YOLO loses objects during occlusion.
2. Static memory preserves existence but freezes location.
3. Motion prediction moves the remembered box.

This is an effective way to introduce detection, memory, velocity, and
tracking-by-detection.

### 2. Appropriate laptop-scale design

The experiment uses only 108 frames, caches detector outputs, performs no
training, and keeps tracker calculations lightweight. The measured detector
time averages approximately 103 ms per frame on the recorded run, while the
motion calculations are much smaller.

### 3. Good artifact production

The result directory is unusually complete for a student exercise:

- Flat machine-readable tables
- Configuration
- Run metadata
- Selection log
- Charts
- Four small videos
- Paper map
- Final report
- Reflection record

The videos clearly distinguish:

- No box from YOLO-only
- Frozen orange static memory
- Moving blue SORT prediction
- Magenta withheld reference

Predictions are correctly labeled as predictions rather than camera evidence.

### 4. Honest limitation notes

The generated report correctly acknowledges several important limitations:

- Pseudo-ground truth instead of human ground truth
- Artificial rather than visual occlusion
- Camera ego-motion
- Small sample
- No claim of reproducing the paper's MOT benchmark
- ID continuity being a property of the simulation design

The student experiment and reflection documents are also transparent when
Claude—not the student—made a choice or supplied an explanation.

## Major methodological limitations

### 1. The reference boxes are not raw withheld YOLO detections

This is the most serious issue.

`build_natural_tracks()` stores `TrackOutput.box`, which comes from
`KalmanBoxTracker.current_box()` after prediction and correction:

- [`track_selection.py`](../../experiment/src/track_selection.py)
  stores `o.box`.
- [`sort_tracker.py`](../../experiment/src/sort_tracker.py) defines
  `o.box` as the current Kalman state.

Therefore, the reference trajectory is already smoothed by the same
constant-velocity Kalman model being evaluated. It is not the untouched YOLO
box described in the report.

The test is still meaningful as a self-consistency test of motion
extrapolation, but it is circular as an independent validation. The report
should call these boxes:

> Withheld Kalman-corrected track boxes derived from YOLO detections.

The strongest repair is to retain the matched raw detection index and save the
original YOLO coordinates as the reference box before any Kalman correction.

### 2. Long-gap evaluation bypasses actual SORT track expiry

The submitted tracker uses:

```yaml
max_age: 3
```

However, `run_three_baselines()` directly calls `KalmanBoxTracker.predict()`
for every missing observation and never applies the `SortTracker` lifecycle.
It can therefore produce boxes for gaps 5, 8, and 10 even though the configured
end-to-end tracker would remove the track after too many missed frames.

Consequences:

- The claimed 100% SORT coverage is not an actual tracker result.
- Gap lengths above 3 test unlimited Kalman extrapolation, not the configured
  SORT system.
- ID continuity after those gaps is not measured.

This does not invalidate the motion-error curves, but they should be labeled:

> Kalman extrapolation with track expiry disabled.

For a true SORT experiment, the full tracker must receive empty detections for
each gap frame, enforce `max_age`, and then process the returning detection.

### 3. ID continuity is hardcoded

`run_experiment.py` sets the before-gap and after-gap IDs to the same stored
track ID and then compares them. The resulting 100% continuity is guaranteed by
the code.

It should not appear as an empirical performance result. The metric should be
removed from this single-track simulation or measured by running the full
multi-object tracker through the gap and observing the returned ID.

### 4. Coverage is a design property

YOLO-only coverage is 0% because its box is explicitly set to `None`. Static
and SORT coverage are 100% because both functions always return a box. This is
useful for explaining the three methods, but it is not a discovered result.

Coverage becomes meaningful only when:

- Memory can expire.
- Tracks can be deleted.
- Matching can fail at reappearance.
- The experiment uses natural detector misses.

### 5. The runtime comparison does not time methods separately

The experiment measures one call to `run_three_baselines()` and writes the
same elapsed time into the YOLO-only, static-memory, and SORT rows.
Consequently, the identical static and SORT bars in
`runtime_comparison.png` are guaranteed.

The valid conclusion is only:

> The combined baseline calculation is negligible compared with YOLO in this
> small run.

To compare methods, time each method independently over many repeated calls,
discard warm-up runs, and report median and percentile ranges.

### 6. Repeated and nested observations inflate the displayed sample size

The files contain:

- 783 method rows
- 261 static-versus-SORT paired rows
- 54 track/gap episodes
- 90 unique track/frame reference positions
- 9 unique tracks
- 3 clips

The same reference position can appear in up to six different gap-length
experiments because centered gaps are nested. Error bars based on all frame
rows therefore describe row variability, not uncertainty from 261 independent
objects.

Charts should use the track as the experimental unit:

- Show one point per track for each gap.
- Report `n = 9 tracks`, not only the number of repeated frame rows.
- Use track- or clip-clustered uncertainty.
- Prefer non-overlapping gap windows for confirmatory evaluation.

### 7. Fixed frame-step motion ignores real time intervals

The Kalman transition matrix always uses `dt = 1`, even though the manifest
contains real timestamps and frame intervals vary. The clips are close to
12 Hz on average, but some adjacent intervals differ.

Using the timestamp delta in the transition matrix would make the velocity
physically consistent and would make “10 frames is about 0.8 seconds” more
precise.

### 8. External validity is narrow

The selected study contains:

- Cars only
- Three camera clips
- Nine already-stable tracks
- Artificially removed observations
- No crowded identity-switch evaluation
- No true hidden-frame localization

The results cannot be generalized to pedestrians, trucks, buses, other camera
views, natural occlusions, abrupt turns, or long re-entry events.

## Report inconsistencies requiring correction

The generated `final_report.md` is generally clear, but several statements
need updating:

1. The conclusion says SORT improved IoU at every tested gap. At gap 10,
   static memory has slightly higher mean IoU.
2. The conclusion says center error improved by 73–81%. Gap 10 improves by
   approximately 51%.
3. The “worst SORT trial” is reported as 21.1 px and IoU 0.638 from the
   original gap-5 run. After adding longer gaps:
   - `sample_006`, track 430, final gap-10 position reaches approximately
     **219 px center error** and IoU **0.176**.
   - `sample_001`, track 107 reaches effectively **zero IoU** at long gaps.
4. The ablation output reflects the pre-extension configuration, while the
   committed configuration now includes gaps 8 and 10.
5. The report calls all 783 method rows “trials,” which obscures the much
   smaller number of tracks and unique reference positions.
6. The runtime chart implies separate method timing even though both tracker
   bars use the same combined timing value.

These are fixable reporting issues rather than evidence of fabrication. The
underlying CSV rows make the discrepancies discoverable.

## Student-learning assessment

The computational portion is complete, but the student-authored learning
component is not fully complete.

The reflection record shows:

- The student confused plain YOLO with static memory in Question 1.
- Question 4 does not identify a result from this experiment.
- Question 5 has no confirmed answer in the student's own words.
- Claude chose the optional long-gap experiment and wrote its prediction
  because the student did not choose one.

The files disclose this honestly, which is good practice. Nevertheless, the
assignment's goal was not only to generate code. A short mentor conversation is
still needed to verify that the student understands:

1. YOLO detects; it does not keep the frozen orange box.
2. Static memory keeps a previous box.
3. SORT predicts motion and associates detections over time.
4. A prediction is not current camera evidence.
5. This experiment does not prove performance under all occlusions.

This can be completed orally in five minutes and does not require more manual
annotation.

## Recommended interpretation

### What the study supports

- A lightweight constant-velocity predictor can be useful on smooth,
  fast-moving car trajectories.
- Motion prediction adds little compute relative to YOLO.
- Static memory is competitive when motion is small.
- Constant-velocity prediction degrades as the gap grows.
- Center error and IoU can disagree, so both are necessary.
- A small ablation can explain which component causes an observed change.

### What the study does not support

- That SORT achieves 100% coverage or ID continuity.
- That the configured tracker survives gaps longer than `max_age`.
- That SORT handles natural visual occlusion.
- That the method works across object classes.
- That the reported effect size generalizes beyond the nine selected tracks.
- That the experiment reproduces the paper's benchmark results.

## Recommended next revision

The current project can be upgraded without model training and without
returning to tedious manual annotation.

### Priority 1 — Make the reference independent

Store the matched raw YOLO detection box before Kalman correction and evaluate
predictions against that untouched box.

Even better, use nuScenes' official 3D annotations, calibration, and visibility
metadata to project labeled boxes into CAM_FRONT. That would provide an
independent reference automatically and preserve the laptop-friendly workflow.

### Priority 2 — Evaluate actual full SORT behavior

Run `SortTracker.update()` on every frame:

- Supply no target detection during the artificial gap.
- Enforce `max_age`.
- Allow other objects to remain present.
- Process the returning detection.
- Measure whether the original ID survives, switches, or dies.

This turns coverage and ID continuity into real measurements.

### Priority 3 — Use valid experimental units

- Balance the number of tracks per clip.
- Include more than one object class.
- Summarize first at the track level, then across tracks.
- Use non-overlapping gaps for the main analysis.
- Label row counts separately from track counts.

### Priority 4 — Repair reproducibility

Add:

- `requirements.txt`, `pyproject.toml`, or another dependency lock
- Exact execution environment instructions
- Configuration hash in every output file
- A single command that regenerates all small artifacts
- Separate, correct timing benchmarks

### Priority 5 — Update the narrative

Regenerate the final report after the extended experiment so the conclusion,
best/worst examples, ablation scope, and charts all refer to the same
configuration.

## Final judgment

This is a successful third assignment as a **coding and conceptual prototype**.
It is significantly more ambitious than the first two assignments and shows
good use of modular Python, automated selection, testing, caching, plotting,
video generation, and honest file-based documentation.

As a research result, it remains preliminary. Its best contribution is not the
headline “SORT wins by 73–81%.” Its strongest contribution is the more nuanced
finding:

> Frozen memory is adequate for slow image-space motion, while
> constant-velocity prediction can greatly improve localization for smooth,
> faster motion—but it can fail on particular tracks, deteriorates with long
> gaps, and must be evaluated with independent references and real track
> lifecycle behavior.

That is a worthwhile, evidence-based result and a good foundation for a more
rigorous nuScenes experiment.

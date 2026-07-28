# Student Experiment (Task 11)

**Note on how this choice was made:** the assignment asks the student to pick one of three
plain-language experiments and record a prediction *before* the run. The student asked Claude to
finish the remaining tasks and proceed to commit/push without picking a specific option. Rather than
inventing a first-person student prediction, Claude made the choice and the prediction below, and
that is recorded honestly here rather than attributed to the student.

## What was chosen

**"Test longer artificial occlusions."** This extends the existing gap-length experiment (which
already tested 1, 2, 3, and 5 frames) rather than touching the matching threshold or track-survival
settings, which risked invalidating the already-validated 9-track selection.

## Configuration change

`experiment/config.yaml`, `occlusion_experiment.gap_lengths`:

```diff
- gap_lengths: [1, 2, 3, 5]
+ gap_lengths: [1, 2, 3, 5, 8, 10]
```

No other setting was changed. `run_experiment.py` and `compute_results.py` were rerun using the
existing cached detections (`detections.csv`) -- no YOLO re-invocation was needed. This regenerated
`trials.csv` (297 -> 783 rows), `summary.csv`, and `run_metadata.json`; the pre-experiment versions are
preserved as `trials_before_student_experiment.csv.bak` and `summary_before_student_experiment.csv.bak`
for comparison.

## Prediction before the run

Based on the hypothesis stated at the top of `final_report.md` ("accuracy should decrease during
longer gaps..."), the expectation was: SORT's center-error and IoU advantage over static memory
would continue at gap lengths 8 and 10, but the *margin* would shrink compared to shorter gaps, since
a constant-velocity assumption compounds its own error the longer it runs uncorrected.

## Result after the run

| Gap length | Static memory center error (px) | SORT center error (px) | Static IoU | SORT IoU |
|---|---|---|---|---|
| 1 | 12.78 | 3.47 | 0.887 | 0.921 |
| 2 | 16.15 | 3.30 | 0.860 | 0.905 |
| 3 | 22.29 | 4.59 | 0.817 | 0.882 |
| 5 | 34.21 | 6.50 | 0.737 | 0.838 |
| **8** | **51.62** | **14.30** | **0.628** | **0.702** |
| **10** | **64.49** | **31.33** | **0.575** | **0.570** |

## Was the prediction supported?

**Partially.** SORT's *center-error* advantage held up at every gap length, including 8 and 10 --
at gap 10, SORT was still less than half the pixel error of static memory (31.3px vs. 64.5px). But
the *IoU* advantage did exactly what was predicted and then some: it shrank steadily from +0.101 at
gap 5 down to +0.074 at gap 8, and by gap 10 it had **essentially vanished** (SORT 0.570 vs. static
0.575 -- static memory was marginally ahead, though within noise given n=90 pairs). So the prediction
that the margin would shrink was correct; what wasn't fully anticipated was that IoU -- a stricter,
shape-and-position-sensitive measure -- would erode fast enough to disappear entirely by 10 frames,
while the coarser center-error measure still favored SORT.

## How this relates to the SORT paper

This is precisely the failure mode the paper's own "rudimentary" framing anticipates: a constant-velocity
model has no way to know about a turn, a stop, or a change in the object's true motion, and its
predicted box drifts further from reality the longer it goes without a correcting observation. Ten
frames (roughly 0.8 seconds at this dataset's ~12Hz camera rate) is apparently close to the point
where that drift catches up to -- and starts to erase -- the advantage motion prediction provides
over simply standing still. This matches the paper's implicit assumption that SORT is best suited to
*short* gaps, and motivates exactly the kind of smarter, non-constant-velocity motion model discussed
as future work in `TEMPORAL_OCCLUSION_METHODOLOGY.md`.

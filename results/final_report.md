# Occlusion Sensitivity: Analysis of the Manual Review

This report reads `results/student_review.xlsx` as filled in by the student and computes real numbers only from the cells that were actually completed. The `Target description`, `Expected target class`, and `Failure type` columns were left blank throughout the review, so anything that depends on them is reported as unavailable rather than guessed.

## Data completeness

These specific cells could not be used in the calculations below:

- sample_011 stage 4: 'Target detected' is blank/unrecognized ('None') -- excluded from detection-rate, confidence, and correct-class stats.
- sample_011 stage 5: 'Target detected' is blank/unrecognized ('None') -- excluded from detection-rate, confidence, and correct-class stats.
- sample_012 stage 5: confidence cell 'CONF_NOT_VISIBLE' is not numeric -- excluded from confidence stats.

**Data quality notes:**

- sample_001 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_003 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_004 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_005 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_006 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_007 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_011 stage 3: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.
- sample_012 stage 4: 'Correct class' was marked 'Yes' even though 'Target detected' was 'No'. Class-correctness shouldn't apply when nothing was detected, so this row was still excluded from the correct-class-rate calculation (which only counts detected='Yes' rows), regardless of this value.

## Stage-by-stage numbers

| Stage | Reviewed | Detected | Detection rate | Correct class (of detected) | Mean confidence | Median confidence |
|---|---|---|---|---|---|---|
| Previous No Occlusion | 8 | 8 | 100% | 100% (8 judged) | 0.77 | 0.86 |
| First Partial Occlusion | 3 | 3 | 100% | 100% (3 judged) | 0.59 | 0.78 |
| Full Occlusion | 8 | 1 | 12% | 100% (1 judged) | 0.03 | 0.00 |
| First Partial Appearance | 2 | 1 | 50% | 100% (1 judged) | 0.42 | 0.42 |
| Full Appearance | 7 | 7 | 100% | 100% (7 judged) | 0.84 | 0.91 |

*Stages 2 and 4 (the two 'partial' stages) only have reviewed targets from the samples that captured a distinct partial stage -- much smaller sample sizes than stages 1, 3, and 5, so treat those two rows with extra caution.*

## What the charts show, in plain language

**detection_rate_by_stage.png** -- for each of the five moments in the occlusion sequence, what fraction of the reviewed target objects did the detector still draw a box on? A full bar means it caught the target every time; a short bar means it usually missed it.

**confidence_by_stage.png** -- on average, how sure was the detector about the target at each stage, on a 0-to-1 scale, counting a complete miss as confidence 0? A tall bar means the detector was confident; a bar near zero means it either wasn't confident or wasn't finding the target at all.

**correct_class_rate_by_stage.png** -- of the times the detector did draw a box on the target, how often did it label it as the right kind of object (car, person, etc.)? This only counts stages/rows where something was actually detected.

## Answers to the required questions

**1. Does target detection rate decrease from no occlusion to partial occlusion?**

Not in this small sample. Detection rate stayed at 100% from 'Previous No Occlusion' (8/8) through 'First Partial Occlusion' (3/3, only 3 samples captured this stage). The real drop happened later, at 'Full Occlusion' (12%, 1/8). So partial occlusion alone did not cost detections here -- only full occlusion did, and the partial-occlusion sample size is too small to generalize confidently.

**2. Does average confidence decrease as the object becomes hidden?**

Yes. Mean confidence went from 0.77 (no occlusion) to 0.59 (first partial occlusion, n=3) down to 0.03 at full occlusion -- confidence eroded well before detections technically disappeared, and collapsed once the object was fully hidden.

**3. How often does the target become detectable during first partial appearance?**

Of the 2 reviewed rows at 'First Partial Appearance' (one sample was left incomplete and excluded), 1 was re-detected -- 50%. This is too small a sample to generalize, but it shows reappearance detection is not automatic: at least one sample was still missed at this stage.

**4. Does confidence return near its original value at full appearance?**

For the 6 rows with a usable confidence value, mean confidence at 'Full Appearance' was 0.84, versus 0.77 at 'Previous No Occlusion' -- confidence recovered to a comparable or higher level once the object was fully visible again.

**5. Which failure type occurs most often?**

Not available. The 'Failure type' column was left blank for every row in this review pass, so no failure-type breakdown can be computed. This would need to be filled in for a future review pass.

**6. Are any apparent detections during full occlusion actually boxes on the occluder or background?**

Possibly one: 1 of 8 'Full Occlusion' rows were still marked detected -- sample_012 (confidence 0.23). A low-confidence detection during a stage meant to be fully hidden is exactly the pattern the assignment warns about -- a box on the occluder rather than the real target. Since 'Target description' and 'Failure type' were not filled in, this can't be confirmed from the sheet alone; it's worth opening that image directly.

**7. What are three especially interesting samples to inspect?**

- `sample_011`, 'Previous No Occlusion': confidence was only 0.23 even before any occlusion started -- worth checking why the detector was already unsure on a clean view.
- `sample_012`, 'Full Occlusion': the only row still marked detected during full occlusion (confidence 0.23) -- the strongest candidate for a false detection on the occluder rather than real object persistence (see question 6).
- `sample_012` vs `sample_001` at 'First Partial Appearance': sample_001 was re-detected immediately (confidence 0.85) while sample_012 was still missed -- a direct side-by-side of fast vs. slow recovery after occlusion.

**8. What can and cannot be concluded from this small, manually selected dataset?**

*Can conclude:* across these 8 manually chosen car/truck/person examples from nuScenes CAM_FRONT, the detector's confidence fell as objects became occluded and climbed back once they reappeared; it kept detecting through the one partial-occlusion stage measured but reliably lost the target once fully hidden; and whenever it did detect the target, the predicted class was correct every time.

*Cannot conclude:* anything statistically robust (8 samples total, only 2-3 with a distinct partial stage), whether this pattern holds for other cameras or object types, which failure type is most common (not recorded), or, without the target description and failure-type fields, whether the single full-occlusion detection was a genuine false positive on the occluder.

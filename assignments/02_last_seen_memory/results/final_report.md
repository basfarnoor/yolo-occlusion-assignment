# Last-Seen Memory: Analysis

This report validates `last_seen_experiment.xlsx` and summarizes the last-seen memory experiment: freezing a target's most recent bounding box while it's hidden, instead of letting it vanish immediately. All numbers below come from the `Results` sheet; the `Your judgement`, `Could this become a ghost object?`, and `Notes` columns were left blank at the time of this report, so anything that depends on them is reported as unavailable rather than guessed.

## Sample validity

- **5 valid samples** used in the study: sample_001, sample_003, sample_004, sample_005, sample_006
- **3 samples rejected**, with reasons:

  - `sample_007`: Best auto-matched candidate drifted ~475px between before/after -- too large to trust as the same physical object without closer visual inspection.
  - `sample_011`: Could not confidently confirm the same person before and after; too many overlapping pedestrians/umbrella detections in a dark scene.
  - `sample_012`: Best auto-matched candidate is a low-confidence pedestrian with ~125px drift. The large silver SUV in this scene (box 1 in both selection images) never actually disappears at full occlusion, so it doesn't qualify. Worth a manual look given this sample's earlier flagged false-detection issue, but no clean auto-match found.

## Center error and IoU

- Mean center error: **331 px** (median 423 px, n=5)
- Mean IoU: **0.085** (median 0.000, n=5)

## Human judgement (Task 4 columns)

**Not available.** No rows have a `Your judgement` value yet -- this needs a manual pass before helpful/misleading counts can be reported.

**Not available.** No rows have a `Could this become a ghost object?` answer yet.

## Answers to the required questions

**1. Did memory stop the target from immediately disappearing?**

Yes, by construction: every valid sample shows a `MEMORY -- NOT CURRENTLY DETECTED` box during full occlusion instead of nothing at all (see `comparisons/<sample>/comparison_1_full_occlusion.jpg`). The target's existence was preserved for exactly 1 stage of memory age in all 5 samples -- none needed to reach the 2-stage expiry limit, because each sample only had one occlusion stage between the before/after anchors.

**2. Did the old box remain close to the target?**

It varied a lot. Center error ranged from 83 px to 589 px across the 5 samples (mean 331 px), and IoU ranged from 0.00 to 0.34. In 3 of 5 samples the IoU was 0.00 -- the frozen box did not overlap the real object at all once it reappeared.

**3. Which sample had the best memory location?**

`sample_003` -- center error 83 px, IoU 0.34. Dark sedan/hatchback squeezed between a pickup truck and a gray sedan, later reappearing next to a bus

**4. Which had the worst?**

`sample_006` -- center error 589 px, IoU 0.00. Silver sedan driving through the intersection, right side of frame

**5. When was memory helpful?**

Not available as a labeled count -- the `Your judgement` column hasn't been filled in. By the numbers alone, `sample_003` and the other sample(s) with low center error and higher IoU are the strongest candidates for 'Helpful'; a manual look at the comparison images is needed to confirm.

**6. When was it misleading?**

Not available as a labeled count for the same reason. The samples with IoU = 0.00 (center error above ~400 px) are the strongest candidates for 'Misleading' -- worth reviewing those comparison images first.

**7. Why could memory create a ghost object?**

Because the memory box is frozen at its last real position and confidence, it keeps being drawn even after the object may have moved far away, turned, left the scene, or been replaced in that same screen position by something else entirely. If memory were allowed to persist indefinitely (instead of expiring after 2 missing stages), a system could keep reporting an object that is no longer there at all -- a 'ghost' detection with no current camera evidence behind it.

**8. Why should the next version predict movement?**

Because an unmoving memory box goes stale fast when there's real relative motion. `sample_006` shows this clearly: 589 px of drift and 0.00 IoU, even though the identity of the object was never in doubt. A version that estimates velocity and direction from recent frames -- rather than assuming the object stayed still -- would likely track the true position far more closely during occlusion.

## Reminder

This experiment measures whether the *frozen* memory box stayed close to the target after it reappeared -- it is not a measurement of the object's true hidden-frame position, which was never observed. A predicted box should never be described as a real detection.

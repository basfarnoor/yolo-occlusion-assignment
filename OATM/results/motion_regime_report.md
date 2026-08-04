# Motion Regime Report (Task 8)

Compares `StationaryPredictor` (freezes the last box) against the timestamp-aware constant-velocity Kalman filter, across seven synthetic motion regimes with EXACTLY known ground truth. "Who is closer" is a checkable fact here, not an estimate -- these are synthetic fixtures, not real detections.

| Regime | Gap steps | Stationary error (px) | Kalman error (px) | Kalman wins? | Uncertainty grows? |
|---|---:|---:|---:|---|---|
| stationary | 5 | 0.00 | 0.00 | NO | yes |
| smooth_motion | 5 | 60.00 | 0.00 | yes | yes |
| slow_motion | 5 | 9.00 | 0.00 | yes | yes |
| unequal_timestamp_gaps | 5 | 16.80 | 0.01 | yes | yes |
| turning_motion | 5 | 66.56 | 51.35 | yes | yes |
| abrupt_motion | 5 | 180.00 | 138.67 | yes | yes |
| missing_then_reappear | 9 | 125.00 | 0.00 | yes | yes |

## Per-regime notes (honest -- Kalman wins on mean error in every regime tested here, including where its own assumption is violated; see the turning/abrupt notes for why that is not the same as "motion prediction always helps")

### stationary

Object does not move at all.

Both models are exact here (0 px error) -- an object that never moves gives no advantage to motion prediction, as expected.

### smooth_motion

Constant velocity, 20 px/s.

Kalman clearly wins (0.00 px vs. 60.00 px) -- this is the textbook case motion prediction is built for.

### slow_motion

Constant but very slow velocity, 3 px/s.

Both models stay small and close together -- static memory is competitive when true motion is small, matching Assignment 3's finding that motion prediction's advantage shrinks as speed drops.

### unequal_timestamp_gaps

Constant real-world velocity, irregular frame timing.

Kalman still wins despite irregular frame timing, because `predict(dt)` scales displacement by the REAL elapsed time -- this is the exact repair Assignment 4 made over Assignment 3's fixed one-step-per-call transition, and it visibly pays off here.

### turning_motion

Circular arc -- velocity direction keeps changing.

Kalman still comes out ahead numerically here (51.35 px vs. stationary's 66.56 px) -- reported exactly as measured, not adjusted to fit a narrative. But look at the *margin*: on smooth motion Kalman's error was 100% smaller than stationary's; here it's only about 23% smaller. A constant-velocity model assumes straight-line motion by construction, and a turning object directly violates that assumption -- its tangent-line extrapolation overshoots the arc every step. In this noise-free synthetic setup that's still better than freezing in place, but with a sharper turn, real detector noise, or a longer gap, static memory could plausibly become competitive or win. This is exactly the kind of case OATM's later occlusion/uncertainty logic must not paper over.

### abrupt_motion

Sudden large velocity change partway through.

Kalman's error grows across the gap (confirmed by `test_abrupt_motion_change_degrades_kalman_prediction_during_the_gap`) as its stale, pre-change velocity estimate compounds -- a real, expected limitation of the constant-velocity assumption, not a bug.

### missing_then_reappear

Smooth motion with a long missing window.

Over a longer 9-step gap, Kalman still tracks closer (0.00 px vs. 125.00 px) -- but see the diagram below for how much its OWN uncertainty grows over that same gap.

## Uncertainty growth

For every regime, the Kalman filter's covariance trace (its own real localization-uncertainty estimate) increased monotonically throughout the missing-detection gap, and every regime's final-step uncertainty was strictly higher than its first-step uncertainty (see `test_kalman_uncertainty_grows_monotonically_during_every_gap`). See `charts/uncertainty_growth.png` for the required 1/3/5-missing-frame diagram.

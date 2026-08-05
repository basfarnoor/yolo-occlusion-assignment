# Selective OATM Methodology

## Research question

Can a causal, camera-only occlusion gate extend ByteTrack only during plausible
occlusion and thereby improve hidden-object recall at equal or lower ghost
duration, without changing normal visible association?

## Primary hypothesis

ByteTrack is retained as the association backbone. A missing mature track is
admitted to `PREDICTED_HIDDEN` only when a different visible detection overlaps
its predicted location and plausibly covers it. A short grace period absorbs
isolated detector misses. Tracks near an outward image boundary are classified
as exits, and tracks without continuing occlusion support expire quickly.

This selective policy should outperform a longer unconditional ByteTrack
buffer at a matched ghost-risk operating point. Longer persistence alone is
not considered an OATM contribution.

## Online inputs and privileged evidence

The deployed method receives current and previous camera-derived detections and
its own track history only. It is causal. nuScenes visibility, LiDAR-supported
boxes, calibration, instance tokens, and ego poses may be used to construct or
score experiments, but never enter the online tracker.

## State and evidence contract

- `OBSERVED_STRONG`: current high-confidence detection.
- `OBSERVED_WEAK`: current low-confidence detection associated in ByteTrack's
  second stage.
- `PREDICTED_HIDDEN`: no matched detection; output is explicitly a temporal
  prediction supported by current occluder evidence or bounded grace.
- `LOST`: evidence is insufficient to report the track.
- `EXITED`: motion predicts departure through the image boundary.

Predictions are never described or scored as current visual detections.

## Selective occlusion gate

For an unmatched track, the gate uses:

1. Track maturity before disappearance.
2. Intersection over the predicted target area with an unclaimed current
   detection, rather than symmetric IoU alone.
3. An occluder scale constraint: a plausible foreground occluder should not be
   substantially smaller than the covered region.
4. Outward boundary motion as explicit exit evidence.
5. A bounded one-frame grace interval for ordinary detector instability.

The MVP deliberately excludes appearance memory because the previous OATM
ablation reduced same-ID recovery. It also excludes learned depth and future
frames.

## Termination policy

A selectively hidden track terminates when any of these conditions holds:

- Predicted outward boundary exit.
- No occluder support after the grace interval.
- Hidden duration exceeds the configured maximum.
- Localization uncertainty exceeds its ceiling.

Later work will add predicted occluder-clearance timing and causal camera-motion
compensation. These are not claimed as implemented here.

## Comparison methods

1. Frozen ByteTrack configuration.
2. ByteTrack with a longer unconditional buffer.
3. Selective OATM with identical detection thresholds and association logic.

## Primary metrics

- Hidden-frame coverage and fully bridged event rate.
- Same-ID recovery and wrong-object association.
- Mean localization error during hidden windows.
- Ghost event rate and ghost duration on verified exits/losses.
- Visible precision and recall.
- Runtime.

Natural occlusion, controlled visual occlusion, detector intervention, and
synthetic validation must always be reported separately.

## Acceptance criteria

The first real-data claim requires Selective OATM to exceed the best tuned
ByteTrack baseline in hidden recall at equal or lower ghost duration, preserve
or improve same-ID recovery, introduce no additional wrong associations, and
lose less than one percentage point of visible precision. Results must be
scene-disjoint and include event/scene-level uncertainty intervals.

Synthetic tests can validate mechanics but cannot satisfy this claim.

## Current scope and claim boundary

The present implementation is an engineering and synthetic-development
baseline. It does not yet demonstrate superiority on natural nuScenes
occlusions and makes no global novelty or autonomous-driving performance claim.

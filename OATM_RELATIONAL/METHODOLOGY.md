# Relational OATM Methodology

## Research question

Can an explicit, causal target--occluder relation improve the hidden-recall
versus ghost-duration frontier over tuned ByteTrack and Selective OATM on
driving video?

## Hypothesis

An unmatched mature target should persist only while a tracked visible
occluder explains its disappearance. Joint target/occluder prediction should
estimate when their boxes will separate; failure to reappear after that time
is evidence for termination. Causal background-derived camera compensation
should reduce localization drift caused by ego motion.

## Online camera-only boundary

Online inputs are current and earlier camera frames, camera-derived detections,
and tracker history. nuScenes 3D boxes, visibility, calibration, LiDAR/radar,
instance tokens, and recorded ego pose are privileged evaluation evidence only.
No future frame is available online.

## Relational state

Every supported hidden target may own one primary `OcclusionRelation`:

- `FORMING`: a candidate visible occluder explains disappearance.
- `ACTIVE`: overlap and motion remain consistent.
- `CLEARING`: predicted separation has occurred; reappearance is expected.
- `RESOLVED`: the original identity reconnects.
- `FAILED`: evidence contradicts persistence or reappearance does not occur.

The relation records target and occluder IDs, coverage, relative motion,
expected clearance, probability, timing, and camera-motion quality. Multiple
candidates are scored, but only one primary relation controls persistence.

At relation formation, the target box is encoded in an occluder-centric
coordinate system (normalized center offset and scale). While support remains
active, the current visible occluder decodes that anchor to bound hidden-box
drift. This is causal and camera-only; it does not use evaluation geometry.
The primary occluder is immutable for one hidden episode. On later frames, the
decoded anchor must remain within a configured center-residual and scale ratio
of the target's independent causal motion prediction. Inconsistent support
enters `CLEARING` instead of moving the target. Reappearance uncertainty has a
hard spatial cap, and a resolved relation is archived before the next frame so
it cannot terminate a new hidden episode.


## Camera-motion compensation

ORB features are extracted from background pixels outside tracked foreground
boxes. RANSAC estimates a causal partial-affine transform, while the MVP applies
only its bounded translation component; directly compounding per-frame scale
caused unacceptable long-sequence drift in the first real pilot. When quality
is insufficient, the identity transform is used and explicitly reported;
recorded nuScenes ego pose is never substituted into the headline method.

Camera compensation remains implemented as an ablation but is disabled in the
promoted configuration. A second natural-event pilot with translation-only
compensation still produced severe drift because camera translation was added
to a Kalman state whose image-space velocity already included that motion. It
must not be restored to the promoted method until state prediction and camera
motion are fused in one stabilized coordinate system and pass held-out tests.

## Occlusion probability

A deterministic, interpretable score combines target-relative coverage,
occluder scale, image-space depth ordering, trajectory agreement, track
maturity, and camera-motion quality. It is an engineering score, not a
calibrated posterior until held-out calibration is demonstrated.

## Expected clearance and termination

Target and occluder boxes are propagated over a bounded horizon. The earliest
step at which target coverage falls below the clearance threshold defines the
expected clearance. Persistence ends on predicted exit, unsupported relation,
excess uncertainty, maximum duration, or failure to reappear after the
clearance grace window.

## Reappearance association

After ByteTrack's strong and weak rounds, unmatched detections may reconnect a
relational hidden track using class agreement, uncertainty-normalized center
distance, scale consistency, and expected timing. One detection and one track
can be used at most once. Generic MobileNet appearance memory remains excluded
because it harmed the earlier held-out identity result.

Once a track is relationally hidden it is excluded from ordinary ByteTrack
association. Reconnection must pass the stricter third stage, and an active
visible occluder blocks reconnection to nearby same-class detections. This
prevents the identity hijack observed in the first natural pilot.

## Evaluation

Compare frozen ByteTrack, ByteTrack buffer sweep, Selective OATM, relational
ablations, and complete Relational OATM on identical observations. Report
synthetic, controlled visual, detector intervention, natural, and verified
negative events separately. Select the development configuration that maximizes
hidden recall subject to ghost duration no worse than the best ByteTrack point.

## Claim boundary

Synthetic experiments validate mechanics only. Real-data superiority requires
higher hidden recall at equal or lower ghost duration, no additional wrong
associations, equal or better same-ID recovery, and less than one percentage
point visible-precision loss on scene-disjoint evaluation.

## Current evidence status

Synthetic run `eac923a94d04` passes the mechanism-level acceptance checks:
complete Relational OATM reached 1.000 mean hidden coverage, 1.000 same-ID
recovery, 2.000 negative ghost frames, 4.403 px center error, and zero measured
wrong associations across eight deterministic scenarios.

Natural pilot `806945a64e0d` fixes the prior catastrophic localization drift:
Relational OATM center error is 16.019 px versus 15.858 for OATM_UPDATED and
20.672 for ByteTrack-12. On the two linkable events it reached 0.620 hidden
coverage, 0.500 fully bridged rate, and one same-ID recovery, compared with
0.430, 0.000, and zero for OATM_UPDATED. ByteTrack-12 still reached 0.760
coverage and one same-ID recovery. The pilot therefore supports an improvement
over OATM_UPDATED on these events, but it is too small and does not satisfy the

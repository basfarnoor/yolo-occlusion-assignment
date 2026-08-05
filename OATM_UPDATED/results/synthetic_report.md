# Selective OATM Synthetic Development Study

Run ID: `ec2393733008`. Runtime: 0.020s. Random seed: 42 (no stochastic draws).

## Question

Does evidence-gated persistence bridge a longer true occlusion than ByteTrack
while producing fewer stale predictions on ordinary misses and exits?

## Aggregate results

| method         |   mean_occlusion_coverage |   fully_bridged_rate |   same_id_recovery_rate |   mean_negative_ghost_duration_frames |   max_negative_ghost_duration_frames |
|:---------------|--------------------------:|---------------------:|------------------------:|--------------------------------------:|-------------------------------------:|
| bytetrack      |                     0.917 |                0.500 |                   0.500 |                                 5.000 |                                    5 |
| bytetrack_long |                     1.000 |                1.000 |                   1.000 |                                 6.000 |                                    6 |
| selective_oatm |                     1.000 |                1.000 |                   1.000 |                                 0.500 |                                    1 |

## Interpretation

This deterministic fixture validates the intended mechanism. It is not a
natural or controlled-visual nuScenes result and cannot establish real-world
superiority. The next required experiment must rerun the detector on
pixel-modified images and evaluate verified natural events and exits.

## Evidence boundary

- Inputs are synthetic detections, not camera pixels.
- Occluder boxes are current camera-like detections; no privileged labels enter
  a tracker.
- The same association thresholds and motion implementation are used by all
  methods.
- `bytetrack_long` tests whether unconditional extra lifetime alone is enough.

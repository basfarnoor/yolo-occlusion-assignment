# Relational OATM Extended Synthetic Study

Run ID: `eac923a94d04`. Scenarios: 8. Seed: 42.

## Question

Does explicit target--occluder memory, clearance reasoning, and causal camera
compensation improve the recovery--ghost tradeoff over ByteTrack and Selective OATM?

## Results

| method                  |   mean_occlusion_coverage |   fully_bridged_rate |   same_id_recovery_rate |   mean_center_error_px |   mean_negative_ghost_frames |   wrong_associations |   mean_runtime_ms_per_scenario |
|:------------------------|--------------------------:|---------------------:|------------------------:|-----------------------:|-----------------------------:|---------------------:|-------------------------------:|
| bytetrack_b12           |                     1.000 |                1.000 |                   1.000 |                  4.814 |                        6.333 |                    0 |                          1.415 |
| bytetrack_b5            |                     0.943 |                0.800 |                   0.600 |                  4.813 |                        5.000 |                    0 |                          8.539 |
| relational_camera       |                     1.000 |                1.000 |                   1.000 |                  2.816 |                        2.000 |                    0 |                         46.410 |
| relational_complete     |                     1.000 |                1.000 |                   1.000 |                  4.403 |                        2.000 |                    0 |                          1.768 |
| relational_no_clearance |                     1.000 |                1.000 |                   1.000 |                  4.403 |                        2.667 |                    0 |                          2.151 |
| selective_oatm          |                     0.920 |                0.600 |                   0.600 |                  4.013 |                        1.000 |                    0 |                          1.352 |

## Interpretation boundary

These deterministic synthetic scenarios validate mechanisms and ablations;
they do not establish nuScenes or general driving superiority. Natural,
controlled-visual, and negative real-image studies remain separate required evidence.

## Included stress cases

Short and long occlusion, moving and multiple occluders, abrupt camera pan,
ordinary miss, field-of-view exit, and failed expected reappearance.

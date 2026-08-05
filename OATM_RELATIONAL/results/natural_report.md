# Relational OATM Natural-Event Pilot

Run ID: `6c0cec3cd955`. Fresh detector artifacts were used. The reviewed OATM
manifest contains 6 accepted events; conclusions apply only to
events that each method could link at the pre-occlusion reference frame.

| method              |   linked_events |   mean_hidden_coverage |   fully_bridged_rate |   same_id_recoveries |   new_id_recoveries |   not_recovered |   mean_center_error_px |   runtime_seconds |
|:--------------------|----------------:|-----------------------:|---------------------:|---------------------:|--------------------:|----------------:|-----------------------:|------------------:|
| bytetrack_b12       |               2 |                  0.760 |                0.500 |                    1 |                   1 |               0 |                 20.672 |             1.380 |
| bytetrack_b5        |               2 |                  0.245 |                0.000 |                    0 |                   2 |               0 |                 10.272 |             1.137 |
| relational_camera   |               2 |                  1.000 |                1.000 |                    0 |                   2 |               0 |                294.494 |           376.132 |
| relational_complete |               2 |                  1.000 |                1.000 |                    0 |                   2 |               0 |                307.126 |            23.410 |
| selective_oatm      |               2 |                  0.430 |                0.000 |                    0 |                   2 |               0 |                 15.858 |             1.064 |

This is a nuScenes-mini pilot, not a statistically powered superiority claim.
LiDAR-supported projection is privileged evaluation evidence only.

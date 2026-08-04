# Natural Event Selection Log

## Automated mining

Candidates require BOTH an official visibility-label decline-then-recovery AND a plausible closer occluder (another instance overlapping the target's box, sitting nearer the camera by `center_depth_m`) in the same frame -- two independent signals, per this task's requirement.

- Instances with a decline-recovery pattern AND a plausible occluder (auto-accepted as candidates): **54**
- Instances with a decline-recovery pattern but NO plausible occluder found (auto-rejected, logged, never shown for review): **19**
- Shortlisted for human review (top 15 by run length, then occluder overlap): **15**

## Student review (verbatim, recorded exactly as given -- not invented)

- Accepted: **6**
- Unsure: **8**
- Rejected: **1**

| # | Scene (split) | Class | Review | Reason (student's words) |
|---|---|---|---|---|
| 1 | `6f83169d` (development) | car | **unsure** | unclear to my eye |
| 2 | `fcbccedd` (validation) | car | **unsure** | unclear to my eye |
| 3 | `fcbccedd` (validation) | car | **accepted** | the rest are good |
| 4 | `cc8c0bf5` (development) | car | **unsure** | unclear to my eye |
| 5 | `cc8c0bf5` (development) | pedestrian | **unsure** | unclear to my eye |
| 6 | `6f83169d` (development) | car | **unsure** | unclear to my eye |
| 7 | `6f83169d` (development) | pedestrian | **accepted** | the rest are good |
| 8 | `de7d80a1` (development) | pedestrian | **accepted** | the rest are good |
| 9 | `fcbccedd` (validation) | pedestrian | **accepted** | the rest are good |
| 10 | `6f83169d` (development) | car | **unsure** | more than one occluder -- too much overlap to confidently judge this as one event |
| 11 | `6f83169d` (development) | pedestrian | **unsure** | more than one occluder -- too much overlap to confidently judge this as one event |
| 12 | `cc8c0bf5` (development) | pedestrian | **unsure** | unclear to my eye |
| 13 | `cc8c0bf5` (development) | pedestrian | **accepted** | the rest are good |
| 14 | `de7d80a1` (development) | car | **rejected** | wrong objects |
| 15 | `fcbccedd` (validation) | car | **accepted** | the rest are good |

Only `accepted` events are eligible for OATM MVP evaluation (Task 11). `unsure` events are kept in the manifest for traceability but excluded from the accepted evaluation set -- they are not silently promoted to accepted. This mini result is a **pilot**, not a final statistical conclusion: 15 reviewed candidates, 6 accepted, is a small sample from one dataset split.

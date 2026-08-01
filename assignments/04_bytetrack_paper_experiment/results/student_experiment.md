# Student Experiment

**Student's exact choice:** "Let lost tracks wait longer before removal."

**Student's prediction, in her own words:** "cost us" -- i.e. she predicted that letting tracks wait longer would cost more wrong reconnections/false associations, rather than being a clean improvement.

**Configuration change:** `track_buffer` raised from **5** to **10** frames (and `sort_baseline.max_age` raised identically, to keep the SORT-vs-ByteTrack comparison fair -- both methods must share the same buffer).

**What stayed fixed:** `detection_floor`, `high_score_threshold`, `new_track_threshold`, both IoU thresholds, the detector, the clips, and the 5 selected target tracks. Only the window-length-7 complete-absence trial was rerun (the case where the buffer boundary actually matters) -- cached detections were reused, no YOLO re-run.

## Results before vs. after

| Target track | Method | Reconnected (buffer=5) | Reconnected (buffer=10) | False associations in window (buffer=5) | False associations in window (buffer=10) |
|---|---|---|---|---|---|
| `clip_sample_001_track10` | high_confidence_sort | False | True | 2 | 0 |
| `clip_sample_001_track10` | bytetrack | True | True | 0 | 0 |
| `clip_sample_001_track3` | high_confidence_sort | False | True | 0 | 0 |
| `clip_sample_001_track3` | bytetrack | True | True | 0 | 0 |
| `clip_sample_001_track9` | high_confidence_sort | False | True | 0 | 0 |
| `clip_sample_001_track9` | bytetrack | False | True | 0 | 0 |
| `clip_sample_011_track31` | high_confidence_sort | False | False | 0 | 0 |
| `clip_sample_011_track31` | bytetrack | False | False | 0 | 0 |
| `clip_sample_011_track32` | high_confidence_sort | False | True | 0 | 0 |
| `clip_sample_011_track32` | bytetrack | False | False | 0 | 0 |

**5** (method, track) pairs that failed to reconnect at `track_buffer=5` succeeded at `track_buffer=10`.
**0** (method, track) pairs picked up MORE in-window false associations at `track_buffer=10` than at `track_buffer=5`.

## Was the prediction supported?

**Not in this sample -- and the reason why is worth explaining.** No (method, track) pair picked up
*additional* false associations at `track_buffer=10`. In fact, `clip_sample_001_track10`'s two false
associations at `track_buffer=5` (high-confidence SORT) **disappeared** at `track_buffer=10`.

Looking at the mechanism: at `track_buffer=5`, SORT's original track died partway through the
7-frame absence window (it exceeded the buffer), and a coincidentally-nearby detection with poor
overlap to the true target got wrongly attached to a *newly-born* track -- a false association
caused by premature death-and-rebirth, not by the original track drifting off target. At
`track_buffer=10`, the same original track simply survived the whole window on motion prediction
alone, so there was no death, no rebirth, and nothing for a wrong detection to attach to. The longer
buffer removed the failure mode by preventing the gap between losing a track and a new one starting.

So the student's prediction -- that waiting longer would "cost us" more wrong reconnections -- named
a real and plausible risk (a track that drifts for longer on pure guesswork has more chances to grab
the wrong nearby object once real evidence returns), but in this specific small sample that risk did
not materialize; instead, the opposite failure mode (dying too early and letting something else take
its place) turned out to be the one actually happening, and a longer buffer fixed it. Both
predictions -- "this could go wrong via more false associations" and the unstated alternative "this
could go right by preventing premature death" -- are valid readings of the same underlying
mechanism, and this small experiment happened to land on the second one. With only 5 target tracks,
neither direction should be treated as the general rule.

## Connection to the ByteTrack paper

The paper's `track_buffer` (called the lost-track survival window) is exactly this parameter. The paper does not claim a longer buffer is free -- keeping a track alive longer without real evidence is a bet that the object will return before something else takes its place, and the student's prediction names precisely the risk the paper's track-state design (tracked/lost/removed) exists to manage.

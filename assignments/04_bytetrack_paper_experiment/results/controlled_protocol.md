# Controlled Experiment Protocol

## Target selection

Eligibility pass: 5 of 42 natural targets eligible (min_track_length=14, min_confidence=0.5).
  - clip_sample_001 track 0: rejected (begins or ends at the image boundary)
  - clip_sample_001 track 1: rejected (only 12 frames, needs >= 14)
  - clip_sample_001 track 2: rejected (begins or ends at the image boundary)
  - clip_sample_001 track 3: ELIGIBLE (ok)
  - clip_sample_001 track 4: rejected (average raw confidence 0.31 below 0.5)
  - clip_sample_001 track 5: rejected (average raw confidence 0.50 below 0.5)
  - clip_sample_001 track 6: rejected (only 4 frames, needs >= 14)
  - clip_sample_001 track 7: rejected (average raw confidence 0.30 below 0.5)
  - clip_sample_001 track 8: rejected (only 7 frames, needs >= 14)
  - clip_sample_001 track 9: ELIGIBLE (ok)
  - clip_sample_001 track 10: ELIGIBLE (ok)
  - clip_sample_001 track 11: rejected (only 13 frames, needs >= 14)
  - clip_sample_001 track 12: rejected (only 9 frames, needs >= 14)
  - clip_sample_003 track 13: rejected (begins or ends at the image boundary)
  - clip_sample_003 track 14: rejected (class 'traffic light' not in allowed set ('car', 'truck', 'bus', 'person', 'bicycle', 'motorcycle'))
  - clip_sample_003 track 15: rejected (begins or ends at the image boundary)
  - clip_sample_003 track 16: rejected (only 8 frames, needs >= 14)
  - clip_sample_003 track 17: rejected (only 1 frames, needs >= 14)
  - clip_sample_003 track 18: rejected (only 3 frames, needs >= 14)
  - clip_sample_003 track 19: rejected (begins or ends at the image boundary)
  - clip_sample_003 track 20: rejected (only 9 frames, needs >= 14)
  - clip_sample_003 track 21: rejected (only 4 frames, needs >= 14)
  - clip_sample_003 track 22: rejected (only 2 frames, needs >= 14)
  - clip_sample_006 track 23: rejected (begins or ends at the image boundary)
  - clip_sample_006 track 24: rejected (only 8 frames, needs >= 14)
  - clip_sample_006 track 25: rejected (begins or ends at the image boundary)
  - clip_sample_006 track 26: rejected (only 9 frames, needs >= 14)
  - clip_sample_006 track 27: rejected (begins or ends at the image boundary)
  - clip_sample_011 track 28: rejected (begins or ends at the image boundary)
  - clip_sample_011 track 29: rejected (average raw confidence 0.32 below 0.5)
  - clip_sample_011 track 30: rejected (only 12 frames, needs >= 14)
  - clip_sample_011 track 31: ELIGIBLE (ok)
  - clip_sample_011 track 32: ELIGIBLE (ok)
  - clip_sample_011 track 33: rejected (only 1 frames, needs >= 14)
  - clip_sample_011 track 34: rejected (only 9 frames, needs >= 14)
  - clip_sample_011 track 35: rejected (only 13 frames, needs >= 14)
  - clip_sample_011 track 36: rejected (only 2 frames, needs >= 14)
  - clip_sample_011 track 37: rejected (only 1 frames, needs >= 14)
  - clip_sample_011 track 38: rejected (only 1 frames, needs >= 14)
  - clip_sample_011 track 39: rejected (only 6 frames, needs >= 14)
  - clip_sample_011 track 40: rejected (only 7 frames, needs >= 14)
  - clip_sample_011 track 41: rejected (only 6 frames, needs >= 14)
Final selection: 5 target(s): clip_sample_001#3, clip_sample_001#9, clip_sample_001#10, clip_sample_011#31, clip_sample_011#32

## Experiment A -- confidence demotion

For each eligible target, a centered window of length in [1, 2, 3] frames is chosen (deterministically, from the middle of the target's natural detection span). The target's raw YOLO box is kept completely unchanged; only its confidence score is overwritten to **0.2** (inside the low-confidence band) for exactly those frames. No other detection -- the target's own detections outside the window, or any other object's detections anywhere -- is touched. Both a fresh SortTracker and a fresh ByteTrackTracker are then run over the ENTIRE clip (every frame, in chronological order, real lifecycle enforced), and the tracker's output is compared against the target's ORIGINAL, undemoted raw YOLO box -- labeled **pseudo-ground-truth** throughout, never manually verified ground truth.

Windows tested: 15 demotion events across 5 targets.

## Experiment B -- complete detection absence

Same target tracks, windows of length in [1, 2, 3, 7] frames. The target's detection row is REMOVED ENTIRELY for those frames (not demoted) -- other objects' detections are left untouched, so ordinary false-association risk stays live. Window length 7 is deliberately longer than the configured `track_buffer` (5) to test whether the track is genuinely allowed to expire, rather than surviving by construction as in Assignment 3.

Windows tested: 20 absence events across 5 targets.

## Identity measurement (repairs Assignment 3's hardcoded ID continuity)

The target's track ID immediately before each window is found by matching tracker output boxes against the target's own known raw box (best IoU, same class) -- a legitimate, honest lookup, never a hardcoded assumption. `id_continuous_from_before_window` in controlled_trials.csv is only True when the SAME numeric track ID the real tracker assigned before the window is still present afterward -- if the track expired or a different track claims the location, this is correctly recorded as False, not silently assumed.

## Repeated-row note

Each (target, mode, window length) combination produces one row per evaluated frame per method. The main experimental unit for analysis is the (target, mode, window length) EVENT, not the frame row -- `summary_by_event.csv` and `summary_by_track.csv` (Task 11) report counts at both levels explicitly, never only the larger row count.

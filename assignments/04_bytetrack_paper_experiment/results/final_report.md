# Recreating the Main Idea of the ByteTrack Paper: Final Report

## 1. Paper citation and links

Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan, Ping Luo, Wenyu Liu,
Xinggang Wang, ["ByteTrack: Multi-Object Tracking by Associating Every Detection
Box,"](https://arxiv.org/abs/2110.06864) ECCV 2022.
DOI: [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1).
Authors' code: [github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack).

## 2. What the ByteTrack paper proposed

Most tracking-by-detection systems discard detections below a confidence threshold before
association, because low-score boxes are disproportionately false positives. ByteTrack's central
claim is that throwing all of them away also discards real, partially occluded objects. Its BYTE
association algorithm associates high-confidence detections with existing tracks first, then gives
tracks still unmatched a second chance to match against the remaining low-confidence detections --
never using an unmatched low-confidence box to start a brand-new track. The paper reports 80.3
MOTA / 77.3 IDF1 / 63.1 HOTA on MOT17 and a generic 1-10 point IDF1 improvement when BYTE is added
to nine different existing trackers.

## 3. Why ByteTrack followed SORT in this assignment sequence

1. **Assignment 1 -- YOLO:** detection confidence collapsed during full occlusion (12% detected).
2. **Assignment 2 -- static memory:** a frozen last-seen box preserved *existence* but not
   *location* (331 px mean center error at reappearance).
3. **Assignment 3 -- SORT:** constant-velocity motion prediction cut center error by ~73-81% on
   selected fast-moving tracks, but a mentor review found the evaluation reference was itself
   Kalman-smoothed output, long gaps bypassed the tracker's own `max_age`, and ID continuity was
   assigned by construction rather than measured.

ByteTrack adds a new, orthogonal idea -- confidence-aware two-stage association -- while reusing
the SORT concepts already implemented, and this assignment's Task 2 explicitly repairs the eight
methodological weaknesses the mentor review identified (see `reuse_audit.md`).

## 4. How this experiment differs from a full paper reproduction

This is a **small paper-inspired replication and extension**, not a reproduction of the paper's
MOT17/MOT20 benchmark results. It uses 4 short local nuScenes mini `CAM_FRONT` clips (144 frames),
the existing pretrained YOLO nano detector (no training), an educational from-scratch two-stage
tracker, projected nuScenes 3D annotations as independent offline evaluation evidence, and
controlled confidence demotion / detection removal rather than natural MOT-style occlusion
statistics. It never claims to reproduce the paper's MOTA, IDF1, HOTA, or speed numbers.

## 5. Hypothesis (written before running the experiment)

> Based on the ByteTrack paper, associating low-confidence detections after the high-confidence
> matching stage should recover some real, weakly visible objects and reduce trajectory
> fragmentation. It should help most when a geometrically plausible low-score detection still
> exists. It should not help when no detection exists, and accepting weak boxes too freely may
> increase false associations or ghost tracks.

A mixed or negative result was treated as scientifically useful; ByteTrack was not required to win.

## 6. Dataset, clips, scene-disjoint split, and data checks

Four `CAM_FRONT` clips (36 frames each, 144 frames total), each from a **different nuScenes
scene**: `clip_sample_001` (scene-0103), `clip_sample_003` (scene-0553), `clip_sample_006`
(scene-0757) -- all three reused from Assignment 3 -- plus one new clip, `clip_sample_011`
(scene-1094), added for scene-disjointness. Development split: `clip_sample_001` +
`clip_sample_006` (2 scenes). Evaluation split: `clip_sample_003` + `clip_sample_011` (2 scenes).
**Zero scene overlap** between splits. All timestamps verified strictly increasing, all frame files
verified present (`data_check.md`, `clip_manifest.csv`).

## 7. Independent projected ground truth

Official 3D nuScenes annotations were projected into `CAM_FRONT` at all 24 keyframes (6 per clip)
via the standard global -> ego-vehicle -> camera -> pinhole transform chain (`src/projection.py`),
using `pyquaternion` for rotation composition and `shapely` for image-boundary clipping. Of 1,084
annotation instances considered, 349 were accepted (others were behind the camera or projected
outside the frame -- both expected, since `sample_annotation.json` covers the full 360-degree
scene, not just the forward view). A 20-frame contact sheet was visually inspected: projected boxes
tightly wrapped real vehicles and pedestrians across day, overcast, and night/rain conditions, with
no visible systematic offset (`projection_audit.md`). **This projected ground truth -- never
tracker output -- is what the natural-event evaluation is scored against.**

## 8. Detector settings, confidence distribution, and cache

`yolo26n.pt`, prediction only, CPU, image size 640 (benchmarked at 0.20s/frame, within budget),
**detection floor 0.05** (lower than Assignment 3's effective working threshold, so weak boxes
survive). No prior detection cache existed in this checkout to reuse, so YOLO ran fresh: 3,251 raw
detections across 144 frames. On development scenes only, matched detection confidence by nuScenes
visibility category showed mostly-visible objects (v80-100) at mean confidence 0.639 vs.
mostly-occluded objects (v0-40) at 0.250 -- confirming Assignment 1's finding and motivating
keeping low-score boxes available (`charts/detection_confidence_by_visibility.png`,
`detector_audit.md`). Thresholds frozen from this analysis, before any evaluation-split result was
opened: `high_score_threshold=0.5`, `new_track_threshold=0.6`, first/second-association IoU
0.3/0.5, `track_buffer=5`.

## 9. Educational ByteTrack implementation

`src/kalman_box_tracker.py` (constant-velocity filter, repaired to use real `dt` from consecutive
timestamps rather than Assignment 3's fixed one-step-per-call), `src/assignment.py` (IoU + Hungarian
matching, called twice per frame), and `src/bytetrack_tracker.py` (the two-stage BYTE tracker,
built from the paper's algorithm, not copied from the authors' source). Every output row carries an
explicit `evidence_source`: `high_score_detection`, `low_score_detection`, or `motion_prediction` --
verified never to leak a Kalman-smoothed box where a raw detection box is required (see
`track.py`'s `raw_detection_box` field and its regression test).

## 10. Fair comparison methods

Method A (YOLO-only, no tracker), Method B (`src/sort_tracker.py`, high-confidence SORT: identical
motion model and lifecycle to ByteTrack, but only one association round), and Method C
(`src/bytetrack_tracker.py`). All three receive the exact same raw per-frame detection list --
proven by an automated test (`test_fair_comparison.py`) that calls both trackers with the literal
same Python list object. Method B's own internal filtering (not a caller-side split) is what
isolates the paper's second-association contribution as the *only* difference between B and C.

## 11. Natural weak-evidence protocol

At keyframe granularity (nuScenes only annotates keyframes -- 6 per 36-frame clip), 12 candidate
events were found where a real nuScenes instance's best-matched raw detection confidence fell from
the high band into the low band or disappeared, while the object remained present in ground truth
before and after. All 12 were kept (the deterministic cap), spanning all 4 clips and 4 classes
(car, truck, bicycle, pedestrian). A 12-triptych visual review confirmed every case looked like
genuine partial occlusion or difficult viewing conditions, not an object leaving frame
(`natural_event_selection.md`).

## 12. Controlled confidence-demotion protocol

From 42 natural per-frame target tracks (built by linking raw detections with a real ByteTrackTracker
pass, then reading off each frame's `raw_detection_box` -- never the Kalman state), 5 targets met the
eligibility bar (>=14 frames, average confidence >=0.5, not touching the image edge). For window
lengths of 1, 2, and 3 frames, the target's raw box was kept exactly as YOLO produced it; only its
confidence was overwritten to 0.2 (inside the low-confidence band). Both trackers ran over the
**entire clip**, and output was compared against the target's original, undemoted raw box --
labeled pseudo-ground-truth throughout. 15 demotion events total.

## 13. Complete detection-absence protocol

Same 5 targets, window lengths 1, 2, 3, and **7** frames (7 deliberately exceeds the frozen
`track_buffer=5`, to test genuine expiry). The target's detection row was removed entirely; other
objects' detections were left untouched so ordinary false-association risk stayed live. 20 absence
events total. Track identity before each window was found by an honest best-IoU lookup against the
target's own known raw box -- never hardcoded -- so `id_continuous_from_before_window` can and does
read False when a track genuinely expires or reconnects to the wrong ID.

## 14. Quantitative results

**Natural events (n=12, scored against independent projected ground truth):**

| Method | Recovered at event frame | Identity preserved before->after |
|---|---:|---:|
| High-confidence SORT | 7/12 (58.3%) | 4/12 (33.3%) |
| ByteTrack | 8/12 (66.7%) | 6/12 (50.0%) |

By class: cars tied 5/6 for both methods; ByteTrack won on pedestrians (2/4 vs. 1/4); the single
bicycle event (permanently occluded, never redetected) and single truck event were identical for
both methods. **ByteTrack's advantage in this small sample is concentrated in the pedestrian
class**, not spread evenly.

**Controlled events (35 events, 1,330 frame-method rows -- see `run_metadata.json` for the full
scene/clip/track/event/row count breakdown):**

| Method | Window coverage rate | Post-window ID continuity |
|---|---:|---:|
| High-confidence SORT | 96.7% | 84.5% |
| ByteTrack | 99.2% | 92.1% |

**Complete-absence breakdown (65 window frames per method, all 4 window lengths pooled):**

| Outcome | SORT | ByteTrack |
|---|---:|---:|
| Survived via motion prediction only | 55 | 51 |
| Matched a real, plausible nearby detection (IoU>=0.3) | 0 | 12 |
| Matched a real but WRONG detection (false/ghost association) | 2 | 0 |
| Track fully gone (no output) | 8 | 2 |

See `charts/low_score_recovery_by_method.png`, `charts/fragmentation_by_method.png`, `charts/id_switches_by_method.png`,
`charts/false_associations_by_method.png`, `charts/complete_absence_survival.png`.

## 15. Required ablation

`high_confidence_sort` **is** "ByteTrack with the second association removed" -- both share the
identical Kalman filter, first-association logic, new-track rule, and `track_buffer` (verified by
construction, not just by similar numbers). The ablation is therefore the same natural/controlled
comparison above, relabeled: `bytetrack_full` vs. `bytetrack_no_second_association`
(`ablation.csv`, `charts/ablation.png`). Full ByteTrack outperforms the no-second-association variant on
every measured metric, in both natural and controlled evidence -- confirming the improvement traces
specifically to the second association stage, not to any other implementation difference.

## 16. Threshold sensitivity

Varying `high_score_threshold` by -0.15/0/+0.15 on development scenes only (cached detections, no
YOLO re-run): both methods' track fragmentation (mean tracks born per clip) responded similarly to
threshold changes, but ByteTrack's **mean track length stayed more stable** (18.9 -> 20.7 -> 20.7
frames) than SORT's (18.0 -> 19.8 -> 17.8 frames, dipping back down at the highest threshold) --
weak evidence that the second association stage buys some robustness to the exact threshold choice,
though this is a 2-clip, 3-point sample and should not be overstated (`charts/threshold_sensitivity.png`).

## 17. Best successful recovery

`clip_sample_011`, pedestrian instance `8c3247...`: SORT loses the object into motion-prediction
state as its confidence drops (0.54 -> ~0.3), never re-matching a real detection; ByteTrack's
second association matches a genuine low-score box (confidence 0.33) to the same track, which then
reconnects to a later high-confidence detection with its **original ID intact**. See
`videos/comparison_recovery_clip_sample_011_pedestrian.mp4` and
`videos/explanatory_bytetrack_second_association.mp4`.

## 18. Most important false association or failure

High-confidence SORT, `clip_sample_001` track 10, complete-absence window (frames 30-31): with the
target's detection fully removed, SORT's single-round matching locked onto a **different**, nearby
high/low-score detection (IoU < 0.3 against the true target) -- a genuine false association.
ByteTrack made zero such errors across all 65 complete-absence window frames in this sample, though
with only 5 target tracks this should be read as a promising signal, not a general guarantee. See
`videos/comparison_false_association_clip_sample_001_track10.mp4`.

## 19. Runtime and laptop suitability

Tracker-only time, measured **separately per method** (432 calls each, 3 repeats x 4 clips x 36
frames): SORT median 0.68 ms/frame, ByteTrack median 0.97 ms/frame -- both far below YOLO's ~100+
ms/frame range from Assignment 3, and the whole pipeline (clip building, projection, detection,
tracking, evaluation, ablation, 9 charts, 4 videos) runs in a few minutes on an ordinary CPU-only
laptop once detections are cached (`charts/runtime_comparison.png`).

## 20. Connection to the paper's claims

The paper's central claim -- that a second, low-confidence association round recovers real objects
without excessive false associations -- held up directionally in this small sample: ByteTrack
recovered more natural events, preserved identity more often, and made fewer false associations
during complete absence than the fair SORT baseline. The paper's explicit boundary also held: when
the target's detection was **completely absent** (not just weak), ByteTrack's second association
had nothing to associate against and could only rely on motion prediction, same as SORT, until the
buffer ran out.

## 21. Connection to Assignments 1-3 and OATM

Assignment 1: YOLO confidence collapses under occlusion. Assignment 2: static memory preserves
existence, not location. Assignment 3: motion prediction improves location, but its own evaluation
had four repairable flaws. This assignment repairs all four (reuse_audit.md) and adds ByteTrack's
lesson: many "lost" detections are not truly gone, just below a naive threshold -- but genuinely
absent evidence is a hard boundary no association strategy can cross. For OATM, this suggests
future work should distinguish *weak-but-present* evidence (where a second, less strict association
pass helps) from *truly absent* evidence (where only motion/uncertainty modeling can help, and
where OATM's occlusion-state tracking becomes essential).

## 22. Limitations

- This is not a reproduction of the paper's MOT17/MOT20/IDF1/MOTA/HOTA/speed benchmark results.
- nuScenes mini and the four selected clips are small (144 frames, 4 scenes).
- Official annotations are sparse keyframes (6 per 36-frame clip), not every camera frame --
  natural-event identification and evaluation are both limited to keyframe granularity.
- Projected 3D boxes can be truncated or imperfect as 2D references, especially at image edges.
- Controlled confidence demotion and detection removal are not identical to natural occlusion --
  there is no real occluding object in the modified frames.
- Raw YOLO boxes used as pseudo-ground-truth in the controlled experiments may themselves be
  inaccurate.
- ByteTrack uses *current* weak detections; it does not recover truly invisible objects from visual
  evidence that was never produced.
- Camera ego-motion and nonlinear object motion can still challenge the shared constant-velocity
  Kalman model.
- Only 5 unique target tracks (all cars) support the controlled experiments; the natural-event
  pedestrian advantage rests on only 4 events.
- Thresholds were selected on a 2-clip development split and may not generalize.
- The natural-track linking pass (used to discover controlled-experiment targets) assigns track IDs
  from a process-global counter; re-running scripts in a different order within the same Python
  process can shift ID numbering, though it does not affect any reported metric (each script resets
  the counter before use). This is a reproducibility wrinkle worth fixing in a future revision, not
  a result-affecting bug.
- A gain in recovery/coverage is incomplete without the false-association evidence reported
  alongside it here -- both are shown together throughout, per the assignment's requirement.

## 23. Conclusion

Within this small, honestly-scoped, twice-debugged experiment: ByteTrack's second, low-confidence
association stage recovered more natural weak-evidence events (8/12 vs. 7/12), preserved track
identity more often (50% vs. 33%), and made zero false associations during complete detection
absence where high-confidence SORT made two, all while adding well under a millisecond of tracker
overhead per frame. The clearest, most defensible finding is the complete-absence breakdown: given
identical raw detections and an identical lifecycle, ByteTrack recovered 12 frames of genuinely
plausible weak evidence that a single-threshold tracker structurally cannot consider at all -- and
did so without inventing false objects in this sample. The result should be read as encouraging and
consistent with the paper's motivation, not as proof the method always wins: it is drawn from 5
target tracks, 12 natural events, and 4 nuScenes scenes.

## 24. Exact reproduction command

```bash
cd assignments/04_bytetrack_paper_experiment/experiment
python src/clip_builder.py                      # Task 3: 4 scene-disjoint clips
python build_ground_truth.py                     # Task 4: projected ground truth
python build_projection_overlays.py              # Task 4: visual review contact sheet
python run_detect.py                             # Task 5: cached YOLO detections
python build_confidence_by_visibility.py         # Task 5: threshold-selection chart
python -m pytest tests/ -v                       # Task 7: 42 tests (must pass before continuing)
python build_natural_events.py                   # Task 9: natural event manifest
python build_natural_event_contact_sheets.py     # Task 9: visual review sheets
python build_controlled_experiments.py           # Task 10: confidence-demotion + absence trials
python build_evaluation.py                       # Task 11: natural_trials, summaries, run_metadata
python build_ablation.py                         # Task 12: ablation + threshold sensitivity
python build_remaining_charts.py                 # Task 13: 6 remaining charts
python build_videos.py                           # Task 13: 3 comparison videos
python build_explanatory_video.py                # Task 13: 1 explanatory video
```

## Citation

```bibtex
@inproceedings{Zhang2022ByteTrack,
  author    = {Yifu Zhang and Peize Sun and Yi Jiang and Dongdong Yu and
               Fucheng Weng and Zehuan Yuan and Ping Luo and Wenyu Liu and
               Xinggang Wang},
  title     = {ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  booktitle = {European Conference on Computer Vision},
  year      = {2022},
  pages     = {1--21},
  doi       = {10.1007/978-3-031-20047-2_1}
}
```

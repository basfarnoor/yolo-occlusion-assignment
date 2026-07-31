# Assignment 4: Recreate the Main Idea of the ByteTrack Paper

> This file is both your assignment and your prompt for an LLM coding assistant
> such as Claude Code or Codex.
>
> Give the assistant this entire file. You are **not expected to write Python,
> edit configuration files, use the terminal, label hundreds of boxes, or type
> Git commands**. The assistant must do the coding, testing, execution, chart
> creation, and Git work. Your job is to understand the paper's main idea,
> inspect a few generated visualizations, make one experimental choice, and
> explain what you learned in your own words.

## The Paper

This assignment is based on:

> Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan,
> Ping Luo, Wenyu Liu, and Xinggang Wang,
> **“ByteTrack: Multi-Object Tracking by Associating Every Detection Box,”**
> ECCV 2022.

- [Read the ByteTrack paper on the official ECCV Open Access site](https://www.ecva.net/papers/eccv_2022/papers_ECCV/html/315_ECCV_2022_paper.php)
- [Open the official paper PDF](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820001.pdf)
- [Read the paper on arXiv](https://arxiv.org/abs/2110.06864)
- [See the authors' official ByteTrack repository](https://github.com/FoundationVision/ByteTrack)
- DOI: [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1)

Throughout this assignment, **“the paper” means the ByteTrack paper linked
above**. Every generated paper map, experiment report, and final explanation
must link back to the paper.

## Why ByteTrack Is the Right Fourth Paper

The earlier assignments form this progression:

1. **Assignment 1 — YOLO:** confidence fell as objects became hidden, and YOLO
   often stopped reporting fully occluded targets.
2. **Assignment 2 — static memory:** keeping the last box preserved the idea
   that an object existed, but the unmoving box quickly became stale.
3. **Assignment 3 — SORT:** constant-velocity prediction moved the remembered
   box and often improved short-gap localization, but long gaps, real tracker
   expiry, identity continuity, and independent ground truth still needed more
   rigorous evaluation.

ByteTrack asks a different and very practical question:

> What if an occluded object has not disappeared completely, but its detection
> confidence has fallen below the threshold that a normal tracker accepts?

Many tracking systems discard low-confidence detections because low scores can
be false positives. The ByteTrack paper argues that throwing all of them away
also discards real, partly occluded objects. Its main idea is to associate
high-confidence detections first, then give the still-unmatched tracks a second
chance to match low-confidence detections.

This makes ByteTrack a better next assignment than OC-SORT at this stage.
OC-SORT is an excellent later paper for improving SORT under long occlusion and
nonlinear motion, but Assignment 3 already introduced motion-model failure.
ByteTrack adds a new lesson—**confidence-aware two-stage association**—while
reusing the SORT concepts the student already knows.

ByteTrack also teaches an important limit:

> A low-confidence box is still current visual evidence. If the detector
> produces no usable box at all, ByteTrack cannot create visual evidence from
> nothing; it can only rely temporarily on its motion model and track buffer.

That boundary is directly relevant to the larger OATM project.

## Important Scientific Description

This is a **small paper-inspired replication and extension**, not a reproduction
of the ByteTrack paper's published benchmark results.

The paper trained and evaluated its system on established multi-object tracking
benchmarks, including MOT17 and MOT20, and paired BYTE association with a strong
detector. This assignment will instead:

- Use small chronological `CAM_FRONT` clips from local nuScenes mini data.
- Use the existing pretrained YOLO detections rather than train a new detector.
- Implement an educational version of the paper's two-stage BYTE association.
- Compare high-confidence SORT with ByteTrack on identical detections.
- Use projected nuScenes annotations as privileged evaluation evidence at
  annotated keyframes.
- Use controlled confidence demotion to isolate the paper's main idea.
- Test complete detection absence separately from low-confidence visibility.

The final report must use this exact scientific description. It must never
claim that the assignment reproduced the paper's MOT17, MOT20, IDF1, MOTA,
HOTA, or speed results.

---

## Objectives

By the end of the assignment, you should be able to:

1. Explain why a low-confidence detection is not automatically a false object.
2. Explain the difference between a high detection threshold and a low
   detection floor.
3. Describe ByteTrack's first and second association stages in simple words.
4. Explain why high-confidence detections are matched before low-confidence
   detections.
5. Explain why unmatched low-confidence detections must not start new tracks.
6. Compare YOLO-only, high-confidence SORT, and ByteTrack on identical frames.
7. Distinguish partial visibility from complete detection absence.
8. Measure recovered tracks and harmful false associations together.
9. Explain identity continuity, fragmentation, false tracks, and ghost tracks.
10. Understand why an evaluation reference must be independent of the tracker.
11. Read charts with sample counts and identify conditional rather than
    universal findings.
12. Complete a paper-based experiment by directing an LLM rather than writing
    code manually.

## Research Question

> **Does ByteTrack's second association with low-confidence detections improve
> trajectory continuity during weak or partially occluded visual evidence,
> compared with high-confidence SORT, without creating an unacceptable number
> of false tracks or incorrect associations?**

## Hypothesis

The assistant must place this hypothesis at the top of the generated final
report before running the experiment:

> Based on the ByteTrack paper, associating low-confidence detections after the
> high-confidence matching stage should recover some real, weakly visible
> objects and reduce trajectory fragmentation. It should help most when a
> geometrically plausible low-score detection still exists. It should not help
> when no detection exists, and accepting weak boxes too freely may increase
> false associations or ghost tracks.

A mixed or negative result is scientifically useful. ByteTrack is not required
to win.

---

# Instructions for the LLM Coding Assistant

## Your Role

Complete the assignment autonomously, one numbered task at a time.

- Do all programming, setup, terminal work, testing, analysis, visualization,
  and Git work yourself.
- Never ask the student to write, paste, or edit code or commands.
- Explain every paper concept in ordinary language before using technical terms.
- Ask the student only for the small visual and scientific decisions explicitly
  listed in this assignment.
- Preserve Assignments 1, 2, and 3 as historical evidence.
- Read `AGENTS.md`, the root `README.md`, `LOG.md`, the three earlier assignment
  READMEs, Assignment 3's final report, and Assignment 3's mentor analytical
  study before implementing anything.
- Never modify or delete the original nuScenes files.
- Never train or fine-tune YOLO.
- Keep primary online inference causal: frame `t` may use frame `t` and earlier
  frames, never future frames.
- Keep privileged nuScenes annotations out of the online tracking input. They
  may be used only to build and evaluate the experiment.
- Stop and report the exact problem if required data or metadata is missing.
  Never invent results.

Create all new implementation files under:

`experiment/`

Create all new generated outputs under:

`results/`

Do not copy Assignment 3 code blindly. First audit whether each reused component
is correct, tested, and suitable for this assignment.

---

## Task 1 — Make a Beginner's Map of the ByteTrack Paper

Read at least the paper's abstract, introduction, BYTE association method,
ablation section, conclusion, and the official repository README.

Create:

`results/paper_map.md`

Explain these terms in one or two simple sentences each:

- Object detection
- Detection confidence
- High-confidence detection
- Low-confidence detection
- Detection floor
- Track and track ID
- Tracklet
- Tracked, lost, and removed track states
- Kalman prediction
- IoU
- Hungarian assignment
- First association
- Remaining unmatched track
- Second association
- Track initialization
- Track buffer
- False positive
- False negative
- Identity switch
- Trajectory fragmentation

Use this everyday analogy:

> A teacher first matches students to clearly readable name tags. If someone is
> still unmatched, the teacher takes a second look at blurry name tags and uses
> where each student was standing to decide whether a blurry tag belongs to an
> existing student. A blurry tag may help identify someone already known, but it
> should not automatically create a brand-new student record.

Include this simple pipeline:

```text
camera frame
    -> YOLO detections with confidence scores
    -> split detections into high and low groups
    -> predict existing track locations
    -> first match: tracks with high-score detections
    -> second match: remaining tracks with low-score detections
    -> update matched tracks
    -> mark unmatched tracks lost or remove them after the buffer
    -> start new tracks only from unmatched high-score detections
    -> output current tracked objects with IDs
```

For every pipeline step, point to the section or algorithm step where the same
idea appears in the ByteTrack paper. Paraphrase; do not copy long passages.

Add a short table comparing SORT and ByteTrack:

| Question | SORT-style baseline | ByteTrack |
|---|---|---|
| Which detections are considered? | High-confidence only | High first, then low |
| Number of association rounds | One | Two |
| Can a low-score box update an existing track? | No | Yes, in the second round |
| Can an unmatched low-score box start a new track? | No | No |
| Can either method see a fully hidden object? | No | No |

**Task 1 is complete when:** a beginner can explain why ByteTrack uses two
matching rounds and why it does not trust every weak box equally.

---

## Task 2 — Audit the Previous Assignment Before Reusing Anything

Read:

- `../03_sort_paper_experiment/README.md`
- `../03_sort_paper_experiment/results/final_report.md`
- `../03_sort_paper_experiment/results/mentor_analytical_study.md`
- The Assignment 3 configuration, tracker, evaluation, tests, and detection
  tables.

Create:

`results/reuse_audit.md`

For every component considered for reuse, record:

- Source path.
- What the component does.
- Whether it is reused, repaired, or replaced.
- Tests that protect it.
- Scientific risks that remain.

The assistant must explicitly repair these Assignment 3 weaknesses rather than
carry them forward:

1. Tracker-smoothed boxes must not be called raw withheld YOLO boxes.
2. Long gaps must use the real tracker lifecycle and enforce the configured
   track buffer or expiry rule.
3. Identity continuity must be measured from actual output IDs, not assigned by
   construction.
4. Prediction coverage must reflect real track output and expiry.
5. Each method must be timed separately.
6. Repeated frame rows must not be presented as independent object samples.
7. Real timestamp differences should be used in motion prediction when
   practical.
8. The final evidence must state the number of scenes, clips, tracks, events,
   unique frames, and repeated measurement rows separately.

Do not edit Assignment 3 to hide its limitations. Correct the design in the new
Assignment 4 implementation.

**Task 2 is complete when:** the new experiment has a written scientific reason
for every reused or replaced part.

---

## Task 3 — Build a Small, Scene-Disjoint Evaluation Set

Discover the local nuScenes mini dataset rather than hardcoding an absolute
path. It is expected to contain folders similar to:

```text
data/nuscenes/
  samples/
  sweeps/
  v1.0-mini/
```

Use nuScenes `sample_data.json` `prev` and `next` links to build chronological
`CAM_FRONT` clips.

Requirements:

- Reuse the three verified Assignment 3 clips if their manifests and hashes
  still match the source images.
- Add one new clip from a different scene if needed to support a scene-disjoint
  development/evaluation split.
- Maximum 4 clips.
- Maximum 36 frames per clip.
- Maximum 144 frames total.
- Prefer clips containing partial occlusion, crowding, motion blur, or confidence
  changes rather than only easy isolated objects.
- Split by scene before choosing thresholds or inspecting final metrics.
- Use development scenes for threshold choices and keep evaluation scenes
  untouched until settings are frozen.

Create:

- `results/clip_manifest.csv`
- `results/data_check.md`
- `results/split_manifest.csv`
- Local-only `results/clips/<clip_name>/` images

The manifest must include scene token, sample-data token, timestamp, time since
the previous frame, original path, experiment path, keyframe status, source
assignment if reused, image SHA-256, and split.

Validate chronological order, reciprocal metadata links where present, image
existence, image dimensions, hashes, and scene separation.

### Paper connection

ByteTrack is online. It must process frames in time order without seeing future
frames.

**Task 3 is complete when:** all clips are verified and no scene appears in
both development and evaluation splits.

---

## Task 4 — Create Independent Keyframe Ground Truth

Use nuScenes annotations, calibration, and ego poses only as **privileged
offline evaluation evidence**.

At annotated keyframes:

1. Project official 3D boxes into `CAM_FRONT`.
2. Preserve the nuScenes `instance_token` as the physical-object identity.
3. Preserve category, visibility token, depth, truncation, and point-support
   information.
4. Clip boxes correctly to image boundaries.
5. Reject boxes behind the camera or with invalid projected area.
6. Save both the unclipped and clipped status so difficult cases are auditable.

Create:

- `results/projected_ground_truth.csv`
- `results/projection_audit.md`
- Local-only `results/projection_overlays/`

Required tests include:

- Known coordinate transform.
- Behind-camera rejection.
- Image-boundary clipping.
- Positive finite box area.
- Deterministic ordering.
- Preservation of scene, frame, and instance identity.

Visually inspect at least 20 projected keyframe overlays. The student only
needs to inspect a contact sheet of the clearest and most difficult examples;
she must not correct coordinates manually.

The report must state clearly:

> nuScenes labels are used to evaluate the camera tracker. They are not inputs
> to ByteTrack or SORT during online inference.

### Paper connection

The ByteTrack paper uses benchmark ground truth to evaluate trajectory quality.
This assignment needs its own independent reference so it does not evaluate a
tracker against boxes produced by that same tracker.

**Task 4 is complete when:** projected ground truth passes automated checks and
visual review, or every rejected case has a written reason.

---

## Task 5 — Reuse or Run YOLO Once and Preserve Weak Detections

Use the same lightweight pretrained YOLO model and settings as Assignment 3
when practical:

- Prediction only.
- No training or fine-tuning.
- CPU-compatible.
- Image size 640 unless the earlier documented fallback is required.
- Detection floor no higher than 0.05 so weak boxes remain available.
- Process frames in chronological order.
- Do not load all images into memory at once.

Reuse Assignment 3 cached detections only when the image hash, model name,
weights hash, image size, confidence floor, and package versions match exactly.
Run YOLO only for new or changed frames.

Create:

- `results/detections.csv`
- `results/detector_audit.md`
- Local-only detector cache
- `results/detection_confidence_by_visibility.png`

Every detection row must contain the raw YOLO box, class, confidence, frame
identity, inference time, and cache key. Never replace the raw box with a
tracker-corrected box.

At projected keyframes, match raw detections to ground truth for evaluation and
save the match status. Do not use those ground-truth matches inside the tracker.

Before selecting thresholds, show the distribution of detection confidence by
nuScenes visibility category on development scenes. Freeze these configuration
values before opening evaluation results:

- `detection_floor`
- `high_score_threshold`
- `new_track_threshold`
- First-association IoU threshold
- Second-association IoU threshold
- Track buffer duration

Thresholds must live in a configuration file and never be tuned on evaluation
scenes.

### Paper connection

The paper's central claim depends on retaining low-score detection boxes rather
than discarding them before association.

**Task 5 is complete when:** every frame has a traceable detector result or a
reported failure, and weak detections remain available to both methods.

---

## Task 6 — Implement an Educational BYTE Association Layer

Do not copy the authors' tracker source file. Implement a readable educational
version from the paper's algorithm and cite the paper at the top of every main
source file.

Suggested structure:

```text
experiment/
  config.yaml
  run_experiment.py
  reproduce_all.py
  src/
    clip_builder.py
    projection.py
    detector_cache.py
    geometry.py
    kalman_box_tracker.py
    assignment.py
    track.py
    sort_tracker.py
    bytetrack_tracker.py
    event_selection.py
    controlled_trials.py
    evaluation.py
    visualization.py
    report.py
  tests/
```

The assistant may improve the structure, but must keep the paper's stages easy
to locate.

For each frame, the educational ByteTrack implementation must:

1. Separate detections into high-score and low-score groups using frozen
   configuration thresholds.
2. Predict the current location of existing tracked and eligible lost tracks.
3. Perform the first one-to-one association with high-score detections.
4. Keep unmatched tracks and unmatched high-score detections.
5. Perform the second association between the appropriate remaining tracks and
   low-score detections using IoU, following the paper's method.
6. Update tracks matched in either association stage.
7. Mark unmatched tracks lost and remove them only after the configured buffer.
8. Initialize new tracks only from eligible unmatched high-score detections.
9. Discard unmatched low-score detections instead of starting tracks from them.
10. Output only valid current tracked objects, with explicit state and evidence
    source.

Every output row must say whether its current evidence came from:

- `high_score_detection`
- `low_score_detection`
- `motion_prediction`
- or no output because the track was lost/removed

Use real timestamp differences in the Kalman transition when available. If the
implementation uses fixed steps, justify and test that decision.

### Paper connection

The two association rounds are the BYTE contribution. The motion prediction,
IoU, Hungarian matching, and track lifecycle build on tracking-by-detection
ideas already learned from SORT.

**Task 6 is complete when:** a reader can point from every step in the paper map
to the exact module and test that implements it.

---

## Task 7 — Add Automatic Tests Before Running the Study

At minimum, test that:

1. Identical boxes have IoU 1 and disjoint boxes have IoU 0.
2. Detections are split correctly at exact score boundaries.
3. The same detection cannot match two tracks.
4. The same track cannot match two detections in one association round.
5. High-score matching happens before low-score matching.
6. Only tracks left unmatched after the first round enter the second round.
7. A plausible low-score detection can update an existing unmatched track.
8. An unmatched low-score detection cannot create a new track.
9. An unmatched high-score detection can create a new track when eligible.
10. A lost track survives exactly the configured buffer and then expires.
11. A returning detection may reconnect only through the real association
    process; its ID is never assigned manually.
12. SORT and ByteTrack receive byte-for-byte identical raw detections.
13. Ground-truth tables are inaccessible from the online tracker interface.
14. Frames from different scenes never share a track.
15. Timestamp-aware prediction behaves correctly for unequal frame intervals.
16. The same input, configuration, and seed produce the same outputs.
17. Every output state has exactly one evidence-source label.

Add a small integration fixture with two objects, one weak detection, one false
weak box, a short complete miss, and a returning object. Verify every expected
ID and state by hand in the test definition.

Do not continue to the main experiment while tests fail.

**Task 7 is complete when:** all tests pass and their count is recorded in run
metadata.

---

## Task 8 — Define Fair Comparison Methods

Run these methods on the same ordered frames and the same raw detections:

### Method A — YOLO high-confidence only

Report accepted high-score detections frame by frame without temporal memory.
This is a detection baseline, not a tracker.

### Method B — High-confidence SORT

Use one association round with only high-score detections. Enforce its real
track lifecycle and configured expiry.

### Method C — ByteTrack

Use the same motion model and first association as Method B, plus the paper's
second association with low-score detections.

Keep all shared settings identical:

- Images and frame order.
- Raw detector boxes.
- High-score threshold.
- Motion model.
- First-association similarity and threshold.
- Track initialization rule.
- Track buffer unless a method-specific difference is explicitly required and
  reported.

The main comparison must change only the use of the low-confidence second
association.

**Task 8 is complete when:** an automated check proves that the methods receive
identical inputs and differ only in documented tracking behavior.

---

## Task 9 — Select Natural Weak-Evidence Events Automatically

On development and evaluation scenes separately, find candidate events where:

- The same ground-truth `instance_token` is present before and after the event.
- Raw YOLO confidence falls from the high group into the low group for at least
  one frame, or a high-confidence match temporarily disappears.
- The object remains within the camera view rather than simply exiting.
- The event has enough frames before and after to measure identity continuity.
- Visibility, overlap with a possible occluder, or motion blur provides a
  plausible reason for confidence loss.

Rank candidates deterministically. Keep no more than 12 events total and try to
include more than one scene and object class. If the mini split cannot support
that diversity, report the limitation rather than relax identity validity.

Create:

- `results/natural_event_manifest.csv`
- `results/natural_event_selection.md`
- Local-only `results/natural_event_contact_sheets/`

Each accepted or rejected event must record its reason, source tokens, split,
class, frame range, confidence sequence, visibility sequence, and possible
occluder evidence.

The assistant must visually verify the shortlisted events. The student should
inspect only a small contact sheet and answer:

> Do these examples look like weak or partially blocked objects rather than
> objects that simply left the image?

If the student is unavailable, record the visual review as assistant-only and
do not invent student confirmation.

### Paper connection

The paper motivates ByteTrack with low-score detections caused by occlusion.
This task checks whether the same type of evidence appears naturally in the
local data.

**Task 9 is complete when:** every natural event is traceable and no event is
accepted solely because the tracker performed well on it.

---

## Task 10 — Build Two Controlled Experiments

Natural events may be too few for a clear small experiment. Add two controlled
tests that answer different questions.

### Experiment A — Confidence demotion

Choose valid high-confidence target detections using development rules fixed
before evaluation. For a short middle window:

1. Keep the original raw box coordinates unchanged.
2. Lower only the target detection score into the configured low-confidence
   band.
3. Do not modify any other object's detection.
4. Give the identical modified detection table to high-confidence SORT and
   ByteTrack.
5. Evaluate association against the original raw YOLO box, clearly labeled
   **pseudo-ground truth**.

Test short windows such as 1, 2, and 3 frames when valid. The same target/frame
must not be counted as a new independent object merely because it appears in
several windows.

This isolates the paper's main idea: the box still exists, but a high-threshold
tracker throws it away.

### Experiment B — Complete detection absence

For a separate set of valid windows:

1. Remove the target detection entirely.
2. Run the real full SORT and ByteTrack trackers with lifecycle enabled.
3. Keep other detections present so false matching and ID switches remain
   possible.
4. Process the returning detection normally.

This test establishes ByteTrack's boundary. A second association cannot use a
box that does not exist.

Create:

- `results/controlled_event_manifest.csv`
- `results/controlled_trials.csv`
- `results/controlled_protocol.md`

Keep confidence-demotion and complete-absence results separate in every table,
chart, and conclusion.

**Task 10 is complete when:** each controlled event is recreated exactly from a
seeded manifest and both methods receive identical modified inputs.

---

## Task 11 — Measure Benefits and Harms Together

Use projected ground truth at annotated keyframes for the main natural-event
evaluation. Use original raw detections only as marked pseudo-ground truth for
the controlled confidence-demotion experiment.

For each method, report where applicable:

- Ground-truth object recall.
- Precision or false-positive count.
- IDF1 or another clearly implemented identity-continuity metric.
- Number of ID switches.
- Number of trajectory fragments.
- Fraction of low-score ground-truth objects recovered.
- Correct low-score association rate.
- Incorrect low-score association rate.
- False tracks created from weak evidence.
- Ghost-track count and duration.
- Track survival through complete absence.
- Correct ID recovery after reappearance.
- Center error and IoU at scored frames.
- Tracker-only runtime measured separately for each method.

Report counts at these distinct levels:

- Scenes.
- Clips.
- Unique physical objects or ground-truth instance tokens.
- Natural events.
- Controlled events.
- Unique frames.
- Repeated measurement rows.

Use the track or event—not each repeated frame row—as the main experimental
unit in comparisons and uncertainty displays. Never imply that nested windows
are independent samples.

Save:

- `results/natural_trials.csv`
- `results/summary_by_event.csv`
- `results/summary_by_track.csv`
- `results/summary_by_method.csv`
- `results/run_metadata.json`

Natural and controlled results must never be silently pooled.

### Questions the analysis must answer

1. Did ByteTrack recover real objects from low-confidence detections?
2. Did those recoveries reduce fragmentation or preserve identity?
3. How many wrong associations or false tracks came from weak detections?
4. Did ByteTrack still help when the target detection was completely absent?
5. Which visibility levels, classes, and gap durations benefited most?
6. Were gains consistent across tracks and scenes, or driven by a few cases?
7. How sensitive were conclusions to the high/low score boundary?
8. What did ByteTrack add beyond the SORT implementation from Assignment 3?

**Task 11 is complete when:** every reported number traces to a table row,
configuration hash, source frame, and evaluation-reference type.

---

## Task 12 — Run One Required Ablation and One Sensitivity Check

### Required ablation — remove the second association

Compare:

- Full ByteTrack.
- The identical tracker with the low-score second association disabled.

Keep everything else fixed. This should reduce to the high-confidence
SORT-style association behavior and isolates whether ByteTrack's main paper
contribution caused the change.

### Required sensitivity check — change the high-score threshold

Use development scenes to choose one reasonable lower and one reasonable higher
threshold around the frozen main value. Run the cheap cached tracking stage
only. Do not select the best threshold using evaluation results.

Report whether ByteTrack is more robust than high-confidence SORT to this
threshold change, as investigated in the paper's ablation study.

Create:

- `results/ablation.csv`
- `results/ablation.png`
- `results/threshold_sensitivity.csv`
- `results/threshold_sensitivity.png`

Explain what changed, what stayed fixed, and whether the result supports the
paper's idea. Do not use the phrase “statistically significant” unless an
appropriate test and experimental unit were defined in advance.

**Task 12 is complete when:** the main effect is separated from threshold
choice and tracker bookkeeping.

---

## Task 13 — Generate Clear Charts and Videos

Create at least these charts:

1. `detection_confidence_by_visibility.png`
2. `low_score_recovery_by_method.png`
3. `fragmentation_by_method.png`
4. `id_switches_by_method.png`
5. `false_associations_by_method.png`
6. `complete_absence_survival.png`
7. `runtime_comparison.png`
8. `ablation.png`
9. `threshold_sensitivity.png`

Every chart must:

- Name the evaluation-reference type.
- Show sample counts using tracks/events as the main unit.
- Label units and whether higher or lower is better.
- Keep natural, confidence-demotion, and complete-absence results separate.
- Avoid truncated or misleading axes.
- Display individual track/event points when the sample is small.
- State when a metric is undefined rather than replacing it with zero.

Create at least three short side-by-side videos:

```text
raw YOLO | high-confidence SORT | ByteTrack | evaluation reference
```

Use consistent colors:

- High-confidence detection: green
- Low-confidence detection: yellow
- SORT track: blue
- ByteTrack track updated by high score: cyan
- ByteTrack track updated by low score: orange
- Motion-only prediction: dashed purple
- Offline ground-truth reference: magenta

Every box must state whether it is current visual evidence, a motion prediction,
or offline evaluation reference.

At least one video must show a successful low-score recovery, one must show a
false or ambiguous weak detection, and one must show complete detection absence.

Create one slow explanatory video that pauses when:

- Confidence crosses below the high threshold.
- SORT discards the weak detection.
- ByteTrack enters its second association.
- The weak box either updates the correct track or is rejected.
- A later high-confidence detection returns.

### Paper connection

The videos should make the paper's two-round association understandable without
requiring the student to read the algorithm notation.

**Task 13 is complete when:** the student can point to the first match, second
match, prediction-only state, and offline reference.

---

## Task 14 — Write the Automatic Final Report

Generate:

`results/final_report.md`

Use this structure:

1. Paper citation and links
2. What the ByteTrack paper proposed
3. Why ByteTrack followed SORT in this assignment sequence
4. How this experiment differs from a full paper reproduction
5. Hypothesis recorded before the run
6. Dataset, clips, scene-disjoint split, and data checks
7. Independent projected ground truth
8. Detector settings, confidence distribution, and cache
9. Educational ByteTrack implementation
10. Fair comparison methods
11. Natural weak-evidence protocol
12. Controlled confidence-demotion protocol
13. Complete detection-absence protocol
14. Quantitative results with scene/track/event counts
15. Required ablation
16. Threshold sensitivity
17. Best successful recovery
18. Most important false association or failure
19. Runtime and laptop suitability
20. Connection to the paper's claims
21. Connection to Assignments 1–3 and OATM
22. Limitations
23. Conclusion
24. Exact reproduction command

The report must answer directly:

1. Did ByteTrack beat high-confidence SORT?
2. Was any gain caused by the low-score second association?
3. Did ByteTrack preserve IDs or reduce fragments?
4. What safety cost appeared in false associations or ghost tracks?
5. What happened when there was no detection box at all?
6. Were results consistent across scenes and object classes?
7. How did threshold choice affect each method?
8. How much time belonged to YOLO and to each tracker separately?
9. Which result supports the ByteTrack paper and which result limits it?
10. What should the larger OATM method add beyond ByteTrack?

The limitations section must say:

- This is not a reproduction of the paper's MOT benchmark results.
- nuScenes mini and the selected clips are small.
- Official annotations are sparse keyframes rather than every camera frame.
- Projected 3D boxes can be truncated or imperfect as 2D references.
- Controlled confidence demotion is not identical to natural occlusion.
- Raw YOLO boxes used as pseudo-ground truth may be inaccurate.
- ByteTrack uses current weak detections; it does not recover truly invisible
  objects from visual evidence.
- Camera ego-motion and nonlinear object motion can still challenge the shared
  Kalman model.
- Thresholds selected on a tiny development split may not generalize.
- A gain in recall is incomplete without false-association and ghost analysis.

Include this citation:

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

**Task 14 is complete when:** the report can be understood without opening the
code and every finding is traceable to evidence.

---

## Task 15 — Let the Student Choose One Experiment Without Coding

After showing the main results, ask the student to choose **one** question:

1. “Let ByteTrack accept weaker boxes.”
2. “Make the second match require more overlap.”
3. “Let lost tracks wait longer before removal.”

Before running it, ask:

> What do you think will improve, and what might become worse?

Translate the student's choice into exactly one configuration change. Rerun
only the cheap cached tracking and evaluation stages.

Create:

`results/student_experiment.md`

Record:

- The student's exact choice.
- The student's prediction in her own words.
- The single configuration change.
- What remained fixed.
- Results before and after.
- Whether the prediction was supported.
- How the result relates to the ByteTrack paper.

If the student does not answer, pause this task or record it as incomplete. Do
not choose for her and do not invent a first-person prediction.

**Task 15 is complete when:** the student has made and interpreted one real
scientific choice without editing code.

---

## Task 16 — Short Student Reflection

Create:

`results/student_reflection.md`

Ask these questions one at a time:

1. Why can a low-confidence detection still be useful?
2. What happens in ByteTrack's first association?
3. What happens in its second association?
4. Why should a weak unmatched box not start a new track?
5. Describe one case where ByteTrack helped or made a mistake.
6. Why can ByteTrack not fully solve complete occlusion?
7. What is one reason this small experiment cannot prove ByteTrack always
   works?

Her answers may be one or two sentences each. Fix spelling lightly, but never
replace her meaning or invent an answer. If an answer is confused, explain the
concept, ask her to try once more, and preserve the final response honestly.

This is the only required manual results writing.

**Task 16 is complete when:** the student can distinguish weak visual evidence,
motion prediction, and complete absence in her own words.

---

## Task 17 — Validate, Document, Commit, and Push

Before committing, the assistant must:

1. Run all unit and integration tests.
2. Reproduce every small artifact from one documented command using cached
   detections.
3. Confirm every required CSV, chart, video, and Markdown report exists.
4. Validate required CSV fields, finite coordinates, valid scores, timestamps,
   scene boundaries, and unique IDs.
5. Recalculate summary counts independently from detailed tables.
6. Confirm thresholds were frozen before evaluation results were opened.
7. Confirm natural, demoted-confidence, and complete-absence results remain
   separate.
8. Confirm identity continuity comes from actual tracker outputs.
9. Confirm the real lifecycle is used and expiry is enforced.
10. Confirm runtime was measured separately for each method.
11. Confirm charts identify the experimental unit and reference type.
12. Confirm predictions are never labeled as detections.
13. Confirm the paper is linked and cited.
14. Confirm the required limitations are present.
15. Inspect Git status and staged files for private or large artifacts.

Commit:

- Source code and tests.
- Small configuration files.
- Paper map and reuse audit.
- Compact manifests and summary tables.
- Charts and Markdown reports.
- Student experiment and reflection.
- Only a few compressed videos if each is reasonably sized for GitHub.

Do not commit:

- The nuScenes dataset.
- Copied clip images.
- Model weights.
- Virtual environments.
- Large detector caches.
- Private workbooks.
- Temporary or lock files.
- Credentials, tokens, or authentication data.

Show the student the exact planned file list before committing. Use a clear
message such as:

`add ByteTrack low-confidence association experiment`

Push the completed branch. If authentication is required, use the official
private browser flow and never ask for a password or authentication code.

**Task 17 is complete when:** another student can inspect the paper connection,
reproduce the small experiment, and understand its evidence and limitations.

---

# Required Final Deliverables

The assistant must not call the implemented assignment complete unless these
exist:

```text
README.md

experiment/
  config.yaml
  run_experiment.py
  reproduce_all.py
  src/
  tests/

results/
  paper_map.md
  reuse_audit.md
  data_check.md
  clip_manifest.csv
  split_manifest.csv
  projection_audit.md
  projected_ground_truth.csv
  detector_audit.md
  detections.csv
  natural_event_manifest.csv
  natural_event_selection.md
  controlled_event_manifest.csv
  controlled_protocol.md
  natural_trials.csv
  controlled_trials.csv
  summary_by_event.csv
  summary_by_track.csv
  summary_by_method.csv
  run_metadata.json
  ablation.csv
  threshold_sensitivity.csv
  charts/
  videos/
  final_report.md
  student_experiment.md
  student_reflection.md
```

Large local-only files may be excluded from Git, but their paths, sizes, hashes,
and reproduction instructions must be recorded.

# Laptop Safety Limits

Obey these limits unless the student and mentor explicitly approve more:

- Maximum 4 clips.
- Maximum 36 frames per clip.
- Maximum 144 frames total.
- Maximum 12 accepted natural events.
- Maximum 12 controlled target tracks.
- YOLO nano model only.
- No detector training or fine-tuning.
- CPU-compatible execution.
- One YOLO pass per unique configuration.
- Cached detections for all tracker experiments.
- No raw-data modification.
- No large files committed to GitHub.

If runtime is too long, reduce the number of clips or events before weakening
the scientific controls. Record every reduction.

# Rules the LLM Must Follow

1. The student performs no coding or terminal work.
2. Preserve all earlier assignments and source data.
3. Read the ByteTrack paper and official repository before implementing.
4. Implement the paper's idea educationally; do not copy the official tracker
   file.
5. Use identical raw detections for every compared method.
6. Keep raw detections, tracker outputs, and offline ground truth separate.
7. Never leak nuScenes identity, visibility, depth, or pose into online tracking.
8. Never call low-confidence evidence a guaranteed true object.
9. Never call a motion prediction a current detection.
10. Never call projected ground truth a camera-only tracker input.
11. Enforce real track birth, lost, reactivation, and removal behavior.
12. Never assign matching track IDs manually for evaluation.
13. Tune thresholds only on development scenes.
14. Report false associations and ghosts beside recovered-object recall.
15. Keep natural and controlled studies separate.
16. Use tracks/events as the main experimental unit.
17. Show sample counts and conditional results.
18. Do not claim full ByteTrack benchmark reproduction.
19. Ask the student for her own experiment choice and reflection; do not invent
   them.
20. Validate outputs before committing and keep private data out of Git.

# What Success Looks Like

Success does **not** mean ByteTrack must win every metric.

Success means:

- The student understands why the paper keeps weak detections available.
- Two-stage association is implemented and tested clearly.
- SORT and ByteTrack receive identical detector inputs.
- Independent nuScenes evidence is used for the main keyframe evaluation.
- Controlled score demotion isolates ByteTrack's central idea.
- Complete detection absence is tested as a separate limitation.
- Real tracker lifecycle and identity behavior are measured.
- Benefits and harms are reported together.
- Conclusions use the correct experimental unit and acknowledge small samples.
- The paper's contribution, evidence, and limits are visible without reading
  matrix equations or writing code.

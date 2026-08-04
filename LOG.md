# OATM Project Log

## Current status

- **Phase:** Environment and data-pipeline validation.
- **Active branch:** `master`.
- **Methodology:** Defined in `OATM/METHODOLOGY.md`.
- **Implementation plan:** Defined in `OATM/IMPLEMENTATION_PLAN.md`.
- **Dataset:** nuScenes mini is available locally at `data/nuscenes/`.
- **Immediate milestone:** Prove that a clean OATM pipeline can load nuScenes
  mini, reconstruct chronological `CAM_FRONT` sequences, project annotations,
  and produce a verified candidate-occlusion index.

## 2026-08-04 — OATM Task 6: rebuild and verify baselines

### Change

Wrote `results/reuse_audit.md` auditing every component before reuse from
Assignments 2-4. Reused Assignment 4's geometry/Hungarian-association/
timestamp-aware-Kalman code unchanged into `src/oatm/tracking/`; rebuilt
static memory (Assignment 2 had no reusable tracker-shaped code) and adapted
SORT/ByteTrack to emit OATM's canonical `TrackerOutputRecord` instead of
Assignment 4's lighter-weight output type. All four baselines (YOLO-only,
static memory, SORT, ByteTrack) now share one interface and ran over all 10
scenes from one `configs/tracker.yaml`.

### Reason

Later phases (motion memory, the occlusion state machine, the MVP report)
need a settled, identically-fed baseline layer to compare against -- and
Assignment 2's static memory in particular never had a real lifecycle, so it
could not previously be fairly compared to SORT/ByteTrack's buffer-based one.

### Validation

- 27 new tests, covering: exact
  IoU=1/IoU=0 cases, one-to-one assignment, dt-scaled Kalman prediction,
  birth/missing/reactivation/expiry for all three stateful baselines,
  strong-before-weak ByteTrack ordering, unmatched-weak-cannot-birth,
  scene-boundary track ID isolation, structural inaccessibility of ground
  truth (checked both by source-string search and by `update()`'s exact
  parameter set), determinism, and growing localization uncertainty during
  missing frames (a real Kalman covariance-trace value, not a placeholder).
  91 tests total, all passing. `ruff check` clean.
- Ran all four baselines over all 10 scenes (34,042 output rows, 4.2s).
  Caught and fixed a real bug in my own summary script: naively grouping
  YOLO-only's per-frame `track_id` by `(scene, track_id)` silently manufactured
  fake 97-row-long "tracks" out of unrelated single-frame detections that
  only shared a reused index number -- exactly the kind of misleading metric
  this project exists to prevent. Fixed by reporting those two columns as
  explicitly not applicable for a method with no real identity, rather than
  computing a number that would be misread as persistence.
- Sanity signal even at this pre-OATM stage: ByteTrack produced fewer, longer
  tracks (425 tracks, mean length 23.84) than SORT (470 tracks, 18.44) --
  consistent with reduced fragmentation, though this is baseline behavior
  only, not yet an evaluated result (that's Task 11).

### Decision and next step

Next: Task 7 (build the detector-intervention and controlled-visual-occlusion
event families -- two explicitly separate experiment types, extending
Assignment 4's controlled-experiment pattern rather than calling detection
edits "visual occlusion").

## 2026-08-04 — OATM Task 12: appearance-memory ablation

### Change

Mentor approved Task 12 only (appearance memory), not Task 13, after Task
11's mandatory checkpoint -- based on the identity-switch rates already
visible in that report. Added `oatm.memory.appearance` (`AppearanceAnchor`,
pure numpy, testable without torch/images), `oatm.memory.embedder` (the
actual frozen network -- pretrained MobileNetV3-Small, eval mode, no
gradients, never fine-tuned), `oatm.tracking.reconnection` (a third,
appearance-gated association stage with `appearance_only` and `dual`
modes), and `oatm.tracking.oatm_appearance_adapter.OATMAppearanceTracker`
(Method F plus that optional stage). Ran `run_appearance_ablation.py` +
`build_appearance_ablation_report.py` comparing `motion_only` (Task 11's
OATM MVP, unchanged), `appearance_only`, and `dual` on the SAME 48
controlled events Task 11 used.

### Reason

Task 11's report showed ByteTrack/OATM still mis-assigning a new track ID
4-17% of the time even in the controlled families -- motion-only
association's clearest remaining weak point, and the specific gap Task 12's
question ("does appearance reconnect the correct identity better than
motion alone?") targets directly.

### Validation

- 25 new tests (219 total, all passing; `ruff check` clean), including the
  required hard-negative case: two same-class hidden tracks, one spatially
  near a reappearing detection and one far away -- appearance correctly
  picks the real match by embedding similarity, not the nearer decoy.
- Ran the full ablation on all 48 controlled events x 3 modes (144 rows) in
  79s, embedding only detections within a bounded window around each event
  (documented, does not affect any measured outcome) -- 1,057 unique crops
  embedded via the frozen network.
- **Null/harmful result, preserved and reported honestly, not adjusted:**
  same-ID recovery was 38/48 for `motion_only`, but only 32/48
  (`appearance_only`) and 34/48 (`dual`) -- appearance-based reconnection
  did not improve identity preservation on this sample, and localization
  error among reconnected tracks got markedly worse (`detector_intervention`
  center error: 3.0px motion_only vs. 14.4px appearance_only). The frozen,
  generic ImageNet MobileNetV3-Small embedding evidently does not
  discriminate these specific same-class real objects (cars/pedestrians at
  typical driving-scene crop resolution) well enough to help, and
  occasionally causes a wrong reconnection instead. `hidden_frame_coverage`
  and `fully_bridged` were unchanged across all three modes -- reconnection
  only fires during the post-window recovery search, never during the
  hidden window itself, so this null result is specifically about identity
  correctness at reappearance, not occlusion-bridging.

### Decision and next step

Given the null/harmful result, `OATMTracker` (motion-only, Task 11's
version) remains the frozen method going forward -- `OATMAppearanceTracker`
stays available but is not adopted as the primary method. Task 13
(ego-motion) was not approved and is not started. Next: Task 14 (freeze the
method, scale beyond mini) -- unless the mentor wants to revisit this
ablation's scope (e.g. a different embedding model) first.

## 2026-08-04 — OATM Task 11: first complete OATM MVP study

### Change

Added `oatm.evaluation` (`ground_truth`, `linking`, `event_metrics`,
`global_metrics`) and two scripts: `run_mvp_study.py` (the expensive part --
one full continuous 5-method run over all 10 scenes, plus a fresh event-
scoped rerun for each of the 48 controlled events) and `build_mvp_report.py`
(fast, reads only the immutable outputs already written, regenerates
`results/mvp_report.md` + 4 charts without rerunning any tracker). Also added
`class_name` to `TrackerOutputRecord` (needed for class-aware ground-truth
matching and precision/recall) and an `oatm_mvp` section to
`configs/tracker.yaml`, duplicating Task 10's frozen confidence/uncertainty
values so one config fully describes the run.

### Reason

Comparing YOLO-only/static/SORT/ByteTrack/OATM-MVP needed one common way to
resolve "the same real object" across methods that each assign their own
track_id, one definition of hidden-window bridging usable across all three
event families without conflating them, and metrics separated by family per
the assignment's own requirement.

### Validation

- 19 new tests (194 total, all passing; `ruff check` clean).
- Counts: 10 scenes, 2342 unique frames per method (404 real keyframes with
  ground truth + 1938 unannotated sweep frames), 6 accepted natural events,
  48 controlled events (24 detector_intervention + 24 controlled_visual, all
  48 reruns completed), 44,966 full-run output rows, 270 event-metric rows.
- Caught and fixed three real bugs before trusting any result, each kept in
  the report/tests rather than quietly patched over:
  1. `yolo_only`'s `track_id` is a meaningless fresh per-frame index (already
     known from Task 6) -- the shared event-metrics function was matching it
     by equality anyway, so coincidentally-equal indices from UNRELATED
     detections in different frames were registering as false "coverage" and
     false "same-ID recovery". Fixed with a separate,
     `compute_yolo_only_event_metrics` that only ever matches by real spatial
     overlap against the true location, never by track_id.
  2. Global precision/recall was scored across all 2342 frames per scene, but
     nuScenes only has 3D annotations at the 404 keyframes -- the other 1938
     "sweep" frames have NO ground truth at all, not merely unlabeled ones.
     Every real detection on a sweep frame was counting as a false positive,
     collapsing precision to ~14% for every method uniformly. Fixed by
     restricting precision/recall to keyframe frames only; precision is now
     ~78-85% across methods, a sane number.
  3. (Carried from the prior entry) the OATM tracker's own immature-
     uncertainty bug, already fixed before this run.
- At matched inputs: OATM MVP and ByteTrack bridge occlusion far more often
  than SORT/static memory/raw YOLO across the 48 controlled events (86-98%
  hidden-frame coverage vs 0-8% for yolo_only). OATM MVP's global ghost rate
  (38.4%) is NOT lower than ByteTrack's (32.7%) in this run despite its
  explicit anti-ghost termination -- reported honestly rather than adjusted;
  Task 10's termination thresholds were frozen from noise-free synthetic
  motion, and real detector noise evidently still produces comparable ghost
  duration. The natural-event family is a very small, honestly-flagged
  sample: only 2 of 6 accepted events had a strong enough real detection at
  the pre-occlusion reference frame for ANY method to even establish a
  tracking anchor -- a genuine detector-confidence limitation affecting all
  five methods identically, not a tracking-quality difference.
- Existence-confidence calibration (507 keyframe `PREDICTED_HIDDEN` rows):
  accuracy among kept predictions rises from 20% (threshold 0.0, everything
  kept) to 55% (threshold 0.95, only the most confident 20% kept) --
  monotonic, a real (if imperfect) calibration signal.

### Decision and next step

**Mandatory mentor checkpoint per the assignment.** Task 11 is complete --
`results/mvp_report.md`, `results/run_metadata.json`, 4 charts, and
`results/mvp_event_metrics.csv` are all written, and `build_mvp_report.py`
regenerates the report from immutable outputs alone. Per
`STUDENT_IMPLEMENTATION_ASSIGNMENT.md`, work must stop here until the mentor
decides whether this evidence justifies Task 12 (appearance memory), Task 13
(ego-motion), both, or neither -- no optional component work begins without
that sign-off.

## 2026-08-04 — OATM MVP tracker: wiring Tasks 6/8/9/10 into one tracker

### Change

Added `oatm.tracking.oatm_adapter` (`OATMTracker`, method name `oatm_mvp`):
one per-frame loop combining the two-stage BYTE-style association (Task 6),
timestamp-aware Kalman motion (Task 8), the five-state camera-only evidence
gate (Task 9), and adaptive existence-confidence decay with priority-ordered
anti-ghost termination (Task 10). This is the prerequisite tracker for Task
11's full MVP comparison study -- no appearance memory, no ego-motion yet.

### Reason

Task 11 needs one integrated method to compare against the four existing
baselines; the individual pieces (association, motion, state machine,
termination) were each already tested in isolation, but never together in
one real update loop until now.

### Validation

- 8 new tests in `tests/unit/test_oatm_adapter.py` (183 total, all passing;
  `ruff check` clean on `src` and `tests`).
- Caught and fixed a real state-transition bug in my own first draft before
  it was ever tested: a track that was `OBSERVED_WEAK` last frame and
  received a strong detection this frame was wrongly staying `OBSERVED_WEAK`
  instead of upgrading to `OBSERVED_STRONG`, because a single dict keyed by
  track index conflated "just birthed" tracks with "pre-existing track
  matched again" tracks under one skip condition. Fixed by transitioning
  matched tracks immediately and inline within the association loops
  themselves, and using a separate list purely for confidence bookkeeping.
- Caught and fixed a second, more consequential bug while writing the
  occlusion-bridging test: a freshly-birthed track's Kalman filter starts
  with a deliberately huge velocity covariance (trace ~30,000, vs. the
  frozen `uncertainty_ceiling=500.0`) that only collapses after a SECOND
  real detection runs the correction step. Without a guard, any track
  occluded on the very frame after its own birth -- a common case, not an
  edge case -- was killed instantly by the uncertainty ceiling regardless of
  `existence_floor` tuning, since that check outranks existence_floor in the
  fixed priority order. This is the same root cause Task 10's comparison
  script had already worked around with a 5-frame warm-up, but that warm-up
  had not carried into the real integrated tracker. Fixed by only applying
  the uncertainty-ceiling/existence-floor termination checks once
  `kalman.hits >= 2`; the existing grace-period logic in `classify_event()`
  still bounds this (a track that stays unmatched past
  `max_grace_frames_without_evidence` is cut loose regardless), so this is
  not an unlimited loophole.

### Decision and next step

Next: Task 11, the first complete OATM MVP study -- comparing YOLO-only/
static/SORT/ByteTrack/OATM-MVP on identical inputs across natural,
controlled-visual, and detector-intervention evidence, with a **mandatory
mentor checkpoint** immediately afterward before any optional component
(appearance memory, ego-motion) begins.

## 2026-08-04 — OATM Task 10: adaptive confidence and anti-ghost termination

### Change

Added `oatm.memory.confidence` (hazard-based existence-confidence decay:
`P_exist *= exp(-(beta + alpha*ΔU) * Δt)`, real elapsed seconds, resets only
on new evidence), `oatm.occlusion.termination` (priority-ordered, exactly-
one-reason anti-ghost termination), and `oatm.occlusion.termination_study` +
`scripts/run_termination_comparison.py` comparing a fixed-lifetime policy
against the adaptive one at MATCHED ghost risk. Froze
`configs/termination.yaml` (beta=0.15, alpha=0.01, existence_floor=0.05,
uncertainty_ceiling=500.0).

### Reason

Confidence must never rise without new evidence, every terminated track
needs exactly one traceable reason (never a vague combination), and
comparing lifetimes only at each policy's own best-recall point would hide
the real tradeoff -- the honest comparison has to hold ghost risk constant.

### Validation

- 25 new tests (168 total): confidence is monotonically non-increasing
  without new evidence, a longer real-time gap costs strictly more
  confidence than a shorter one at the same hazard rate (elapsed seconds,
  not frame count), confidence stays in [0,1] over a long gap, exactly one
  termination reason even when all five conditions are simultaneously true
  (fixed priority order), each reason fires correctly in isolation, a clear
  exit terminates immediately while a plausible early occlusion does not,
  and the committed `termination.yaml` values are pinned by a canary test.
- Found and fixed a real bug in the comparison script before trusting its
  output: the adaptive policy scored 0.00 recall at every single
  `existence_floor` setting from 0.01 to 0.6, because the simulation used a
  freshly-constructed Kalman filter (deliberately huge initial velocity
  covariance) directly in the "missing frames" loop, tripping the
  uncertainty ceiling on frame one regardless of any threshold. Fixed by
  warming up the tracker with 5 real observations first, matching how a
  track would actually enter a gap in practice; recall became a sane,
  monotonically increasing curve after the fix.
- At matched ghost risk (<=5 ghost frames), fixed-lifetime (max_missing=5)
  reaches 0.50 recall vs. adaptive's (existence_floor=0.25) 0.40 -- reported
  honestly rather than adjusted to favor OATM's own method. With
  noise-free constant-velocity synthetic motion, a fixed frame count and an
  uncertainty ceiling draw a very similar line; the report explains the
  adaptive policy's advantage should be expected to show more clearly with
  noisier, more variable real motion.

### Decision and next step

Per this task's ordering, camera-derived ego-motion and appearance memory
still wait. Next: Task 11, the first complete OATM MVP study -- comparing
YOLO-only/static/SORT/ByteTrack/OATM-MVP on identical inputs across natural,
controlled-visual, and detector-intervention evidence, with a **mandatory
mentor checkpoint** immediately afterward before any optional component
(appearance memory, ego-motion) begins.

## 2026-08-04 — OATM Task 9: OATM evidence states / state machine

### Change

Added `oatm.occlusion.state_machine` (the exact five-state machine --
`OBSERVED_STRONG`/`OBSERVED_WEAK`/`PREDICTED_HIDDEN`/`LOST`/`EXITED` -- with
an explicit, exhaustive transition table) and `oatm.occlusion.evidence` (a
small rule-based, camera-only classifier: foreground-overlap, confidence
trend, image-boundary + outward motion, elapsed time/grace -- deliberately
NOT using any privileged nuScenes label).

### Reason

The system must distinguish "I can currently see it," "I can barely see
it," "I believe it's hidden," "I've lost it," and "it left" -- and low
confidence alone must never be treated as proof of occlusion.

### Validation

- 40 new tests (158 total): an EXHAUSTIVE 5-state x 5-event (25-combination)
  transition table -- every allowed transition works, every disallowed one
  (including all events on the two terminal states) raises
  `InvalidTransitionError`, not a silent no-op. Plus the required fixtures:
  true occlusion (visible occluder), field-of-view exit (boundary +
  outward motion -- checked and confirmed it wins even when an occluder is
  also present), ordinary detector miss (no occluder, grace spent ->
  insufficient evidence, not hidden forever), poor visibility (a
  barely-above-floor detection is still just a weak detection, never
  occlusion), false initial track (a track born then immediately missing
  goes to insufficient evidence quickly), and compatible/incompatible
  reappearance (the latter proven via Task 6's own association test: a
  wrong-class candidate is never handed to this classifier as a match).
- Explicitly verified a track can only ever be BORN into `OBSERVED_STRONG`
  (never `OBSERVED_WEAK`), and that `LOST`/`EXITED` are structurally
  terminal in the transition table itself, not just guarded by an
  `if`-check that could be forgotten later.

### Decision and next step

Next: Task 10 (adaptive existence/identity confidence and anti-ghost
termination -- the piece that decides how long a `PREDICTED_HIDDEN` track
gets to keep waiting before it must become `LOST`).

## 2026-08-04 — OATM Task 8: motion memory and growing uncertainty

### Change

Added `oatm.memory.{motion,motion_regimes,motion_comparison}` and
`scripts/run_motion_comparison.py`. Built seven synthetic motion-regime
fixtures with EXACTLY known ground truth (stationary, smooth, slow, unequal
timestamp gaps, turning, abrupt, missing-then-reappear) and compared
`StationaryPredictor` (frozen box) against the existing timestamp-aware
Kalman filter on each. Wrote the committed `results/motion_regime_report.md`
and the required diagram, `results/charts/uncertainty_growth.png`.

### Reason

Before adding any occlusion classifier or appearance memory, the project
needs to know, honestly and per motion regime, whether constant-velocity
prediction actually beats freezing the box -- and whether the system's own
uncertainty estimate behaves sensibly while evidence is missing.

### Validation

- 15 new tests (133 total), all passing: uncertainty grows monotonically
  during every regime's gap (not just some), stationary is exact on a
  non-moving object, Kalman clearly wins on smooth motion, the two models
  are close on slow motion, Kalman wins on irregular timestamps (the exact
  payoff of Assignment 4's dt-aware repair), and the abrupt/turning regimes
  are checked for a REPORTABLE result without asserting a predetermined
  winner.
- Caught and fixed an honesty issue in my own first draft: the report
  initially labeled the turning-motion result "negative/mixed" without
  checking the actual numbers -- Kalman still won there (51.35px vs.
  66.56px), just by a much smaller margin than on smooth motion (23% vs.
  100% error reduction). Rewrote that section to report the real numbers and
  explain the margin, rather than asserting a loss that didn't happen.
- Kalman won on mean center error in all seven regimes tested (including
  turning/abrupt, where its assumption is violated) -- an honest result, not
  a forced negative one; the report explains why that is not the same claim
  as "motion prediction always helps" and flags turning/abrupt as the
  regimes most likely to flip with real noise or sharper turns.
- Required diagram confirms uncertainty visibly grows (P-trace 20.7 -> 48.4
  -> 88.8) after 1, 3, and 5 missing frames.

### Decision and next step

Per this task's rule ("do not add visual ego-motion compensation or
appearance embeddings yet"), stopping the motion work here. Next: Task 9
(the OBSERVED_STRONG / OBSERVED_WEAK / PREDICTED_HIDDEN / LOST / EXITED
state machine).

## 2026-08-04 — OATM Task 7: controlled-occlusion event families

### Change

Implemented `oatm.dataset.controlled_occlusion` and
`scripts/build_controlled_events.py`. Selected 6 real target tracks (via the
ByteTrack baseline's now-correct raw detection boxes) and built two
explicitly separate event families: `detector_intervention` (demote/remove
the target's detection row, pixels untouched) and `controlled_visual` (paint
a seeded gray mask over the target on a local image copy, then rerun the
real frozen detector on that copy). 48 events total (24 each), varying
duration (2, 5 frames) and coverage (0.5, 1.0).

### Reason

Detector intervention and visual occlusion test fundamentally different
things (tracker logic in isolation vs. real re-detection under an actual
covered target) and must never be conflated or compared as if they were the
same experiment.

### Validation

- Found and fixed a determinism bug before it shipped: the original seed
  derivation used Python's built-in `hash()`, which is randomized per
  process (`PYTHONHASHSEED`) -- every rerun would have produced different,
  unreproducible masks. Replaced with a SHA-256-based deterministic digest.
- 11 new tests (same-seed same-mask, full-coverage mask matches the target
  box size, deterministic target selection/windowing, all four baselines
  receiving the identical unmutated detection list) plus 3 integration tests
  against the real manifest (both families present and never merged, every
  event has a seed and traceable target, visual events additionally record
  mask box and cache key). 103 tests total, all passing.
- **Visual checkpoint actually performed**, not just described: zoomed into
  full-resolution modified frames and confirmed the gray mask lands exactly
  on the intended target (a pedestrian at a bus stop -- shoes visible below
  the mask edge; a car at a distant intersection). Sent the full six-target
  before/after comparison to the student directly for her own confirmation.
- Confirmed the source nuScenes files are only ever opened for reading
  (`Image.open()` + `.copy()` before any drawing) -- never written to.

### Decision and next step

Next: Task 8 (motion memory -- compare stationary vs. timestamp-aware
constant-velocity prediction across synthetic motion regimes, with growing
localization uncertainty).

## 2026-08-04 — OATM Task 5: run detector once, cache observations

### Change

Implemented `oatm.detection.cache` (content-hash cache: image + model +
weights + image size + confidence floor + package versions) and
`scripts/run_detector.py`. Ran the same pretrained `yolo26n.pt` used in
Assignments 1-4 (prediction only, no training) over all 2,342 CAM_FRONT
frames across all 10 mini scenes, confidence floor 0.05.

### Reason

Every later baseline (SORT, ByteTrack, OATM) must compare fairly -- that
requires one shared, cached detector observation table, never a per-method
re-run.

### Validation

- First run: 2,342/2,342 cache misses (real inference), 49,436 raw
  detections, 285.4s wall-clock (0.17s/frame benchmark, within budget).
- Second run (identical inputs): 2,342/2,342 cache **hits**, 0 misses,
  identical 49,436-row output, 9.9s wall-clock -- a real, not just unit-tested,
  proof the cache works.
- 9 new tests (cache-key sensitivity to every field, round-trip, a
  call-counting fake-inference test proving a hit skips real work, plus
  integration checks against the real generated artifact: no out-of-range
  confidence, every detection traces to a real frame, weak detections
  actually survive the floor). 64 tests total, all passing. `ruff check`
  clean.
- Confidence-by-class table confirms the target MVP classes are well
  represented: 31,367 car, 5,097 person, 3,193 truck, 1,031 bus detections,
  spanning the full 0.05-0.95+ confidence range.

### Decision and next step

Next: Task 6 (rebuild and verify the YOLO-only / static-memory / SORT /
ByteTrack baselines behind one common interface, auditing Assignments 1-4's
code for reuse rather than copying it blindly).

## 2026-08-04 — OATM Task 4: mine and review natural occlusion events

### Change

Implemented `oatm.dataset.event_mining` (visibility decline-then-recovery
detection requiring a plausible closer occluder as a second independent
signal) and `scripts/mine_natural_events.py` /
`scripts/record_natural_event_review.py`. Assigned a scene-disjoint
development/validation/test split (6/2/2 of the 10 mini scenes, seeded) before
any event was selected. Wrote the committed
`results/natural_event_manifest.csv` and `results/natural_event_selection.md`.

### Reason

A visibility-label change alone is not proof of real occlusion (could be a
labeling artifact, an exit, or truncation) -- Task 4 requires two independent
signals plus an actual human visual review before any event counts as usable
evaluation evidence.

### Validation

- 54 candidate instances found a decline-then-recovery pattern with a
  plausible closer occluder; 19 more had the visibility pattern but no
  occluder and were auto-rejected (logged with reasons, never silently
  dropped, never shown for review).
- Shortlisted the top 15 (by occlusion length, then occluder overlap) and
  built a 15-row visual contact sheet (PRE/START/END/POST per candidate, the
  candidate occluder outlined in cyan).
- **The student reviewed the actual contact sheet** (sent directly, not just
  described) and gave a real per-candidate verdict: 6 accepted, 8 unsure
  (mostly "unclear to my eye," two flagged for multiple overlapping
  occluders), 1 rejected ("wrong objects"). Recorded verbatim, not invented.
- 10 new tests (event-boundary detection, occluder depth/overlap logic,
  deterministic scene-disjoint splitting), all passing (56 total). `ruff
  check` clean.
- Confirmed: every scene maps to exactly one split, every non-accepted event
  records why, and event boundaries are strictly ordered pre < start <= end <
  post (all tested against the real generated manifest).

### Decision and next step

Only the 6 `accepted` events are eligible for the later MVP evaluation
(Task 11) -- `unsure` events stay in the manifest for traceability but are
excluded, not silently promoted. This mini result is explicitly a pilot, not
a statistical conclusion (15 reviewed candidates from one dataset). Next:
Task 5 (run the frozen detector once, cache observations for every baseline).

## 2026-08-04 — OATM Task 3: project 3D ground truth into CAM_FRONT

### Change

Implemented `oatm.dataset.projection` (global -> ego -> camera -> pinhole
transform, behind-camera rejection, image-boundary clipping via shapely,
car/pedestrian evaluation-class mapping) and
`scripts/project_annotations.py` + `scripts/build_projection_overlays.py`.
Wrote local-only `projected_ground_truth.parquet` (5,384 accepted rows) and
`projection_rejections.json`, plus the committed `results/projection_audit.md`.

### Reason

Every later phase needs a trustworthy, independently-derived 2D ground truth
to score the tracker against -- one that can never leak into the camera-only
online path (METHODOLOGY.md's camera-only boundary).

### Validation

- 17 new tests (unit + integration), all passing (60 total in the project):
  known coordinate transform, behind-camera rejection, boundary clipping with
  nonzero truncation, positive finite area, deterministic ordering, MVP
  class-mapping, and identity preservation against the real dataset.
- Independent cross-check: the transform was reimplemented with `scipy`'s
  rotation library (instead of `pyquaternion`) and agrees to 1e-9 across 3
  rotation/translation cases -- satisfies the "compare with an independent
  reference" requirement without standing up the full official devkit.
- 5,384 of 18,538 considered annotations accepted (rest were behind the
  vehicle or outside the frame, as expected since annotations cover the full
  360-degree scene, not just CAM_FRONT); 3,492 map to the MVP's car/pedestrian
  scope.
- Visually reviewed 50 overlay frames (5 contact sheets, all 10 scenes, day/
  night/rain) -- tight alignment throughout, no systematic error, nothing
  needed a silent correction.
- `ruff check` clean; `git add -n` shows only source/tests/docs.

### Decision and next step

Task 3 has no separate student checkpoint in the assignment (only "automated
checks pass and every discrepancy is explained"), so continuing directly.
Next: Task 4 (mine and review natural occlusion events from real visibility
transitions) -- this task DOES have a student checkpoint requiring an actual
contact-sheet review with accept/reject/unsure answers, so it will pause for
that.

## 2026-08-04 — OATM Task 2: read-only mini audit + chronological CAM_FRONT index

### Change

Implemented `oatm.dataset.nuscenes_index` (chain-walking from each scene's
`prev`/`next` links, not filename or timestamp sort) and
`scripts/audit_dataset.py`. Wrote local-only `OATM/artifacts/frame_index.parquet`
(2,342 rows) and `dataset_audit.json`, plus the committed
`OATM/results/dataset_audit_summary.md`.

### Reason

Task 2's question: can the exact chronological CAM_FRONT stream be
reconstructed per scene, from the dataset's own links, with zero missing
files, zero broken links, and zero out-of-order frames? Nothing downstream
(projection, detection, tracking) can be trusted until this foundation is
verified.

### Validation

- Mini quality gate **PASSED** on the first real run against the local
  dataset: 10/10 scenes, 404/404 keyframes, 2,342/2,342 CAM_FRONT records,
  zero missing image files, zero non-monotonic timelines, every scene chain
  complete with exactly one head/tail and fully reciprocal links.
- 26/26 tests pass (9 new unit tests on synthetic fixtures + 3 new
  integration tests against the real local dataset); `ruff check` clean.
- Frame index rebuild is deterministic across two runs (tested).
- Student answered Task 2's checkpoint in her own words after one
  clarification each on two of the three questions (unsafe filename sorting;
  why future frames are forbidden online; why the large index stays local
  while the compact summary is committed).

### Decision and next step

**Stopping here per the assignment's mandatory mentor checkpoint** -- no
annotation projection, YOLO, tracking, or OATM logic begins until a mentor
reviews this Phase 0-1 output. Proposed a checkpoint commit; awaiting
confirmation before committing and before any Task 3 work starts.

## 2026-08-04 — OATM Task 0 (project map) and Task 1 (Phase 0 scaffold)

### Change

Created `OATM/results/project_map.md` (Task 0: plain-language terms, pipeline,
and method-comparison table). Then built the Phase 0 reproducible scaffold
(Task 1): `OATM/pyproject.toml` (pinned `pydantic`/`pyyaml`, dev deps
`pytest`/`ruff`), a project-local `OATM/.venv/`, `OATM/configs/mini.yaml`,
`src/oatm/config.py` (repo-relative data-root discovery, validated config with
readable errors), `src/oatm/records.py` (typed data contracts for every later
phase's frame index / projected ground truth / detector observations /
occlusion events / tracker output), 15 unit + integration tests, and
`OATM/README.md`. Added OATM-specific ignore rules to the root `.gitignore`.

### Reason

Task 0 gives a plain-English map before any code exists. Task 1 proves the
project can be installed and tested by someone else before any dataset,
detector, or tracker work begins -- exactly the assignment's "no experiment
before a trustworthy foundation" ordering.

### Validation

- Student answered Task 0's three checkpoint questions in her own words
  (`PREDICTED_HIDDEN` vs. a detection; why unlimited persistence would mislead
  recall; why staged builds stop one broken/masking component from hiding
  inside a bigger system).
- 15/15 pytest tests pass (`config` and `records` unit tests, one integration
  smoke test); `ruff check` passes with no findings.
- `find_data_root()` resolves the repo's real local `data/` folder (not
  `data/nuscenes/` as `AGENTS.md` describes -- same discrepancy already noted
  for the completed assignments) with no hardcoded absolute path; a missing or
  malformed config raises a readable `OATMConfigError` (tested).
- `git add -n` on `OATM/` shows only source, config, tests, and docs staged --
  no dataset, weights, `.venv/`, or cache directories.
- No YOLO run, no annotation projection, and no tracker code written yet, per
  Task 1's explicit scope limit.

### Decision and next step

Task 1's own checkpoint ("show `mini.yaml`, explain config vs. hardcoded vs.
private path") was answered inline in chat, not requiring a separate written
response. Next: Task 2 (read-only nuScenes mini audit and chronological
`CAM_FRONT` index) -- ending at the assignment's mandatory mentor checkpoint
before any projection, YOLO, or tracking work begins.

## 2026-08-03 — Added the student-facing OATM implementation assignment

### Change

Created `OATM/STUDENT_IMPLEMENTATION_ASSIGNMENT.md`, a Claude Code-ready master
assignment that divides the OATM implementation into 17 ordered tasks with
plain-language explanations, required outputs, automated checks, student
questions, and mentor stop points.

### Reason

The methodology and implementation plan define the research rigorously, but a
student needs a guided workflow like the four completed assignments: Claude
handles code and terminal work one task at a time while the student reviews the
scientific meaning and makes explicit human decisions.

### Validation

- Cross-checked the task order and terminology against `METHODOLOGY.md` and
  `IMPLEMENTATION_PLAN.md`.
- Preserved the Phase 0–1 stopping gate, causal camera-only boundary,
  scene-disjoint splitting, offline privileged-label boundary, separate event
  families, and recall-versus-ghost evaluation.
- Matched the instructional pattern of the earlier assignments with task goals,
  deliverables, completion checks, LLM rules, and student checkpoints.
- Ran `git diff --check` successfully.

### Decision and next step

Give the complete new assignment to Claude Code, begin with Task 0, and stop
after Task 2 for mentor review before projection or tracking work starts.

## 2026-08-01 — Tightened OATM methodology before experiments

### Change

Revised `OATM/METHODOLOGY.md` and `OATM/IMPLEMENTATION_PLAN.md` using the
evidence and limitations from Assignments 1–4. Separated strong detections,
weak detections, and prediction-only outputs; corrected the camera-only
ego-motion design; narrowed the first MVP; separated confidence quantities;
defined controlled visual occlusion versus detector intervention; and made
ghost-risk tradeoffs and valid experimental units explicit.

### Reason

The prior proposal combined too many unvalidated components, treated low
confidence too much like direct occlusion evidence, and contradicted its
camera-only claim by naming recorded ego pose as online input.

### Validation

- Cross-checked terminology, phase gates, schemas, metrics, and tomorrow's
  starting checklist across both OATM documents.
- Preserved causal inference and the privileged-data boundary.
- Grounded the revised scope in the small, preliminary assignment evidence
  without promoting pilot results into general claims.

### Decision and next step

Begin with Phase 0 and Phase 1 only. Establish the reproducible scaffold and
read-only chronological nuScenes-mini audit before projection, detection, or
tracking experiments.

## 2026-08-01 — Added a reproducible Assignment 4 environment

### Change

Added an assignment-local `pyproject.toml` for uv with the runtime and test
dependencies used by the ByteTrack experiment, plus an explicit `.venv`
ignore rule.

### Reason

Make the completed experiment and its automated checks reproducible without
depending on unrecorded system Python packages.

### Validation

- `uv lock` resolved 73 packages with standard PyTorch wheels, supporting CPU
  execution and CUDA when a compatible allocated resource is available.
- The assignment-local environment passed all 42 automated tests.
- The environment remains local and excluded from Git.

### Decision and next step

Use `uv sync` and `uv run pytest` from the Assignment 4 `experiment/`
directory whenever its saved results or implementation are audited.

## 2026-07-29 — Planned the OATM implementation

### Change

Created `OATM/IMPLEMENTATION_PLAN.md` with the pipeline, data contracts,
repository structure, phases, quality gates, experiment matrix, tests, risks,
definition of done, and a continuation checklist.

### Reason

Turn the methodology into a durable engineering and experimental handoff that
can resume without reconstructing decisions.

### Validation

- Fast-forwarded local `master` to merged `origin/master`.
- Grounded the plan in the methodology and earlier experiments.
- Kept privileged nuScenes evaluation data outside camera-only inference.
- Defined observed mini checks: 10 scenes, 404 keyframes, and 2,342
  `CAM_FRONT` records.

### Decision and next step

Implement Phase 0 and Phase 1 only: scaffold the project and build a read-only
chronological mini audit. Do not begin detection or tracking until that dataset
foundation passes its quality gate.

## 2026-07-29 — Installed nuScenes mini on IBEX

### Change

Downloaded and extracted the official `v1.0-mini.tgz` archive into the
Git-ignored repository data directory.

### Reason

Use the 10-scene mini split to validate the complete pipeline before committing
storage and compute to the trainval split.

### Validation

- Archive size matched the server response: 4,167,696,325 bytes.
- Gzip integrity and archive-path safety checks passed.
- Required `maps`, `samples`, `sweeps`, and `v1.0-mini` directories exist.
- Parsed 10 scenes, 404 keyframe samples, and 31,206 sample-data records.
- Found 2,342 `CAM_FRONT` records.
- Every sensor file referenced by `sample_data.json` exists.
- Git confirms that `data/nuscenes/` is excluded by `.gitignore`.

### Decision and next step

Keep the downloaded archive temporarily for reproducibility. Next, implement a
read-only dataset audit that checks temporal ordering, camera calibration,
ego-pose linkage, annotation projection, and candidate visibility transitions.

## 2026-07-28 — Created the OATM workspace

### Change

Moved the research methodology out of the completed coursework and into the
root-level `OATM/` workspace. Organized three earlier experiments under
`assignments/`.

### Reason

Separate completed learning assignments from the active research project while
preserving their evidence and Git history.

### Validation

- Assignment files, scripts, results, presentations, and samples remain present.
- Internal references were updated after the reorganization.
- Private source data remained untouched and excluded from Git.

### Decision and next step

Use the assignment results as baseline evidence, but implement the research
pipeline cleanly under `OATM/`.

## 2026-07-27 — Experiment 2: Static last-seen memory

### Question

Can a simple memory prevent YOLO from immediately forgetting a fully occluded
object, and how accurately does a frozen last-seen box represent its later
position?

### Method

Selected five valid nuScenes samples from the first experiment. When YOLO lost
the chosen target, the experiment retained its most recent bounding box without
moving or resizing it. Memory was explicitly labeled as a prediction rather
than current camera evidence and expired after two missing stages. The frozen
box was compared with the new YOLO box when the target reappeared.

### Results

- Five samples were valid and three were rejected because target identity or
  true occlusion could not be confirmed reliably.
- Memory kept a target report alive through the one-stage occlusion gap in all
  five valid samples.
- Mean center error at reappearance was 330.6 px; median error was 422.8 px.
- Mean IoU was 0.085; median IoU was 0.
- Three of five frozen boxes had zero overlap with the reappearing target.
- Best case: `sample_003`, with 82.9 px center error and 0.339 IoU.
- Worst case: `sample_006`, with 589 px center error and 0 IoU.
- The human helpful/misleading and ghost-risk columns were not completed, so no
  subjective counts were inferred.

### Decision

Static memory proves that object existence can persist beyond a missed
detection, but a frozen location becomes stale quickly and can create ghost
objects. OATM therefore needs motion prediction, uncertainty growth, explicit
occlusion state, and termination logic.

### Evidence

- `assignments/02_last_seen_memory/results/final_report.md`
- `assignments/02_last_seen_memory/results/summary.csv`
- `assignments/02_last_seen_memory/results/comparisons/`

## 2026-07-26 — Experiment 1: YOLO occlusion sensitivity

### Question

How does increasing visual occlusion affect a pretrained YOLO detector's ability
to detect the same road user?

### Method

Ran pretrained `yolo26n.pt` in prediction-only mode on 30 organized
`CAM_FRONT` images from eight manually selected nuScenes samples. Images
represented up to five stages: visible, first partial occlusion, full
occlusion, first partial reappearance, and full reappearance. One target per
sample was manually reviewed.

### Results

- Detection was 8/8 before occlusion and 3/3 in the available first-partial
  stage.
- Detection fell to 1/8, or 12%, during full occlusion.
- Mean target confidence fell from 0.77 before occlusion to 0.59 during first
  partial occlusion and 0.03 during full occlusion.
- Detection recovered to 7/7 at full appearance, with mean usable confidence
  0.84.
- The first-partial-appearance result was 1/2, too small to generalize.
- The only reported full-occlusion detection, `sample_012` at confidence 0.23,
  may have been a box on the occluder; incomplete review fields prevented a
  definitive classification.

### Decision

The small study supports the core problem statement: single-frame detection
confidence collapses when visual evidence disappears and recovers when the
target returns. The sample is not large enough for statistical claims, so OATM
must build a larger, scene-split occlusion evaluation set.

### Evidence

- `assignments/01_yolo_occlusion/results/final_report.md`
- `assignments/01_yolo_occlusion/results/run_summary.md`
- `assignments/01_yolo_occlusion/results/analysis_summary.csv`

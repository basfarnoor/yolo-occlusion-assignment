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

## 2026-08-07 — Final paper figure pack

### Change or experiment

Replaced the mixed synthetic/pilot paper-figure generator with a final-run-only
publication pipeline. Added a compact canonical table for
`lidar-fixes-20260805`, retained the approved online and offline SVG diagrams,
and generated six additional figures covering the final metric profile,
selective persistence, identity/localization, severe-visibility recall,
occluder-relative geometry, and the causal recovery sequence. Every current
figure is exported as SVG, PDF, and 600-DPI PNG.

### Reason

The paper requires one consistent evidence population, clear terminology, and
figures that remain readable under zoom and print reduction. Earlier paper
figures mixed synthetic mechanism evidence and a two-linkable-event pilot with
the final scene-disjoint result.

### Validation or evidence

The generator asserts the final run ID, validation population, IoU 0.30 gate,
1,873-annotation denominator, rounded F1 consistency, supported OATM metric
rankings, and the known overall/severe-visibility recall limitation. All output
formats pass structural, resolution, and DPI checks; rendered figures were
inspected at full and reduced size for clipping and label overlap. The full
test suite passes (39 tests), `ruff check .` passes, and `git diff --check`
reports no whitespace errors.

### Decision and next step

Use the eight figures listed in `OATM_RELATIONAL/results/paper_figures/README.md`
for the paper, poster, and presentation. Keep legacy synthetic and pilot assets
only for traceability, and preserve the final guide's claim boundaries in every
caption and discussion.

## 2026-08-07 — Simplified legacy natural-event matrix

### Change

Removed the obsolete `Selective OATM` comparison row from the natural-event
outcome matrix and regenerated the matching SVG, PDF, and 600-DPI PNG. The
remaining OATM, ByteTrack-12, and ByteTrack-5 rows now occupy the full matrix.

### Validation

Confirmed that the removed method no longer appears in the SVG, inspected the
figure at full and reduced size, checked the PDF structure, and verified the
PNG dimensions and 600-DPI metadata.

## 2026-08-06 — Consolidated presentation guide around final evaluation

### Change or experiment

Reworked `OATM_RELATIONAL/REPORT_PRESENTATION_GUIDE.md` so run
`lidar-fixes-20260805` is the only numerical result used in the report, slides,
and poster. Removed synthetic, pilot, earlier-study, and internal development
metrics from the results narrative. Added one matched final comparison, a
cohesive strongest-insights section, final-run reproduction commands, and
updated slide, poster, caption, examiner, and checklist guidance.

### Reason

The student presentation needs one consistent evidence population rather than
a mixture of experiments with different protocols and sample sizes. Positive
findings should be easy to communicate, while weaknesses remain explicit and
consolidated under Limitations.

### Validation or evidence

Cross-checked every retained number against the validation rows of
`lidar-fixes-20260805`; confirmed the guide contains no synthetic or natural
pilot result references and passes `git diff --check`.

### Decision and next step

Use only the final scene-disjoint table for numerical claims. Lead with OATM's
precision, F1, MOTA, IDF1, identity, localization, predicted-hidden precision,
and unsupported-track control; keep recall, severe visibility, mini-dataset
scale, and annotation limitations in the dedicated limitations section.

## 2026-08-06 — Clarified LiDAR-supported occlusion interpretation

### Change or experiment

Expanded the student presentation guide with the exact offline occlusion
interpretation: nuScenes visibility tokens define coarse visible-fraction
bins, projected 3D annotations provide `CAM_FRONT` evaluation boxes, and LiDAR
point counts are sensitivity metadata rather than an occlusion classifier.
Added an examiner-ready explanation of the same boundary.

### Reason

Calling the protocol a LiDAR evaluation can incorrectly imply that zero LiDAR
returns prove occlusion or that LiDAR enters OATM inference. Neither is true.

### Validation or evidence

Cross-checked the guide against local `visibility.json`, the annotation
projection implementation, and the visibility/LiDAR stratification code.

### Decision and next step

Describe visibility token `1` as the most-occluded coarse annotation bin, not
exact pixel-level occlusion truth. Continue to require manual `CAM_FRONT`
review or dedicated labels for exact event-level occlusion claims.

## 2026-08-05 — Boundary and silent-reactivation lifecycle repair

### Change or experiment

Replaced edge-touch exit termination with an outward-motion plus visible-box
fraction rule. Added an optional conditional mature-grace ablation and a
promoted one-frame silent `DORMANT` identity state. Dormant tracks emit no
prediction, cannot compete in ordinary ByteTrack association, and reconnect
only through a separately thresholded strict reappearance stage. Frozen the
development-selected configuration at visible fraction 0.60, one dormant
frame, and dormant reappearance score 0.75; conditional mature grace remains
disabled.

### Reason

The first LiDAR-supported run showed excessive `predicted_exit` terminations,
low truncated-object recall, and a recall gap to ByteTrack. Generic grace
increased false persistence, while the first dormant prototype hijacked normal
detections and raised identity switches. Silent strict reactivation preserves
an identity opportunity without emitting an unsupported box.

### Validation or evidence

- Frozen-development ablations are stored under the ignored
  `OATM_RELATIONAL/lidar_eval/results/` directory. The selected candidate
  improved development recall from 0.324 to 0.351, F1 from 0.467 to 0.489,
  MOTA from 0.225 to 0.232, IDF1 from 0.383 to 0.399, and truncated recall
  from 0.453 to 0.487; identity switches fell from 56 to 55.
- `uv run --frozen pytest -q`: 37 passed; `uv run --frozen ruff check .`:
  passed.
- Full frozen-cache run `lidar-fixes-20260805` processed 2,342 causal
  `CAM_FRONT` frames and scored 3,492 projected annotations.
- On 1,873 validation annotations at IoU 0.30, final OATM reached 0.841
  precision, 0.266 recall, 0.404 F1, 0.184 MOTA, 0.303 IDF1, and 59 identity
  switches. ByteTrack-5 reached 0.734, 0.276, 0.401, 0.142, 0.296, and 64;
  ByteTrack-12 reached 0.553, 0.286, 0.377, 0.011, 0.274, and 82.
- Relative to the pre-repair validation run, final OATM improved recall by
  1.66 points, F1 by 1.62 points, MOTA by 0.37 points, IDF1 by 1.61 points,
  and reduced identity switches by two. Overall precision fell by 3.2 points
  and fragmentation increased from 27 to 28. Predicted-hidden precision rose
  from 0.378 to 0.394, while truncated-object recall rose from 0.301 to
  0.364 (+6.32 percentage points).

### Decision and next step

Promote the visible-fraction exit rule and one-frame strict silent reactivation.
The final method now narrowly exceeds ByteTrack-5 in validation F1 and leads
both ByteTrack arms in precision, MOTA, IDF1, identity switches, localization,
and predicted-hidden precision. It still trails both in overall and
most-occluded-bin recall, so report a balanced-metric improvement rather than
universal superiority. The next revision should improve relation formation on
severe occlusions without lengthening generic prediction output.

## 2026-08-05 — Scene-disjoint CAM_FRONT LiDAR-supported evaluator

### Change or experiment

Added `OATM_RELATIONAL/lidar_eval/`, a separated three-stage evaluation
package: frozen YOLO cache validation or optional GPU regeneration, causal
`CAM_FRONT` tracking for ByteTrack-5, ByteTrack-12, and promoted Relational
OATM, followed by privileged offline matching against projected official
nuScenes annotations. Added a four-hour single-A100 Slurm job, versioned config,
non-overwriting per-run result directories, hashes/runtime provenance, a full
README, and focused boundary/matching/metrics tests.

The evaluator uses class-aware Hungarian matching at IoU 0.30 with 0.10/0.50
sensitivity; separates observed from predicted-hidden outputs; reports
precision, recall, MOTA, IDF1, switches, fragmentation, localization, false
outputs, and a clearly labeled unsupported-keyframe-track ghost proxy; and
stratifies by class, visibility, depth, LiDAR-point support, and truncation.
Five development and five validation scenes are assigned by stable seeded hash.

### Reason

The six-event/two-linkable-event natural pilot is not large enough to establish
real-data behavior and mixes dense sweep persistence with sparse-keyframe
localization. The wider evaluation must use LiDAR-supported annotations only as
offline evidence, preserve the camera-only inference boundary, avoid treating
unannotated sweeps as empty ground truth, use common ground-truth denominators,
and expose anti-ghost versus persistence tradeoffs rather than optimize a
favorable isolated score.

### Validation or evidence

- `uv run --frozen ruff check lidar_eval`: passed.
- `uv run --frozen pytest -q`: 27 passed (20 existing plus 7 evaluator tests).
- `bash -n lidar_eval/submit_a100.sbatch`: passed.
- Existing frozen detector cache validated: 2,342 `CAM_FRONT` frames, 49,436
  camera detections, matching model hash/configuration and valid row geometry.
- Full integration run `local-validation-split-20260805` completed over all ten
  mini scenes and scored 3,492 annotations; all outputs and metadata remain in
  the ignored local result directory for inspection.
- On 1,873 annotations in five validation scenes at IoU 0.30, Relational OATM
  reached 0.873 precision, 0.249 recall, 0.180 MOTA, 0.287 IDF1, and 20.053 px
  mean center error. ByteTrack-5 reached 0.734, 0.276, 0.142, 0.296, and 21.084;
  ByteTrack-12 reached 0.553, 0.286, 0.011, 0.274, and 21.361 respectively.
  Relational OATM's predicted-hidden precision was 0.378 versus 0.250 and 0.140,
  but its severe-visibility recall was 0.077 versus 0.093 and 0.107.

### Decision and next step

The evaluator is ready for `sbatch`. The current evidence shows strong
anti-ghost precision, MOTA, and localization but does **not** show overall or
severe-occlusion recall superiority over ByteTrack. Treat this as the measured
design target for the next method revision: improve relation formation and
safe short-term persistence without sacrificing the false-prediction control.
Do not claim superiority over ByteTrack until that improvement survives the frozen
scene-disjoint protocol.

## 2026-08-05 — Final OATM report and presentation guide

### Change or experiment

Added `OATM_RELATIONAL/REPORT_PRESENTATION_GUIDE.md` as the single student
writing source for the report, slides, and poster. It presents the public method
as OATM, includes report-ready methodology, Mermaid pipeline and state-machine
drawings, parameters, experiment design, comparisons with YOLO-only, last-seen
memory, SORT, and ByteTrack, final result tables, claim boundaries, suggested
slides/poster layout, captions, examiner questions, and ready-to-use prose.

### Reason

The student needs one evidence-backed narrative that describes the final method
without exposing internal development versions or mixing results produced under
different experimental protocols.

### Validation or evidence

Cross-checked the final OATM numbers against synthetic run `eac923a94d04` and
natural run `806945a64e0d`; cross-checked earlier-study values against their
committed final reports. Direct performance claims use only identical-input
ByteTrack arms. The guide preserves the camera-only causal boundary, separates
synthetic and natural evidence, states the two-linkable-event limitation, and
contains no internal method-version names.

### Decision and next step

Use this guide as the canonical student-facing writing reference. Keep broader
ByteTrack superiority and benchmark-reproduction claims out of the report until
controlled visual, verified negative, and larger scene-disjoint studies exist.


## 2026-08-05 — Relational localization and lifecycle repair

### Change or experiment

Traced the two linkable natural events frame by frame and repaired three causal
state failures in `OATM_RELATIONAL`: primary occluder ownership is immutable
within a hidden episode, decoded occluder anchors must agree with independent
target motion within bounded center/scale residuals, and reappearance search has
a hard spatial cap. Resolved relations are archived before the next frame so a
stale clearing age cannot terminate a new hidden episode.

### Reason

The earlier 387.890 px error came from tracked occluders jumping hundreds of
pixels and dragging their targets, followed by distant weak detections being
accepted under uncertainty. The first conservative repair fixed localization
but reduced coverage to 0.163; lifecycle tracing then exposed the stale-resolved
relation bug and a normal-perspective residual that was too strict.

### Validation or evidence

- `pytest -q`: 20 passed, including inconsistent-anchor, distant-reappearance,
  same-class hijack, and resolved-relation lifecycle regressions.
- Synthetic run `eac923a94d04`: 1.000 hidden coverage and same-ID recovery,
  4.403 px error, 2.000 negative ghost frames, and zero wrong associations.
- Natural run `806945a64e0d`: Relational OATM reached 0.620 hidden coverage,
  0.500 fully bridged rate, one same-ID recovery, and 16.019 px error.
  OATM_UPDATED reached 0.430, 0.000, zero, and 15.858 px respectively.
  ByteTrack-12 retained higher coverage at 0.760.

### Decision and next step

The catastrophic localization problem is fixed and the safeguards are promoted.
The current pilot supports improvement over OATM_UPDATED on two linkable events,
but not a general ByteTrack-superiority claim. Next build controlled visual and
verified negative sets, then calibrate and evaluate on scene-disjoint splits.


## 2026-08-05 — Relational OATM implementation and evaluation

### Change or experiment

Created `OATM_RELATIONAL/` as an isolated `uv` workspace extending
`OATM_UPDATED` with explicit target--occluder state, occluder-centric geometry,
expected-clearance termination, protected third-stage reappearance, a causal
camera-motion ablation, reproducible nuScenes preparation/detection scripts,
tests, compact reports, metadata, and charts. Hidden relational tracks are no
longer eligible for ordinary ByteTrack association while an active occluder is
visible.

### Reason

Test whether added relational structure improves the hidden-recall versus ghost
duration frontier without presenting longer generic track lifetime as an OATM
contribution.

### Validation or evidence

- `uv run pytest -q`: 17 passed; `uv run ruff check .`: clean at final audit.
- Synthetic run `e52db2dee70d`: Relational OATM achieved 1.000 mean hidden
  coverage, 1.000 same-ID recovery, 4.403 px center error, 1.333 negative ghost
  frames, and zero wrong associations. ByteTrack-5 achieved 0.943 coverage,
  0.600 same-ID recovery, 4.813 px error, and 5.000 ghost frames.
- Read-only nuScenes preparation: 10 scenes, 2,342 CAM_FRONT frames, 404
  keyframes, and 5,384 accepted privileged projected annotations.
- Cold YOLO26n CPU detector run: exactly 49,436 detections over 2,342 frames;
  weights SHA-256 `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`.
- Final natural pilot `55b3a52bf229`: only 2/6 reviewed events were linkable.
  Relational OATM coverage was 0.457 versus 0.245 ByteTrack-5, 0.430 Selective
  OATM, and 0.760 ByteTrack-12. Relational error was 387.890 px and both
  recoveries used new IDs. Camera-enabled pilots also showed severe drift.

### Decision and next step

The complex architecture and experiment pipeline are implemented, but the
real-data superiority acceptance check failed. Keep camera compensation disabled
and do not claim ByteTrack superiority. Next stabilize target--occluder
selection/localization, build human-verified real negative events, run controlled
visual experiments, then repeat a statistically powered scene-disjoint study.


## 2026-08-05 — OATM_UPDATED selective-occlusion foundation

### Change

Created `OATM_UPDATED/` as an isolated `uv` workspace with methodology,
pipeline, implementation plan, configuration, source, tests, compact results,
and ignored artifacts. `SelectiveOATMTracker` retains the audited ByteTrack
association/Kalman contracts but admits a missing mature track only with
camera-derived visible-occluder evidence or a bounded one-frame grace period.

### Reason

The prior MVP did not outperform ByteTrack overall. The updated experiment asks
whether selective activation improves hidden recall at matched ghost duration;
a longer-buffer ByteTrack arm prevents extra lifetime from being mistaken for a contribution.

### Validation or evidence

- `uv.lock` generated with uv 0.12.1; run `ec2393733008` used Python 3.12.13.
- `ruff check .` clean; `pytest -q`: 7 passed.
- Synthetic development: Selective OATM and long-buffer ByteTrack both reached
  100% occlusion coverage and same-ID recovery. Mean miss/exit ghost duration
  was 0.5 frames for Selective OATM, 5.0 for ByteTrack, and 6.0 for long-buffer ByteTrack.
- Tests exposed and drove repair of stable occluders disappearing from gate evidence.

### Decision and next step

This validates mechanics, not real-data superiority. Next regenerate detector
and projection artifacts, add verified exit/loss negatives, tune on development
scenes only, and evaluate controlled visual then held-out natural occlusions.

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

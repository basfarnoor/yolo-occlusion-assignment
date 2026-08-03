# OATM Student Implementation Assignment

> This file is both your assignment and your working prompt for Claude Code.
>
> Open the `yolo-occlusion-assignment` project in Claude Code and give Claude
> this entire file. Say:
>
> **"Help me complete the OATM project one numbered task at a time. Do the
> coding and terminal work yourself, explain each step in simple language, run
> all checks, and stop at every student or mentor checkpoint."**
>
> Do **not** ask Claude to complete the whole project in one session. This is a
> research project, so each stage must be understood and verified before the
> next stage begins.

## What You Are Building

OATM means **Occlusion-Adaptive Temporal Memory**.

Imagine that a camera sees a pedestrian, but a bus passes in front of that
person for a few moments. A normal single-image detector may behave as if the
pedestrian disappeared. OATM tries to remember the pedestrian, estimate where
they may be while hidden, reconnect the same identity when they return, and
stop the prediction if it becomes unsafe or unreliable.

OATM must distinguish three very different statements:

1. **"I can see the object now."** A current camera detection supports it.
2. **"I cannot see it, but I have evidence that it is temporarily hidden."**
   This is a temporal prediction, not a detection.
3. **"I no longer have enough evidence."** The track must be ended instead of
   becoming a ghost object.

The system will use a frozen object detector and add carefully tested temporal
logic around it. You are not training a large new detector.

## Research Question

> Can an occlusion-aware, uncertainty-calibrated temporal memory improve object
> recall and identity continuity during temporary camera occlusions, compared
> with single-frame detection and conventional tracking, without creating an
> unacceptable number of false persistent tracks?

## Your Main Learning Goals

By the end of this project, you should be able to explain:

- Why an object detector can lose a hidden object.
- The difference between a detection, a track, and a prediction.
- Why video frames must be processed in chronological order.
- Why motion can help during a short occlusion but can also drift.
- Why low-confidence evidence is useful but is not proof of occlusion.
- How OATM distinguishes `OBSERVED_STRONG`, `OBSERVED_WEAK`,
  `PREDICTED_HIDDEN`, `LOST`, and `EXITED`.
- Why uncertainty should increase while an object is unseen.
- Why recovered-object recall and ghost-track risk must be reported together.
- Why nuScenes LiDAR-supported labels may evaluate the system but may not be
  secretly supplied to the camera-only method.
- How automated tests, saved configurations, and experiment records make a
  research result trustworthy.

## The Most Important Scientific Boundary

The final online OATM method is **camera-only and causal**.

- At frame `t`, it may use the current camera image and earlier information.
- It may never use a future frame to improve the current prediction.
- nuScenes 3D annotations, instance identities, visibility labels, LiDAR,
  calibration, and recorded ego poses may help construct or evaluate the
  experiment offline.
- Privileged labels must not enter the online tracking decision.
- Recorded ego pose may be used only in a clearly named offline or
  `oracle_ego_pose` diagnostic. It is not an input to the headline method.
- Current detections and memory-only predictions must always have different
  evidence labels.

If Claude is unsure whether some information is allowed online, it must stop
and check `OATM/METHODOLOGY.md` and `OATM/IMPLEMENTATION_PLAN.md` before using
it.

## How You and Claude Will Work Together

Claude performs the programming, terminal, testing, plotting, and routine Git
work. You are responsible for understanding the experiment and making the
small scientific decisions requested at checkpoints.

At the beginning of every task, Claude must tell you:

1. The question this task answers.
2. What it will create or change.
3. Which files are inputs and which are outputs.
4. How it will prove that the result is correct.

At the end of every task, Claude must show you:

1. A simple explanation of what now works.
2. The files it created or changed.
3. The exact tests or validation checks it ran.
4. Any failed checks, uncertainty, or limitation.
5. A short question that checks your understanding.
6. Whether the next task is safe to begin.

After every material task or experiment, Claude must add a dated entry near the
top of `LOG.md` using the repository's required log format. It should then show
the proposed Git changes and suggest a small checkpoint commit. It must not
push private data or bypass a mentor checkpoint.

Claude must not hide errors, invent results, or continue through a failed
quality gate. A negative experimental result is acceptable; an unverified
result is not.

## Required Reading Before Any Implementation

Claude must read these files first:

- `AGENTS.md`
- `README.md`
- `LOG.md`
- `OATM/METHODOLOGY.md`
- `OATM/IMPLEMENTATION_PLAN.md`
- This assignment

Claude should consult the completed assignments as evidence, especially their
READMEs, tests, and final reports. It must preserve those assignments and must
not copy their code blindly. Reusable code must first be reviewed for
correctness, tests, and compatibility with the OATM methodology.

---

# Part A — Build a Trustworthy Foundation

## Task 0 — Make a Beginner's Map of the Project

Before writing code, Claude must create:

`OATM/results/project_map.md`

The map must explain these terms in one or two simple sentences each:

- Object detection
- Bounding box
- Confidence score
- Strong and weak detection
- Track and track ID
- Temporal memory
- Occlusion
- Ordinary detector miss
- Field-of-view exit
- Motion prediction
- Localization uncertainty
- Identity switch
- Ghost track
- Causal or online inference
- Privileged ground truth
- Natural occlusion
- Controlled visual occlusion
- Detector intervention
- Development, validation, and test split
- Ablation study

It must include this simple pipeline:

```text
current camera frame
    -> frozen object detector
    -> strong and weak detections
    -> associate detections with existing tracks
    -> update motion and uncertainty
    -> decide whether an unmatched object is hidden, lost, or exited
    -> keep or terminate the prediction
    -> reconnect the same identity if compatible evidence returns
    -> save detections and predictions with different labels
```

The map must also include a small table comparing:

| Method | What it remembers | Main limitation |
|---|---|---|
| YOLO only | Nothing between frames | Loses objects when evidence disappears |
| Static last-seen memory | Last box | Does not move with the object or camera |
| SORT | Simple motion | Can drift and does not explicitly reason about occlusion |
| ByteTrack | Strong and weak detections plus motion | Still needs visible evidence and can retain wrong tracks |
| OATM | Evidence type, motion, uncertainty, and occlusion state | More rules to validate and a risk of ghost predictions |

### Student checkpoint

Explain to Claude, in your own words:

1. Why `PREDICTED_HIDDEN` is not the same as a detection.
2. Why keeping every missing object forever would give misleading recall.
3. Why OATM is being built in small stages.

Claude may help you understand the questions, but it must not write your
answers for you.

**Task 0 is complete when:** the project map exists and you can explain the
difference between current visual evidence and temporal prediction.

---

## Task 1 — Create the Reproducible OATM Scaffold

The question for this task is:

> Can another computer install, test, and run the empty OATM project in a
> predictable way?

Claude must create the Phase 0 project structure described in
`OATM/IMPLEMENTATION_PLAN.md`. At minimum, this includes:

```text
OATM/
  README.md
  pyproject.toml
  configs/
    mini.yaml
  src/oatm/
    __init__.py
    config.py
    records.py
  scripts/
  tests/
    unit/
    integration/
  results/
    README.md
  artifacts/
```

Requirements:

- Pin and record direct Python dependencies.
- Use repository-relative paths instead of one person's absolute path.
- Validate configuration values and give readable error messages.
- Add deterministic random-seed handling and structured run metadata.
- Configure automated tests, formatting, and static checks.
- Keep environments, datasets, weights, caches, and large generated artifacts
  out of Git.
- Document setup and commands in `OATM/README.md`.
- Do not run YOLO, project annotations, or implement a tracker yet.

Claude must inspect existing uncommitted work before editing and preserve
unrelated files.

### Required checks

- A clean environment can install the project.
- The configuration resolves the local `data/nuscenes/` directory without
  hardcoding an absolute path.
- A tiny test can import the `oatm` package and load `mini.yaml`.
- All configured tests and code-quality checks pass.
- Git does not show the dataset, environment, model weights, or caches.

### Student checkpoint

Ask Claude to show you `mini.yaml` and explain the difference between:

- A setting saved in configuration.
- A value hardcoded inside Python.
- A private local path that must not be committed.

**Task 1 is complete when:** the empty project installs and tests successfully,
and no experiment has begun prematurely.

---

## Task 2 — Audit nuScenes Mini Without Changing It

The question for this task is:

> Can OATM reconstruct the full chronological `CAM_FRONT` stream without
> missing files, broken links, or non-causal ordering?

The dataset is a read-only input. Claude must never rename, move, edit, or
delete original nuScenes files.

Claude must:

1. Discover the local nuScenes mini directory.
2. Load its metadata safely.
3. Reconstruct every `CAM_FRONT` sequence using `prev` and `next` links.
4. Check that timestamps strictly increase within each scene.
5. Check reciprocal links, file existence, image dimensions, keyframe flags,
   calibration references, and pose references.
6. Create a typed chronological frame index.
7. Record the dataset version, package versions, configuration, seed, and run
   time.

Create:

- Local-only `OATM/artifacts/frame_index.parquet`
- Local-only `OATM/artifacts/dataset_audit.json`
- Local-only detailed audit data if needed
- `OATM/results/dataset_audit_summary.md`
- Unit and integration tests for ordering and link validation

Every frame row must include the canonical fields specified in
`OATM/IMPLEMENTATION_PLAN.md`, including scene, frame, token, timestamp,
keyframe status, relative image path, and neighboring tokens. Generated tables
must include a `schema_version`.

### Required mini quality gate

The audit must report:

- Exactly 10 scenes.
- Exactly 404 keyframes.
- Exactly 2,342 `CAM_FRONT` records.
- Zero missing camera images.
- Zero non-monotonic scene timelines.
- Complete calibration and pose references.

If any expected number is different, Claude must investigate and report the
exact reason. It must not change a test merely to make it pass.

### Student checkpoint

Ask Claude to show you one scene as a short table containing frame number,
timestamp, keyframe status, and previous/next links. Then explain:

1. Why sorting only by filename could be unsafe.
2. Why future frames are forbidden during online inference.
3. Why the large Parquet index is local while a compact audit is reviewable.

### Mandatory mentor checkpoint

**Stop here.** Do not begin annotation projection, YOLO, tracking, event
mining, or OATM logic until the mentor reviews the Phase 0 and Phase 1 outputs.

**Task 2 is complete when:** all exact mini checks pass, the audit is
reproducible, and the mentor approves continuation.

---

# Part B — Construct the Experiment

## Task 3 — Project Offline Ground Truth into the Camera Image

The question for this task is:

> Can the official 3D annotations be converted into reliable 2D evaluation
> boxes without leaking labels into online tracking?

Claude must implement and test the global-to-ego-to-camera-to-image transform.
It must reject geometry behind the camera, clip boxes at image boundaries, and
preserve original nuScenes identities and metadata.

Create:

- Local-only `OATM/artifacts/projected_ground_truth.parquet`
- `OATM/results/projection_audit.md`
- Local-only deterministic projection overlays
- Projection and clipping tests

Each valid projected row must follow the canonical projected-ground-truth
contract in `OATM/IMPLEMENTATION_PLAN.md` and include `schema_version`.

Required tests include:

- A known coordinate transform.
- Transform round-trip behavior where applicable.
- Behind-camera rejection.
- Image-boundary clipping.
- Positive finite box area.
- Deterministic ordering.
- Preservation of scene, frame, annotation, and instance identity.

Claude must compare a subset with the official nuScenes development tools or
another independent reference. At least 50 overlays must be visually reviewed.
The student may inspect contact sheets, but must not manually edit coordinates.

Every report must state:

> These annotations are privileged offline evaluation evidence. They are not
> inputs to the online camera-only tracker.

**Task 3 is complete when:** the automated checks pass and every reviewed
projection discrepancy is explained rather than silently corrected.

---

## Task 4 — Find and Review Natural Occlusion Events

The question for this task is:

> Which nuScenes objects have trustworthy visible-hidden-visible events that
> can be used to test OATM?

Claude must group annotations by scene and `instance_token`, identify visibility
decline followed by recovery, and reject likely exits or truncation. Candidate
ranking may use privileged metadata because this is offline dataset
construction, not online inference.

Each candidate should preserve:

- Scene, sample, frame, annotation, and instance tokens.
- Pre-occlusion, start, end, and post-occlusion boundaries.
- Visibility pattern.
- Target class, size, depth, and image-boundary information.
- Possible occluder and overlap evidence where available.
- Review status and rejection reason.
- A split derived from the scene, never from neighboring frames.

Create:

- Local-only candidate event table and review images/videos
- `OATM/results/natural_event_manifest.csv`
- `OATM/results/natural_event_selection.md`
- An immutable accept/reject review record
- Tests for event boundaries and scene-disjoint splits

Natural event review must keep these causes separate:

- True occlusion.
- Field-of-view exit.
- Image truncation.
- Ordinary detector miss.
- Poor image quality.
- Ambiguous or unsupported event.

Visibility change alone is not enough. An accepted event needs at least two
independent signals supporting occlusion and traceable visual review evidence.

### Student checkpoint

Review a small contact sheet selected by Claude. For each shown candidate,
choose `accept`, `reject`, or `unsure`, and write a short reason. Claude must
record your choice exactly; it must not pretend you reviewed unseen examples.

**Task 4 is complete when:** every accepted event is traceable to source data,
has review evidence, and belongs to a scene-disjoint split. The mini result must
be called a pilot, not a final statistical conclusion.

---

## Task 5 — Run the Frozen Detector Once and Cache Its Observations

The question for this task is:

> Can every comparison method receive the exact same detector evidence?

Use a lightweight pretrained YOLO-family detector consistent with the earlier
assignments where practical. Do prediction only; do not train or fine-tune the
detector.

Claude must:

1. Process images chronologically.
2. Keep both strong and weak detections above a documented low floor.
3. Record raw detector boxes without replacing them with tracker-smoothed boxes.
4. Fingerprint the image, model, weights, image size, confidence floor, and
   relevant package versions in the cache key.
5. Reuse cached output only when the complete key matches.
6. Match detections to ground truth only in the offline evaluation layer.

Create:

- Local-only detector cache
- Local-only or compact `OATM/artifacts/detections.parquet`
- `OATM/results/detector_audit.md`
- A compact confidence-distribution chart or summary
- Tests proving that cache hits do not rerun inference

Each detector row must follow the canonical detector-observation contract and
must say which frame and configuration produced it.

Thresholds must be selected on development scenes and saved in configuration
before evaluation scenes are opened. Claude must never tune thresholds after
seeing final test results.

**Task 5 is complete when:** every input frame has a traceable detector result
or documented failure, and all methods can read the same immutable observation
table.

---

## Task 6 — Rebuild and Verify the Baselines

The question for this task is:

> What do standard methods achieve before any new OATM logic is added?

Implement or safely adapt these methods behind one common interface:

1. YOLO-only output.
2. Static last-seen memory.
3. SORT.
4. ByteTrack-style strong/weak association.

Claude must audit code from Assignments 1–4 before reuse. For every reused,
repaired, or replaced component, create:

`OATM/results/reuse_audit.md`

The audit must record the source path, decision, tests, and remaining scientific
risk.

All methods must receive identical ordered frames and raw detections. Track IDs
must never cross scene boundaries. Reappearance IDs must come from real
association, never from ground-truth identity or manual assignment.

Required tests include:

- Identical and disjoint IoU cases.
- One-to-one assignment.
- Timestamp-aware Kalman prediction.
- Track birth, missing, reactivation, and expiry.
- Strong-before-weak ByteTrack association.
- An unmatched weak detection cannot create a new track.
- Scene boundaries reset all active tracks.
- Ground-truth tables are inaccessible through the online tracker interface.
- Identical inputs and seed produce identical outputs.

**Task 6 is complete when:** all baselines run from one configuration, share
identical evidence, and pass synthetic plus one-clip integration tests.

---

## Task 7 — Build Two Controlled-Occlusion Families

The question for this task is:

> How do the methods behave under precisely known missing or weakened evidence?

These two experiments must remain separate:

### Family A — Detector intervention

Demote or remove selected detector rows while leaving image pixels unchanged.
This isolates tracker behavior. It must never be called visual occlusion.

### Family B — Controlled visual occlusion

Modify copied input pixels using seeded masks or realistic foreground objects,
then rerun the same frozen detector. Never edit the original nuScenes images.

Vary duration, coverage, target class and size, and relative motion. Record the
source frame, target, seed, mask or transformation, coverage, cache key, and
event family so every altered frame can be recreated exactly.

Create:

- `OATM/results/controlled_event_manifest.csv`
- `OATM/results/controlled_protocol.md`
- Local-only altered images and detailed outputs
- Tests for deterministic recreation and method parity

### Student checkpoint

Inspect several controlled visual examples. Confirm that the selected target is
actually covered and that the source image remains unchanged.

**Task 7 is complete when:** every method receives the same events, every event
can be regenerated from its manifest, and natural, visual, and intervention
results cannot be merged accidentally.

---

# Part C — Implement the OATM Minimum Viable Method

## Task 8 — Add Motion Memory and Growing Uncertainty

The question for this task is:

> Is motion prediction better than freezing the last box during short moving
> occlusions, and does the system know when its prediction becomes uncertain?

Begin with two models:

1. Stationary prediction.
2. Timestamp-aware constant-velocity Kalman prediction.

For each active track, store recent locations, size, timing, velocity, state,
and covariance or another explicit localization-uncertainty representation.
Uncertainty must grow during missing evidence.

Test synthetic tracks with:

- A stationary object.
- Smooth motion.
- Slow motion.
- Unequal timestamp gaps.
- Turning motion.
- Abrupt motion.
- A missing interval followed by reappearance.

Compare stationary and constant-velocity prediction by motion regime. Report a
negative or mixed result honestly if motion does not help.

Do not add visual ego-motion compensation or appearance embeddings yet.

### Student checkpoint

Ask Claude to draw one simple example showing a predicted box and an uncertainty
region after one, three, and five missing frames. Explain why the uncertainty
region should expand.

**Task 8 is complete when:** motion prediction is tested against static memory,
uncertainty grows correctly, and the result is reported without selecting only
favorable tracks.

---

## Task 9 — Implement the OATM Evidence States

The question for this task is:

> Can the system distinguish current strong evidence, current weak evidence,
> plausible occlusion, loss, and exit?

Implement these exact states:

- `OBSERVED_STRONG`
- `OBSERVED_WEAK`
- `PREDICTED_HIDDEN`
- `LOST`
- `EXITED`

Initial evidence may include confidence decline, foreground overlap, a
camera-derived relative-depth proxy, trajectory consistency, boundary and
outward motion, uncertainty, and elapsed time.

Low confidence alone is weak support, not proof of occlusion.

Every output row must include an unambiguous evidence source such as:

- `strong_detection`
- `weak_detection`
- `motion_prediction`
- No output after loss or exit

Rules that must always hold:

- `OBSERVED_STRONG` and `OBSERVED_WEAK` have a current associated detection.
- `PREDICTED_HIDDEN` has no current detection and must be clearly marked as a
  prediction.
- `LOST` cannot silently reconnect to an old ID; a later object starts a new
  track unless the defined lifecycle still permits real association.
- `EXITED` ends the track.
- A state decision may use only current or earlier online information.

Required tests include an exhaustive transition truth table and fixtures for:

- True occlusion.
- Field-of-view exit.
- Ordinary miss without occluder evidence.
- Poor visibility.
- False initial track.
- Compatible and incompatible reappearance.

**Task 9 is complete when:** every allowed transition is tested, impossible
transitions are rejected, and the method does not keep every missing detection
as hidden.

---

## Task 10 — Add Adaptive Confidence and Anti-Ghost Termination

The question for this task is:

> When should OATM stop remembering an unseen object?

Keep these quantities separate:

- Detector confidence.
- Existence confidence.
- Identity confidence.
- Localization uncertainty.

Existence confidence must decay with elapsed time and incremental uncertainty.
It cannot increase without new evidence. A track may terminate because of:

- Existence confidence below its floor.
- Localization uncertainty above its ceiling.
- Predicted field-of-view exit.
- An impossible occluder relationship.
- Failure to reappear when expected.

Every terminated track must record exactly one primary termination reason.
Thresholds are tuned only on development or validation scenes and then frozen.

Compare:

1. A fixed lifetime.
2. An uncertainty-aware adaptive lifetime.

Compare them at matched ghost risk, not only at their best recall setting.

Required tests include:

- Confidence monotonicity without new evidence.
- Correct use of elapsed seconds rather than frame count alone.
- Exactly one termination reason.
- Earlier termination for clear exits than plausible occlusions.
- Thresholds do not change when evaluation scenes run.

**Task 10 is complete when:** OATM can bridge a supported short occlusion, stop
an unsupported or exited track, and explain every decision in its saved output.

---

# Part D — Evaluate Before Adding Complexity

## Task 11 — Run the First Complete OATM MVP Study

The question for this task is:

> Does the motion-state-termination MVP improve the tradeoff between hidden
> recall and harmful false persistence?

Compare on identical inputs:

- YOLO only.
- Static memory.
- SORT.
- ByteTrack.
- Fixed-window memory if feasible.
- OATM MVP.

Keep results separate for:

1. Natural occlusions.
2. Controlled visual occlusions.
3. Detector interventions.

Report at minimum:

- Occluded-object recall.
- Visible-object precision and recall.
- Identity preservation and identity switches.
- Center error and IoU while hidden.
- Ghost event rate and ghost duration.
- Maximum recovered gap.
- Reappearance recovery latency.
- Existence-confidence calibration.
- Runtime for each method.

The main experimental unit is an event or track, not a repeated frame row.
Reports must state counts of scenes, clips, tracks, events, unique frames, and
measurement rows separately.

Create:

- Detailed immutable method outputs in `OATM/artifacts/`
- Compact result tables in `OATM/results/`
- Charts showing benefits and harms together
- `OATM/results/mvp_report.md`
- `OATM/results/run_metadata.json`
- A one-command cached reproduction entry point

Required plots include, where the sample supports them:

- Hidden recall versus ghost duration.
- Identity preservation versus wrong-object association.
- Localization error versus gap duration.
- Risk-coverage or calibration behavior across persistence thresholds.

The report must include failures, limitations, sample sizes, configuration,
run ID, commit, and manifest identities. It must not claim that mini proves
general autonomous-driving performance.

### Mandatory mentor checkpoint

Stop after the MVP report. The mentor must decide whether the evidence justifies
adding appearance memory, visual ego-motion, both, or neither.

**Task 11 is complete when:** one command can regenerate compact reports from
immutable outputs and every reported number is traceable to its source run.

---

# Part E — Optional Components That Must Earn Their Place

## Task 12 — Add Clear-View Appearance Memory if Approved

Only begin this task after the Task 11 mentor checkpoint.

The question for this task is:

> Does a frozen clear-view appearance anchor reconnect the correct identity
> better than motion alone?

Use a frozen embedding model. Update the appearance anchor only when a track is
clearly visible and eligible. Freeze it during occlusion so the occluder's
appearance cannot overwrite the target memory.

Association may combine appearance, class, predicted location, scale, and
motion direction. Add hard-negative tests with nearby same-class objects.

Run an ablation comparing motion only, appearance only if meaningful, and dual
memory. Preserve a null or harmful result.

**Task 12 is complete when:** occluded frames cannot update anchors, association
is deterministic, identity switches are explicit, and the ablation shows
whether appearance earned its added complexity.

---

## Task 13 — Add Camera-Derived Ego-Motion if Approved

Only begin this task after the simpler motion model and MVP have passed.

The question for this task is:

> Does estimating background image motion from camera frames reduce prediction
> error caused by movement of the vehicle itself?

Estimate motion causally from consecutive images using robust feature matching
or optical flow. Exclude tracked foreground regions where practical. If visual
motion estimation fails, report low confidence instead of silently applying an
unreliable correction.

Compare:

1. No ego-motion compensation.
2. Camera-derived visual compensation.
3. `oracle_ego_pose`, using recorded pose only as a clearly labeled diagnostic.

The oracle variant must never be reported as the camera-only headline result.

**Task 13 is complete when:** the three variants are clearly separated, primary
inference never reads recorded pose metadata, and the ablation states when
visual compensation helps or fails.

---

# Part F — Freeze, Scale, and Report

## Task 14 — Freeze the Method and Scale Beyond Mini

Do not begin this task until Phases 0–4 and the OATM MVP work correctly on mini.

Claude must:

1. Freeze event definitions, thresholds, schemas, and evaluation code.
2. Audit the nuScenes trainval installation without modifying source data.
3. Create scene-disjoint research splits without using the label-hidden
   official test set for tuning.
4. Repeat natural-event review on a larger subset.
5. Reuse the same code and schemas; scaling should change configuration, not
   algorithm logic.
6. Use cached or scheduled GPU inference where available, while keeping a CPU
   smoke test.
7. Record every scene, configuration, package version, seed, run ID, and commit.

Do not open final evaluation results until thresholds and decisions are frozen.

**Task 14 is complete when:** mini and trainval use one implementation, all
splits are scene-disjoint, and every final event and run is traceable.

---

## Task 15 — Run Ablations and Write the Final Scientific Report

The final study should compare the approved methods and these relevant
ablations:

- Motion only.
- Appearance only, if implemented.
- Dual memory, if implemented.
- No ego-motion compensation.
- Visual compensation, if implemented.
- Oracle-pose diagnostic, if run.
- No explicit occlusion gate.
- Fixed lifetime.
- No anti-ghost logic.
- Complete OATM.

Create:

- `OATM/results/final_report.md`
- Final compact tables and charts
- Failure-case visualizations
- `OATM/results/student_reflection.md`
- Updated reproduction instructions

The report must answer:

1. Which component improved hidden-object recall?
2. What did that improvement cost in ghost duration or false association?
3. Which occlusion lengths and motion types were recoverable?
4. When did motion prediction drift?
5. Did the state machine distinguish occlusion from exit and ordinary miss?
6. Did appearance or visual ego-motion justify its complexity?
7. How well calibrated was existence confidence?
8. Which conclusions hold for natural events, controlled visual events, and
   detector interventions separately?
9. What cannot be concluded from the sample size and dataset?
10. What is the next defensible research step?

The student reflection must be written by the student in her own words. Claude
may ask questions and check clarity, but must not invent her experience or
opinions.

**Task 15 is complete when:** the final report connects every claim to evidence,
reports benefits and harms together, and includes negative results and limits.

---

## Task 16 — Validate, Document, and Save the Work Safely

Before any final commit, Claude must:

1. Run all unit, integration, and regression tests.
2. Reproduce compact outputs from documented commands.
3. Confirm all required reports, tables, charts, configs, and manifests exist.
4. Validate schemas, finite coordinates, score ranges, chronological ordering,
   scene boundaries, and identifier uniqueness.
5. Recalculate summary values independently from detailed outputs.
6. Confirm no evaluation-scene tuning occurred.
7. Confirm the three experiment families remain separate.
8. Confirm online trackers cannot read privileged ground truth or future data.
9. Confirm predictions are never labeled as detections.
10. Confirm ghost duration and identity harms accompany recovery metrics.
11. Update `LOG.md` with the question, change, reason, validation, result,
    decision, and next step.
12. Inspect Git status and show the student the exact proposed commit list.

Commit only reviewable source code, tests, configuration, compact manifests,
small result tables, charts, and Markdown reports.

Do not commit:

- The nuScenes dataset.
- Raw copied images or large generated clips.
- Model weights.
- Detector caches.
- Virtual environments.
- Private workbooks.
- Credentials or authentication data.
- Temporary or operating-system files.

If pushing requires authentication, Claude must use the official private sign-in
flow. The student must never give Claude a password, verification code,
recovery code, or two-factor authentication code.

**Task 16 is complete when:** another researcher can understand the experiment,
reproduce the compact outputs, trace every result, and see the project history
without receiving private data.

---

# Required Final Deliverables

The exact implementation may improve as the project develops, but completion
requires the following categories:

```text
OATM/
  README.md
  METHODOLOGY.md
  IMPLEMENTATION_PLAN.md
  STUDENT_IMPLEMENTATION_ASSIGNMENT.md
  pyproject.toml
  configs/
  src/oatm/
  scripts/
  tests/
  results/
    README.md
    project_map.md
    dataset_audit_summary.md
    projection_audit.md
    natural_event_manifest.csv
    natural_event_selection.md
    detector_audit.md
    reuse_audit.md
    controlled_event_manifest.csv
    controlled_protocol.md
    mvp_report.md
    run_metadata.json
    final_report.md
    student_reflection.md
  artifacts/                  # local-only large or regenerable outputs
```

If a file is intentionally local-only, the reports must record its path,
purpose, size or row count, schema, hash where useful, and reproduction command.

# Rules Claude Must Follow

1. Work on only one numbered task at a time.
2. Explain the purpose before implementation and the evidence afterward.
3. Never ask the student to write or paste code or terminal commands.
4. Preserve completed assignments and original nuScenes data.
5. Treat `OATM/METHODOLOGY.md` as the scientific source of truth.
6. Keep online inference camera-only and causal.
7. Keep raw detections, tracker outputs, and offline ground truth separate.
8. Never use ground-truth identity to assign an online track ID.
9. Never call low confidence proof of occlusion.
10. Never call a prediction a current detection.
11. Keep natural occlusion, controlled visual occlusion, and detector
    intervention results separate.
12. Split by scene before selecting events or thresholds.
13. Tune only on development or validation scenes.
14. Report ghost tracks, identity errors, and runtime alongside recovery.
15. Use events or tracks as the main experimental unit.
16. Include sample counts and failure cases.
17. Add tests before trusting a new scientific component.
18. Record configurations, seeds, versions, run IDs, manifests, and commits.
19. Stop at failed gates and mentor checkpoints.
20. Never invent student review, measurements, or conclusions.

# What Success Looks Like

Success does not require OATM to beat every baseline.

Success means:

- The student can explain what each component does and why it was added.
- The data foundation and projection are verified before tracking begins.
- Every comparison method receives identical camera and detector evidence.
- OATM clearly marks observed, weakly observed, predicted, lost, and exited
  states.
- Motion uncertainty and confidence behave predictably through missing frames.
- The method improves a useful recovery-versus-risk tradeoff, or a trustworthy
  negative result explains why it does not.
- Natural and controlled evidence support appropriately limited claims.
- No privileged information leaks into the camera-only method.
- Every important result can be reproduced and traced to code, configuration,
  data manifest, and run metadata.
- The student finishes able to describe both the promise and the safety risk of
  temporal object persistence.

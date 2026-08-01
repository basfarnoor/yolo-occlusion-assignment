# Reuse Audit: What Assignment 4 Takes From Assignment 3

This audits every component of `assignments/03_sort_paper_experiment/experiment/`
for reuse in the ByteTrack experiment. It is grounded in
[`../../03_sort_paper_experiment/results/mentor_analytical_study.md`](../../03_sort_paper_experiment/results/mentor_analytical_study.md),
which independently reviewed Assignment 3 and found real methodological problems.
Assignment 4's own instructions require repairing eight of those problems rather
than carrying them forward silently — each decision below states which repair
item(s) it addresses.

## Component-by-component decision

### `src/geometry.py` — box math (center, area, IoU, center error, state conversion)

- **What it does:** Pure, stateless bounding-box arithmetic. No tracker state, no
  randomness.
- **Decision: REUSE unchanged.** Copied verbatim into
  `assignments/04_bytetrack_paper_experiment/experiment/src/geometry.py`.
- **Tests that protect it:** `test_geometry.py` — identical/disjoint-box IoU,
  center-error of a known shift, state round-trip. All reusable as-is.
- **Scientific risk remaining:** None identified. This module has no coupling to
  SORT's specific tracking logic.

### `src/kalman_box_tracker.py` — constant-velocity Kalman filter

- **What it does:** A from-scratch 7-state Kalman filter (`cx, cy, s, r, vx, vy,
  vs`) implementing predict/update for one box.
- **Decision: REUSE the filter structure, REPAIR the time step.** The current
  `predict()` hardcodes `F[0,4] = 1.0` etc., i.e. it always advances the state as
  if exactly one time unit has passed, regardless of the real interval between
  frames. Assignment 4's Task 6 requires `predict(dt)` to scale the
  velocity contribution by the *actual* timestamp difference between
  consecutive frames (available in the clip manifest), matching the paper's
  online, causal frame-by-frame operation more faithfully. **Addresses required
  repair #7** ("real timestamp differences should be used in motion prediction
  when practical").
- **Tests that protect it:** `test_kalman_prediction.py` (moving box keeps
  moving; a new observation corrects an imperfect prediction) — reusable
  patterns, but a new test must confirm `predict(dt)` scales displacement
  proportionally to `dt` (required test #15, "timestamp-aware prediction
  behaves correctly for unequal frame intervals").
- **Scientific risk remaining:** The constant-velocity assumption itself is
  unchanged from SORT — this is expected and matches the paper; ByteTrack's
  contribution is the association logic, not a new motion model.

### `src/assignment.py` — IoU cost matrix + Hungarian matching

- **What it does:** Builds an IoU cost matrix restricted to same-class pairs
  above a threshold and solves it with `scipy.optimize.linear_sum_assignment`.
- **Decision: REUSE unchanged**, called **twice per frame** — once for the
  high-confidence first association, once for the low-confidence second
  association (with its own configurable IoU threshold). The function is
  already generic over its input list, so no code change is needed to support
  BYTE's two-round structure.
- **Tests that protect it:** `test_assignment.py` (no double-assignment, class
  mismatch, sub-threshold rejection) — directly reusable, plus new tests for
  ordering (required tests #5-6: high-score matching happens first; only tracks
  still unmatched after round 1 enter round 2).
- **Scientific risk remaining:** None new. Same well-tested primitive as
  Assignment 3.

### `src/sort_tracker.py` — real track lifecycle (predict/associate/correct/birth/death)

- **What it does:** The actual online tracker: predicts every track, associates
  detections via `assignment.py`, corrects matches, births new tracks from
  unmatched detections, and — critically — **removes tracks whose
  `time_since_update` exceeds `max_age`** every frame, unconditionally.
- **Decision: REUSE this lifecycle as the foundation, EXTEND into a two-stage
  `ByteTrackTracker`.** This is the one piece of Assignment 3 the mentor did
  *not* flag as broken — its single-stage lifecycle is correct and already
  enforces real track expiry. The flaw was that `baselines.py` (see below)
  bypassed this class entirely for the long-gap evaluation. Assignment 4's
  tracker will call this same real `.update()`-per-frame lifecycle for every
  method (YOLO-only, high-confidence SORT, ByteTrack), extended with:
  - A second association stage against low-score detections before track
    removal.
  - An `evidence_source` field on every output row
    (`high_score_detection` / `low_score_detection` / `motion_prediction` /
    none), which Assignment 3's `TrackOutput` did not have.
  - `dt`-aware prediction (see above).
  **Addresses required repairs #2** (real lifecycle/expiry enforced for every
  gap length) **and #4** (prediction coverage reflects real track output/expiry,
  not a guaranteed box).
- **Tests that protect it:** `test_track_lifecycle.py` (track survives within
  `max_age`, expires beyond it, deterministic replay) — directly reusable
  template; new tests must additionally confirm the second-stage-specific
  invariants (required tests #7-9: a plausible low-score box can update a track,
  an unmatched low-score box never creates one, an unmatched high-score box
  still can).
- **Scientific risk remaining:** Two-stage association is new code, not covered
  by Assignment 3's tests at all — it needs its own full test file
  (`test_bytetrack_association.py` or similar) before any results are trusted.

### `src/detector_cache.py` — content-hashed YOLO detection cache

- **What it does:** Keys cached detections by image SHA-256 + model name +
  weights hash + image size + confidence + package versions; skips re-running
  YOLO on a cache hit.
- **Decision: REUSE unchanged.** This exactly matches Task 5's requirement to
  "reuse Assignment 3 cached detections only when [every key field] matches
  exactly. Run YOLO only for new or changed frames." No modification needed —
  the cache key is already the right granularity.
- **Tests that protect it:** Assignment 3 has no dedicated cache test (a gap).
  Assignment 4 should add one: same key round-trips detections; a changed
  confidence floor produces a cache miss.
- **Scientific risk remaining:** None to the caching mechanism itself. The risk
  is upstream — Assignment 4 needs a **lower detection floor (≤0.05)** than
  Assignment 3 used for its final experiments, so cached entries must be
  re-validated against the new floor, not blindly assumed reusable just because
  the image hash matches.

### `src/track_selection.py` — deterministic natural-track discovery + eligibility

- **What it does:** Runs a lightweight internal `SortTracker(max_age=1)` pass to
  link raw detections into candidate track segments, then filters them by class,
  length, confidence, displacement, and edge-margin rules, with logged,
  deterministic relaxation and seeded capping when too many/few segments
  qualify.
- **Decision: REUSE the *pattern* (deterministic relaxation logging + seeded
  sampling + full audit trail), REPLACE the eligibility rule itself.**
  Assignment 3's rules select "a long, confident, moving car" — the right
  criterion for testing motion prediction. Assignment 4's Task 9 needs a
  different criterion entirely: a track whose *confidence* crosses from the
  high group into the low group (or briefly produces no match) while the same
  physical instance persists on both sides, per independent nuScenes identity
  (`instance_token`), not just per-tracker continuity. That is a new selection
  function, built the same disciplined way (deterministic, logged, capped at
  12 events) but against different evidence.
- **Tests that protect it:** New tests are required; Assignment 3's eligibility
  tests do not transfer since the rule itself is different.
- **Scientific risk remaining:** Needs its own validation that "confidence drop"
  candidates are not simply track-ID churn from the internal linking pass, which
  is why the assignment requires a small human-inspected contact sheet
  (Task 9) before accepting any natural event.

### `src/baselines.py` (`run_three_baselines`) — the artificial-gap evaluator

- **What it does:** For a hand-picked pre-gap history, constructs a single
  `KalmanBoxTracker` directly (not through `SortTracker`), feeds it real boxes
  up to the gap, then calls `.predict()` in a loop with **no `.update()` calls
  and no `max_age` check** for every withheld frame — regardless of gap length.
- **Decision: REPLACE. Do not carry this pattern into Assignment 4.** This is
  the component the mentor's review identified as the most serious issue:
  1. The "withheld" reference boxes it evaluates against
     (`NaturalTrack.boxes`, populated from `TrackOutput.box`) are themselves
     **Kalman-corrected SORT output**, not raw YOLO boxes — a circular
     reference. Assignment 4 fixes this by evaluating against **independent
     nuScenes-projected ground truth** (Task 4) instead of any tracker's own
     output.
  2. Predicting in a bare loop with no `max_age` enforcement let gap lengths of
     5, 8, and 10 produce a box even though the configured `max_age: 3` tracker
     would have deleted that track long before. Assignment 4 instead removes
     detections (or demotes their score) from the tracker's real per-frame
     `.update()` input and lets the actual `ByteTrackTracker` lifecycle decide
     whether the track survives — including possibly not surviving.
  3. ID continuity in the calling code (`run_experiment.py`, see below) was
     hardcoded to "before equals after" by construction, never actually
     measured from tracker output.
  **Addresses required repairs #1, #2, and #3.**
- **Tests that protect it:** `test_baselines.py` proves the three methods see
  *identical* input, which is a good property to keep — Assignment 4's
  `controlled_trials.py` replacement must satisfy the same "identical raw
  detections across methods" invariant (required test #12), just implemented
  through real `.update()` calls instead of a bypassed prediction loop.
- **Scientific risk remaining:** None once replaced as specified — the whole
  point of the replacement is to remove this component's risks.

### `src/evaluation.py` (`group_summary`) — grouped summary statistics

- **What it does:** Groups trial rows by arbitrary keys and reports mean,
  median, std, and **explicit sample count** per group — never hides `n`.
- **Decision: REUSE the aggregation pattern, EXTEND the metric set and the
  grouping unit.** The "always show n, never hide sample count" discipline is
  worth keeping exactly. But Assignment 3's `n` was a row count that silently
  mixed nested, non-independent gap-frame observations from the same few
  tracks (mentor issue: "783 rows are not 783 independent experiments").
  Assignment 4 must additionally aggregate one level up — by track/event first,
  across tracks second — and report both the row count and the distinct
  track/event count side by side, never only the larger, more impressive-looking
  number. **Addresses required repair #6** ("repeated frame rows must not be
  presented as independent object samples") and required repair #8 (state
  scene/clip/track/event/frame/row counts separately).
- **Tests that protect it:** None in Assignment 3 (a gap); Assignment 4 should
  add a test confirming grouped stats never hide `n` and that track-level and
  row-level counts are reported as distinct fields.
- **Scientific risk remaining:** New metrics (IDF1/ID-switch count,
  fragmentation, ghost-track count/duration) are not simple means and need
  their own careful, tested implementations — `group_summary`'s pattern extends
  to them but does not implement them.

### `src/visualization.py` — drawing helpers for comparison videos

- **What it does:** Dashed/dotted rectangle drawing, box labeling, and
  side-by-side panel/grid composition, with a fixed color-per-method scheme.
- **Decision: REUSE the drawing primitives, EXTEND the color/label scheme.**
  The dashed-vs-solid-vs-dotted rectangle distinction (prediction vs.
  detection vs. reference) is exactly the visual grammar Assignment 4 also
  needs, just with more colors (high-confidence green, low-confidence yellow,
  SORT blue, ByteTrack-high-score cyan, ByteTrack-low-score orange,
  motion-only dashed purple, offline reference magenta) and an explicit
  evidence-source string on every label, per Task 13's spec.
- **Tests that protect it:** None (visual code is not unit-tested in Assignment
  3 either); acceptable, since correctness here is checked by the required
  human visual inspection of contact sheets/videos, not automated tests.
- **Scientific risk remaining:** None beyond correctly wiring the new
  evidence-source labels through from the tracker output.

### `src/clip_builder.py` — chronological clip construction from `sample_data` prev/next chain

- **What it does:** Discovers the local nuScenes root, walks the `prev`/`next`
  chain around a manifest-anchored frame, trims to a frame budget, copies
  frames, and validates strictly-increasing timestamps and file existence.
- **Decision: REUSE the chain-walking and validation logic, REPAIR for
  scene-disjoint splitting.** The temporal-ordering and validation logic is
  sound and directly reusable. It is missing one thing Assignment 4 requires:
  it never records which nuScenes **scene** a clip belongs to, so there is no
  way to guarantee a development/evaluation split shares no scene. Assignment 4
  adds scene-token lookup (via `sample.json` → `scene_token`) to the manifest
  and a fourth clip drawn from a scene distinct from the three reused ones.
- **Tests that protect it:** Assignment 3 has no dedicated clip-builder test (a
  gap, since correctness was checked via the generated `data_check.md`
  instead). Assignment 4 should add an explicit test that no scene token
  appears in both the development and evaluation split.
- **Scientific risk remaining:** None to the chain-walking itself; the risk was
  purely the missing scene-disjointness guarantee, now repaired.

### Orchestration scripts (`run_experiment.py`, `compute_results.py`, `build_charts.py`, `build_videos.py`, `compute_ablation.py`, `run_detect.py`, `build_contact_sheet.py`)

- **What they do:** Task-specific glue: load config/detections, run the
  experiment, compute summaries, draw charts/videos, run the ablation.
- **Decision: REUSE the overall shape (config-driven, cache-first, one script
  per pipeline stage), REPLACE the trial-generation logic inside them.** The
  separation into small, rerunnable stages is good engineering practice worth
  keeping. The actual experiment logic each script drives (`run_three_baselines`
  and the hardcoded-ID bookkeeping in `run_experiment.py`) is the part being
  replaced per the `baselines.py` decision above.
- **Scientific risk remaining:** None beyond what's already noted for the
  components they call.

### `config.yaml` — versioned experiment settings

- **What it does:** Detector, tracker, and occlusion-experiment settings in one
  YAML file.
- **Decision: REUSE the pattern, EXTEND the schema.** Adds
  `detection_floor`, `high_score_threshold`, `new_track_threshold`, a
  second-association IoU threshold, and `track_buffer`, all frozen on
  development scenes before any evaluation-split result is opened (Task 5's
  explicit requirement).
- **Scientific risk remaining:** None; this is a plain settings file.

### Test suite as a whole (`tests/`)

- **Decision: REUSE the pytest conventions (one file per module, `conftest.py`
  adding `src/` to `sys.path`), ADD the ByteTrack-specific tests required by
  Task 7** (17 tests plus an integration fixture) that have no Assignment 3
  analog: score-boundary splitting, two-stage ordering, low-score-cannot-birth,
  scene isolation, timestamp-aware `dt`, and single evidence-source-per-output.

## How this audit satisfies Assignment 4's required repair list

| # | Required repair | Where it's addressed above |
|---|---|---|
| 1 | Tracker-smoothed boxes must not be called raw withheld YOLO boxes | `baselines.py` replaced; evaluation reference becomes nuScenes-projected ground truth (Task 4), not tracker output |
| 2 | Long gaps must use the real tracker lifecycle and enforce `max_age`/expiry | `sort_tracker.py` extended into `ByteTrackTracker`, run via real per-frame `.update()`; `baselines.py`'s bypass removed |
| 3 | Identity continuity measured from actual output IDs, not assigned by construction | Hardcoded ID bookkeeping in `run_experiment.py` removed; real tracker output IDs used |
| 4 | Prediction coverage must reflect real track output and expiry | Same fix as #2 — coverage can now genuinely be less than 100% |
| 5 | Each method must be timed separately | New requirement for the Assignment 4 trial runner; Assignment 3's shared single timer is not reused |
| 6 | Repeated frame rows must not be presented as independent samples | `evaluation.py`'s `group_summary` pattern extended to report track/event counts alongside row counts |
| 7 | Real timestamp differences used in motion prediction when practical | `kalman_box_tracker.py`'s `predict()` repaired to take `dt` |
| 8 | State scene/clip/track/event/frame/repeated-row counts separately | `clip_builder.py` repaired to record scene tokens; `evaluation.py` extended to report every count level |

**Task 2 is complete when:** the table above shows every reused-or-replaced
component has a written scientific reason, and no Assignment 3 weakness is
carried forward without an explicit fix.

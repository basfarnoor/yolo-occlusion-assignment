# Reuse Audit: Baselines (Task 6)

Audits every component considered for reuse from Assignments 3-4 before
building the four OATM baselines (YOLO-only, static memory, SORT, ByteTrack)
behind one common interface. Assignment 4's own
[`reuse_audit.md`](../../assignments/04_bytetrack_paper_experiment/results/reuse_audit.md)
already repaired Assignment 3's methodological flaws (circular reference
boxes, bypassed track expiry, hardcoded ID continuity) -- this audit builds on
that already-repaired code, not on Assignment 3's original version.

## Component-by-component decision

### `geometry.py` (box math: IoU, center, area, state conversion)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/geometry.py`
- **Decision: REUSE unchanged.** Pure, stateless arithmetic with no tracker
  coupling. Copied verbatim into `src/oatm/tracking/geometry.py`.
- **Tests:** Assignment 4's `test_geometry.py` patterns reused directly.
- **Remaining risk:** none identified.

### `assignment.py` (IoU cost matrix + Hungarian matching)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/assignment.py`
- **Decision: REUSE unchanged.** Already generic over its inputs; ByteTrack's
  two-stage call pattern (same function, twice per frame, different detection
  subsets) carries over directly.
- **Tests:** no-double-assignment, class-mismatch, sub-threshold rejection
  (from Assignment 4) plus a new required test for the exact IoU=1/IoU=0 cases
  this task lists explicitly.
- **Remaining risk:** none new.

### `kalman_box_tracker.py` (timestamp-aware constant-velocity filter)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/kalman_box_tracker.py`
- **Decision: REUSE unchanged.** This already has the `predict(dt)` repair
  from Assignment 4 (Assignment 3's version incorrectly assumed exactly one
  frame per step regardless of real elapsed time). Directly satisfies this
  task's "timestamp-aware Kalman prediction" requirement with no further work.
- **Tests:** Assignment 4's dt-scaling test reused; this task's required
  synthetic motion regimes (stationary, smooth, slow, unequal gaps, turning,
  abrupt, missing-then-reappearing) are the Task 8 motion-memory test suite,
  not repeated here -- Task 6 only needs the tracker to run correctly, Task 8
  characterizes its motion-regime behavior.
- **Remaining risk:** the covariance trace (`self.P`) is exposed here for the
  first time as `localization_uncertainty` in `TrackerOutputRecord` output --
  new wiring, covered by a new test confirming it grows during missing frames.

### `track.py` (per-frame output shape, `evidence_source`)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/track.py`
- **Decision: REPLACE with `oatm.records.TrackerOutputRecord`.** Assignment
  4's lightweight `TrackOutput` only had `evidence_source` +
  `raw_detection_box`; OATM's canonical contract
  (`IMPLEMENTATION_PLAN.md` section 3) additionally requires
  `existence_confidence`, `identity_confidence`, `localization_uncertainty`,
  and `memory_age_frames/seconds` on every row, for every method -- so results
  are comparable across baselines and the eventual OATM method without a
  schema translation step. Baselines that do not model a concept report an
  honest, documented default rather than a fabricated sophisticated value:
  - `existence_confidence`: `1.0` while a track has any output at all (none of
    the four baselines implement adaptive existence decay -- that is OATM's
    own contribution, Task 10).
  - `identity_confidence`: `1.0` for all four baselines (none model identity
    confidence explicitly).
  - `localization_uncertainty`: `0.0` for YOLO-only (a current detection, not
    a prediction); a real Kalman covariance-trace value for SORT/ByteTrack;
    elapsed-time-since-update for static memory (a crude proxy, since it has
    no real state-estimation uncertainty).
- **Tests:** new -- confirms `evidence_source`/state combinations, and that
  baseline confidence fields are always their documented constant.
- **Remaining risk:** a reader could mistake a baseline's `1.0`
  existence/identity confidence for a real calibrated estimate; the report
  format must always state that baselines don't model these fields.

### `sort_tracker.py` (single-association-round tracker)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/sort_tracker.py`
- **Decision: REUSE the lifecycle, ADAPT the output.** The predict/associate/
  correct/birth/death loop and the "receives the same raw detections as
  ByteTrack, filters internally" design are unchanged. Only the return type
  changes (`TrackerOutputRecord` instead of the lightweight `TrackOutput`).
- **Tests:** Assignment 4's lifecycle tests (survives within buffer, expires
  beyond it, low-score never matched) reused; new tests add scene-boundary
  track ID isolation and the exact IoU=1/IoU=0 cases.
- **Remaining risk:** none new.

### `bytetrack_tracker.py` (two-stage BYTE association)

- **Source:** `assignments/04_bytetrack_paper_experiment/experiment/src/bytetrack_tracker.py`
- **Decision: REUSE the two-stage logic, ADAPT the output.** Same repair
  history as `sort_tracker.py` above -- including the `raw_detection_box`
  fix (the exact bug this task's "ground-truth tables inaccessible" and
  "predictions never called detections" spirit protects against; see
  Assignment 4's reuse_audit.md for the original discovery).
- **Tests:** Assignment 4's full association suite (stage ordering,
  low-score-cannot-birth, real-ID reconnection, single evidence-source label)
  reused; new tests add scene-boundary isolation.
- **Remaining risk:** none new.

### Static last-seen memory (Assignment 2)

- **Source:** Assignment 2 was a standalone script
  (`assignments/02_last_seen_memory/scripts/last_seen_memory.py`), not a
  reusable class -- it froze a box without any tracker-interface shape.
- **Decision: REPLACE with a new, small `StaticMemoryTracker`** behind the
  same interface as the other three baselines (so all four can be driven by
  one runner over identical frames/detections). Keeps Assignment 2's central
  idea (freeze the last matched box exactly, never move it) but adds the real
  track lifecycle (birth/expiry) Assignment 2 never had, so it can be
  fairly compared on the same buffer/expiry terms as SORT and ByteTrack.
- **Tests:** new -- box never moves during a gap, track still expires after
  the configured buffer, matches a returning detection via IoU like the
  others (not by construction).
- **Remaining risk:** none -- this is intentionally the simplest baseline.

### YOLO-only (Assignment 1)

- **Source:** Assignment 1 had no tracker at all by design.
- **Decision: REUSE the pattern from Assignment 4's `run_yolo_only`** (report
  accepted detections frame-by-frame, `track_id` carries no real identity).
  Adapted to emit `TrackerOutputRecord` with `state=OBSERVED_STRONG`,
  `existence_confidence=1.0`, `identity_confidence=1.0` (there is no real
  identity to be confident about -- documented, not glossed over), and
  `localization_uncertainty=0.0`.
- **Tests:** new -- confirms no track ever persists past a single frame.
- **Remaining risk:** none.

## Shared requirements verified across all four baselines

- All four receive the exact same ordered frames and the exact same raw,
  unfiltered per-frame detection list (proven by an automated test using the
  literal same Python list object, following Assignment 4's pattern).
- A fresh tracker instance is created per scene; no track ID can span two
  scenes (tested).
- No baseline's `update()` method accepts a ground-truth parameter, and none
  import the projection/ground-truth modules (tested, following Assignment
  4's structural-inaccessibility test pattern).
- Reappearance always goes through real IoU-based association; no baseline
  special-cases reappearance using stored ground-truth identity.

**Task 6 is complete when:** all four baselines run from one configuration,
receive identical evidence, and pass both the synthetic unit tests and a
real one-scene integration test -- verified below.

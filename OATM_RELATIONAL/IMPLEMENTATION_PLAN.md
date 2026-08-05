# Implementation Plan

## Foundation

- Relational state and scoring.
- Camera-motion estimation and fallback.
- Clearance prediction and relation-aware termination.
- Third-stage reappearance association.
- Synthetic comparisons, ablations, tests, and poster charts.

## Real-data milestone

- Regenerate detector/projection artifacts locally.
- Hash and reuse reviewed manifests with provenance.
- Add verified exit, miss, and failed-reappearance negatives.
- Compare ByteTrack buffers 3/5/8/12, Selective OATM, relational ablations,
  and the complete method on identical inputs.

## Completion status (2026-08-05)

- Completed: isolated `uv` workspace, relational state machine, occluder-centric
  geometry anchor, clearance termination, protected reappearance association,
  camera ablation, 20 tests, synthetic ablations, nuScenes preparation, cold
  detector rerun, natural pilot, metadata, compact tables, and poster charts.
- Rejected from promotion: camera compensation, after two real pilots showed
  severe drift.
- Not completed: a statistically powered scene-disjoint real evaluation,
  controlled-visual rerun, and human-verified real negative-event set.
- Fixed: anchor-consistency and reappearance bounds reduced natural localization
  error from 387.890 px to 16.019 px while raising coverage to 0.620.
- Current boundary: better than OATM_UPDATED on the two linkable events, but
  below ByteTrack-12 coverage and not statistically powered.
- Next research step: build verified negatives and controlled-visual events,

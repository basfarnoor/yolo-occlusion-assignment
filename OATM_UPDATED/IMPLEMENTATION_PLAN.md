# Implementation Plan

## Completed foundation

- Isolated `uv` project with locked dependency intent.
- Selective gate built on audited ByteTrack/Kalman/association components.
- Deterministic synthetic occlusion, miss, and exit study.
- Unit and integration tests.

## Next real-data milestone

Experiment question: does Selective OATM outperform tuned ByteTrack on
pixel-modified controlled visual occlusions at matched ghost duration?

Required checks:

- Regenerate frame index, projected ground truth, and detector cache.
- Freeze scene split and thresholds before evaluation.
- Run ByteTrack buffer sweep and Selective OATM gate/termination sweep on
  development scenes.
- Report controlled visual results separately from detector interventions.
- Add verified exit/loss negatives and wrong-association counts.

## Subsequent ablations

1. Occlusion gate removed.
2. Occluder-conditioned termination removed.
3. Camera-motion compensation removed.
4. Fixed lifetime substituted for selective lifetime.
5. Optional ReID appearance gate, only after a suitable embedding passes
   held-out identity tests.

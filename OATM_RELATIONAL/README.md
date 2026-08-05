# OATM Relational

Third-generation OATM research workspace. It extends the selective persistence
idea with explicit target--occluder relations, causal camera-motion
compensation, expected-clearance prediction, and relation-aware reappearance.

Status: the architecture and experiment pipeline are implemented. The final
synthetic study passes its mechanism checks. The two-linkable-event nuScenes
pilot improves coverage and identity recovery over OATM_UPDATED with comparable
localization, but still trails ByteTrack-12 coverage. Review the stored reports
before making broader comparative claims.

See `REPORT_PRESENTATION_GUIDE.md` for the canonical student-facing report,
slides, and poster reference. See `METHODOLOGY.md` for claim boundaries and
`PIPELINE.md` for reproduction.

```bash
uv sync
uv run pytest
uv run python scripts/run_relational_study.py
uv run python scripts/prepare_nuscenes.py
uv run python scripts/run_detector.py
uv run python scripts/run_natural_study.py --methods bytetrack_b5 bytetrack_b12 selective_oatm relational_complete --output-stem natural_promoted
```

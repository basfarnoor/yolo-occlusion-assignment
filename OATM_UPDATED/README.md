# OATM Updated

This workspace tests a narrower and more falsifiable OATM design:

> Keep ByteTrack unchanged while visual evidence exists. Extend a missing
> track only when causal camera evidence supports a real occlusion, then
> terminate it when that relationship stops being plausible.

The scientific source of truth is `METHODOLOGY.md`; the executable flow is in
`PIPELINE.md`. Large or regenerable outputs belong in `artifacts/` and compact
reports belong in `results/`.

## Reproduce

```bash
cd OATM_UPDATED
uv sync
uv run pytest
uv run python scripts/run_selective_study.py
```

The first checked-in experiment is a deterministic synthetic development
study. It verifies behavior before detector noise and ambiguous natural-event
labels are introduced. It is not evidence of real-world superiority.

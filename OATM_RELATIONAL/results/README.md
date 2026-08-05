# Results

Commit compact reports, CSV summaries, JSON metadata, and poster charts here.
Keep row-level outputs, detector caches, generated images, and model weights in
the ignored `artifacts/` directory. Never merge synthetic, controlled visual,
detector intervention, natural, or negative-event metrics into one score.

Key outputs:

- `synthetic_report.md`: final mechanism study, run `eac923a94d04`.
- `natural_promoted_report.md`: final promoted real-data pilot, run
  `806945a64e0d`; it improves over OATM_UPDATED but not ByteTrack-12 coverage.
- `natural_report.md`: retained camera-ablation evidence, run `6c0cec3cd955`.
- `detector_audit.md`: frozen detector provenance and exact observation count.
- `nuscenes_preparation.md`: read-only data/projection audit.
- `charts/`: compact presentation figures; use with the evidence caveats above.

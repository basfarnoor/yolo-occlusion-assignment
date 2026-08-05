# Pipeline

1. `uv sync --locked`: reproduce dependencies.
2. `uv run pytest`: validate geometry, relations, causality, lifecycle, and
   scene isolation.
3. `uv run python scripts/run_relational_study.py`: run deterministic synthetic
   development scenarios, ablations, compact metrics, metadata, and charts.
4. `uv run python scripts/prepare_nuscenes.py`: regenerate the read-only
   nuScenes frame index and privileged evaluation projections in `artifacts/`.
5. `uv run python scripts/run_detector.py`: regenerate camera-only detector
   observations. Model weights and row-level detections remain ignored.
6. Run the promoted comparison with:

   ```bash
   uv run python scripts/run_natural_study.py \
     --methods bytetrack_b5 bytetrack_b12 selective_oatm relational_complete \
     --output-stem natural_promoted
   ```

7. Evaluate controlled visual, detector intervention, natural events, and
   verified exits separately on frozen scene splits.
8. Select thresholds on development scenes only, freeze them, and write all
   material outcomes to root `LOG.md`.
9. Run the wider LiDAR-supported, `CAM_FRONT`-only mini evaluation with the
   existing uv environment:

   ```bash
   sbatch lidar_eval/submit_a100.sbatch
   ```

   This job runs causal tracking before loading projected annotations, scores
   only official keyframes, separates observed from predicted-hidden outputs,
   uses deterministic scene-disjoint development/validation populations, and
   records matching-threshold and annotation-quality sensitivity. See
   `lidar_eval/README.md` for cache reuse, optional A100 detector refresh, output
   schemas, and cluster-specific Slurm adjustments.

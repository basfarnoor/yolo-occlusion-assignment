"""Four comparison baselines behind one shared interface: every tracker class
exposes `update(detections, timestamp) -> list[TrackerOutputRecord]`, and
every one receives identical ordered frames and identical raw, unfiltered
per-frame detections (see reuse_audit.md)."""

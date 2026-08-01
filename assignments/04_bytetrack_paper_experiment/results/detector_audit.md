# Detector Audit

Model: `yolo26n.pt` (pretrained nano, prediction only -- no training/fine-tuning).
Device: `cpu`. Confidence floor (detection_floor): **0.05**.
Benchmark average 0.14s/frame at imgsz=640 -- within budget, keeping imgsz=640.

## Cache reuse check (Task 5 / reuse_audit.md)

Assignment 3's detection cache was not found in this checkout -- nothing to reuse; running fresh.

- Frames total: 144
- Cache hits this run: 144 (of which seeded from Assignment 3: 0)
- Cache misses (YOLO actually invoked): 0
- Total raw detections written: 3251

Every detection row carries its cache key, the raw YOLO box, class, confidence, frame identity, and inference time -- the raw box is never replaced with a tracker-corrected box anywhere downstream.

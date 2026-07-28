# Student Conclusion: YOLO Occlusion Sensitivity Experiment

These are the student's own observations from reviewing the annotated images and the analysis in `results/final_report.md`.

**1. What happened to YOLO's confidence when the object became partially hidden?**

Confidence decreased as the object became partially hidden. This matches the measured numbers: mean confidence dropped from 0.77 (no occlusion) to 0.59 (first partial occlusion) before collapsing to 0.03 once fully occluded.

**2. At which stage did YOLO most often lose the target?**

During full occlusion. The detector kept the target through the partial-occlusion stage but lost it almost every time once it was completely hidden (only 1 of 8 reviewed targets was still detected at full occlusion).

**3. Did YOLO immediately detect the target when it started to reappear?**

Mostly, but not always. Most reviewed samples were re-detected right away, but `sample_012` was still missed at "First Partial Appearance" and only picked back up at "Full Appearance" — so immediate re-detection was common but not guaranteed.

**4. Describe one surprising failure.**

In samples 011 and 012, the target object was given three separate bounding boxes around when it first reappeared and when it was fully visible again, instead of a single clean box — the detector split one real object into multiple overlapping detections at those reappearance stages.

**5. Why can a single-image detector struggle with occlusion?**

Single-frame detectors don't take past frames into consideration, which gives them no memory — and that lack of memory is what makes them struggle with occlusion. Without remembering what was there a moment ago, once an object's pixels are covered, the detector has nothing left to work with in that frame.

**6. What information from previous frames might help?**

Knowing the object's velocity (speed and direction) from earlier frames could let a system predict roughly where the hidden object is heading and where it should reappear, instead of treating it as a brand-new object once it becomes visible again.

---

*This is exactly the idea behind the bigger OATM research project described in `PROJECT_EXPLAINED_SIMPLY.md` and [`OATM/METHODOLOGY.md`](../../../OATM/METHODOLOGY.md) — this small experiment is direct evidence for why that memory-based approach is worth building.*

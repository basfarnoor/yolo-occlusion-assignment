# OATM Presentation Script (full version, ~7:04)

Your wording throughout, with every slide's narration matching what's
actually shown on it — including the two brand-new slides (Dataset,
Evaluation Methodology) and the additions that fill gaps between your
script and the actual slide content. New/expanded passages are marked
**[new]** / **[added]**. Timestamps assume ~130-140 wpm.

---

### HOOK (0:00 – 0:41)

> Good morning everyone.
>
> Imagine you're walking across a busy street. As you cross, a delivery
> truck passes in front of you, blocking your view of the traffic. For a few
> seconds, you can't see the cars on the other side. But you don't assume
> they've disappeared. You know they're still there, still moving, even
> though they're temporarily hidden from view.
>
> Humans do this effortlessly. We naturally maintain a mental model of our
> surroundings, allowing us to reason about objects even when we can't see
> them.
>
> But can autonomous vehicles do the same?

*[Advance straight into the title slide — no pivot slide exists anymore.]*

---

### SLIDE 1 — Title (0:43 – 0:57)

> This is OATM—Occlusion-Adaptive Temporal Memory. It's a framework designed
> to give a self-driving car's perception system the same kind of
> short-term memory humans naturally use when objects become temporarily
> hidden.

---

### SLIDE 2 — The Problem (0:57 – 1:20)

> Here's the actual problem.
>
> Most object detectors process each camera frame independently, with no
> memory of previous frames.
>
> In this example, the car is clearly visible, then becomes completely
> hidden behind traffic before reappearing farther down the road.
>
> To a single-frame detector, that middle frame isn't "a hidden car." It's
> simply an empty road.

---

### SLIDE 3 — The Dataset **[new]** (1:20 – 1:58)

> Before going further, a quick note on data and scope. We use nuScenes — a
> large-scale autonomous vehicle dataset with synchronized cameras, LiDAR,
> radar, and 3D box annotations that include a visibility rating for every
> object. This project only uses the front camera — one monocular view, no
> LiDAR or other cameras feeding detection. That keeps things laptop-scale
> and isolates the occlusion-memory problem on its own. The LiDAR and 3D
> annotations aren't wasted, though — they come back later as independent
> ground truth for evaluating OATM.

---

### SLIDE 4 — Research Question & Hypothesis (1:58 – 2:17)

> So our research question is:
>
> Can short-term visual memory allow a camera to continue tracking an object
> while it's temporarily hidden, without introducing false detections?
>
> Our hypothesis is that an occlusion-aware memory system can preserve
> object tracks through short occlusions while avoiding ghost detections.

---

### SLIDE 5 — Evidence 1: Baseline (2:17 – 2:59)

> First, we established the baseline using a pretrained YOLO detector with
> no temporal memory. Running on a CPU, we evaluated eight real occlusion
> sequences from the nuScenes dataset, following one object through five
> stages: visible, partially occluded, fully occluded, reappearing, and
> fully visible again.
>
> The performance drops off a cliff. Detection remains at 100 percent during
> partial occlusion but falls to just 12 percent once the object becomes
> fully hidden. Confidence also drops sharply—from 0.77 to 0.03—before
> recovering when the object becomes visible again. The detector doesn't
> gradually lose the object. It either detects it, or it doesn't.

---

### SLIDE 6 — Evidence 2: Naive Memory (2:59 – 3:40)

> A natural question is: why not simply keep the last bounding box while the
> object is hidden? We tested that idea. In the best case, the prediction
> was only 83 pixels away from the correct position. In the worst case, it
> was off by 589 pixels because the vehicle kept moving while it was hidden.
>
> **[added]** And this wasn't an isolated case. Across all five sequences we
> tested, the average localization error was 331 pixels, and three out of
> five predictions had zero overlap with the ground truth. Freezing the
> last position prevents disappearance—but it doesn't produce an accurate
> prediction.

---

### SLIDE 7 — Evidence 3: SORT Motion Prediction (3:40 – 4:19)

> We also tested a SORT-style tracker, which predicts an object's motion
> instead of freezing its position.
>
> Across nine vehicle tracks, the average localization error dropped from
> 46 pixels to 17 pixels, and overlap with the ground truth improved.
>
> However, these results have important limitations. Most of the
> improvement came from just two tracks, the reference boxes were generated
> using the tracker's own motion model, and the longest occlusions bypassed
> the tracker's expiration rule.
>
> So this shows that motion prediction helps—but it isn't yet an
> independent validation of SORT.

---

### SLIDE 8 — Diagnosis & The Gap (4:19 – 4:47)

> So here's the diagnosis.
>
> Static memory fails because it ignores motion.
>
> SORT predicts motion—as we just saw, that helps—but it doesn't explain
> why an object disappeared. ByteTrack improves data association but still
> relies on visible detections.
>
> Neither distinguishes whether an object is occluded, lost, or has exited
> the scene while adjusting confidence accordingly. That's the gap this
> project is designed to address.

---

### SLIDE 9 — OATM Design (4:47 – 5:30)

> OATM addresses that gap by combining two kinds of memory. **[expanded]**
> Appearance memory stores the last clear view of the object, so it can be
> re-identified once it reappears. Motion memory uses the same predict,
> observe, and correct loop we already tested in the SORT experiment—but
> corrected to use an independent reference and to properly enforce track
> expiration, the two issues that experiment exposed.
>
> Each object is also classified as visible, occluded, lost, or exited, and
> confidence changes according to that state rather than a fixed timer.
> Tracks are removed before they become ghost detections.

---

### SLIDE 10 — Evaluation Methodology **[new]** (5:30 – 6:12)

> So how will we actually know if OATM works? We'll use nuScenes' 3D box
> annotations and LiDAR point cloud as ground truth, projected into the
> front camera view — never derived from the camera's own detections. OATM
> runs exactly as designed, genuinely blind to the object during occlusion,
> and we score its predicted position against that independent LiDAR-based
> ground truth, not against another camera-based tracker. This directly
> fixes the circularity we ran into with SORT, where the reference boxes
> came from the same model being tested. The front camera can't see through
> the occlusion—but LiDAR still can, and that's what makes this an
> independent check.

---

### SLIDE 11 — How OATM Compares (6:12 – 6:25)

> Compared with existing approaches, that's the key idea behind OATM. To be
> clear, OATM has not yet been implemented or benchmarked. These are design
> claims that still require experimental validation.

---

### SLIDE 12 — Roadmap & Close (6:25 – 7:04)

> The next step is to improve the SORT experiment's methodology, build
> dedicated occlusion benchmark datasets, implement OATM, and compare it
> against existing tracking methods. We'll evaluate recall, identity
> preservation, ghost detection rate, and recovery time while separating
> scenes to prevent data leakage. **[added]** Longer term, we want to extend
> this from vehicles to pedestrians and other vulnerable road users.
>
> If OATM performs as expected, it could make autonomous vehicle perception
> more robust when important objects temporarily disappear from view.
>
> Thank you, and I'd be happy to answer any questions.

---

### SLIDE 13 — References

*No narration — this is a Q&A backup slide. Only flip to it if a judge asks
where the related-work numbers come from.*

---

## Notes for practice

- Runs ~7:04 at a relaxed pace — this is the full version, matching
  everything that's actually shown on each slide, including the two new
  slides and the additions that were previously missing.
- If you need to trim toward 5:00, see the earlier discussion: Slide 3
  (Dataset) and the last sentence of Slide 10 (Evaluation Methodology) are
  the least costly cuts; the rest is what a sharp judge is most likely to
  ask about if it's missing, so I'd resist cutting those first.
- Every number traces to `results/final_report.md`,
  `results/last_seen_memory/summary.csv`, and
  `results/sort_paper_experiment/summary.csv`. The 7 papers behind Slide 13
  are listed in full on that slide.

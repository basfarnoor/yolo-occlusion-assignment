# Assignment 2: Give YOLO a Last-Seen Memory

> Give this entire file to Claude Code.
>
> You do not need to write Python, use the terminal, edit spreadsheets, or type Git commands. Claude must do all programming and setup. Your job is to choose the target object, inspect the pictures, and explain what you observe.

## The Idea

In the first experiment, YOLO usually stopped reporting an object when it became fully hidden.

This experiment asks:

> **Can a simple memory stop YOLO from immediately forgetting a hidden object?**

The rule is:

1. When YOLO sees the target, save its most recent bounding box.
2. When YOLO loses the target, reuse that saved box.
3. Clearly label the reused box as a memory prediction.
4. When the target reappears, compare the old box with its new location.

The saved box does not move. This is intentionally simple.

## What You Should Learn

By the end, you should understand:

- The difference between seeing an object and remembering it.
- Why memory can preserve an object's existence.
- Why an unchanged old location becomes inaccurate.
- Why memory must expire to prevent ghost objects.
- Why motion prediction would be a useful next improvement.

## Hypothesis

Before beginning, write down this hypothesis:

> Last-seen memory will stop the target from immediately disappearing, but the saved box will become less accurate when the target or camera moves.

---

## Task 1 — Choose Clear Targets

Claude must read:

- `results/student_review.xlsx`
- `results/all_detections.csv`
- `occluded_samples/manifest.csv`
- The images in `occluded_samples`

Claude must preserve these files.

### What Claude creates

Claude must create:

`results/last_seen_memory/last_seen_experiment.xlsx`

The first worksheet should be named `Target Selection`.

For each sample, Claude must also create two numbered images:

1. The last usable image before full occlusion.
2. The first usable image after reappearance.

Each YOLO box should have a large box number, class, and confidence.

Save these images under:

`results/last_seen_memory/selection_images/`

### What you do

Choose one target per sample and enter:

- A precise description, such as `white truck behind the red van`.
- Its box number before occlusion.
- Its box number after reappearance.
- Whether both boxes show the same physical object.
- Whether the object was hidden rather than leaving the camera view.

Use only **3 to 5 clear samples**. Quality matters more than quantity.

Reject a sample if:

- You cannot identify the same object before and after.
- The object changes between stages.
- The object leaves the image instead of becoming hidden.
- The supposed full-occlusion image still clearly shows the target.

Inspect `sample_011` and `sample_012` especially carefully because their first review contained inconsistent entries.

**Task 1 is complete when:** 3 to 5 valid samples have one clearly identified target each.

---

## Task 2 — Add Last-Seen Memory

Claude must create and run the Python program. You must not write code.

For each selected target:

### When YOLO sees the target

- Draw the current box in **green**.
- Label it `CURRENT YOLO DETECTION`.
- Save its coordinates as the last-seen box.

### When YOLO loses the target

- Copy the last-seen box without moving or resizing it.
- Draw it in **orange with a dashed outline**.
- Label it `MEMORY — NOT CURRENTLY DETECTED`.
- Display `memory age: 1 stage` or `memory age: 2 stages`.

Memory must expire after two missing stages.

Count only available images in their time order. A stage that has no image file does not increase memory age.

The program must not add:

- Motion prediction.
- A Kalman filter.
- Object tracking software.
- Model training.
- Appearance matching.

This task tests only unchanged last-seen memory.

**Task 2 is complete when:** every selected sample uses exactly the same memory rule.

---

## Task 3 — Make Two Visual Comparisons

Claude must save the comparisons under:

`results/last_seen_memory/comparisons/`

### Comparison 1: During occlusion

Show the full-occlusion image twice:

- **Left — YOLO only:** no target box if YOLO lost it.
- **Right — YOLO plus memory:** the orange last-seen box.

The image must clearly state:

> The orange box is a prediction from memory, not current camera evidence.

### Comparison 2: At reappearance

On the first usable reappearance image, show:

- The old memory box in orange.
- The new YOLO target box in green.
- A line between their center points.
- Center error in pixels.
- Box overlap, called `IoU`.

Add this explanation:

- `IoU near 1` means the boxes overlap well.
- `IoU near 0` means the remembered location became stale.

**Task 3 is complete when:** each valid sample has both comparison images.

---

## Task 4 — Review the Results

Claude must add a worksheet named `Results` to:

`results/last_seen_memory/last_seen_experiment.xlsx`

Required columns:

- Sample
- Target description
- Target class
- Last visible stage
- First reappearance stage
- Memory age in stages
- Previous YOLO confidence
- Reappearance YOLO confidence
- Center error in pixels
- Center error as percentage of image width
- IoU
- Your judgement
- Could this become a ghost object? Yes or No
- Notes

For `Your judgement`, choose:

- `Helpful`
- `Partly helpful`
- `Misleading`

Use these meanings:

- **Helpful:** the memory box remained close to the reappearing object.
- **Partly helpful:** it remembered the object, but the location was noticeably stale.
- **Misleading:** it pointed to a substantially wrong location.

Claude must calculate the numerical columns. You provide only the human judgement and notes.

**Task 4 is complete when:** every valid sample has measurements and your judgement.

---

## Task 5 — Ask Claude to Analyze the Experiment

When the workbook is finished, tell Claude:

> I finished reviewing the last-seen results. Validate the workbook and produce the final analysis.

Claude must create:

- `results/last_seen_memory/center_error_by_sample.png`
- `results/last_seen_memory/iou_by_sample.png`
- `results/last_seen_memory/summary.csv`
- `results/last_seen_memory/final_report.md`

The report must include:

- Number of valid and rejected samples.
- Why samples were rejected.
- Mean and median center error.
- Mean and median IoU.
- Number judged helpful, partly helpful, and misleading.
- Number with possible ghost risk.

It must answer:

1. Did memory stop the target from immediately disappearing?
2. Did the old box remain close to the target?
3. Which sample had the best memory location?
4. Which had the worst?
5. When was memory helpful?
6. When was it misleading?
7. Why could memory create a ghost object?
8. Why should the next version predict movement?

The report must not call the memory box a real detection. It must not claim hidden-object accuracy because the true hidden location was not measured.

Every average must show how many samples were used.

---

## Task 6 — Write Your Conclusion

Answer in your own words:

1. What happened when YOLO lost the target?
2. What did the memory system add?
3. Was the old box still accurate when the object reappeared?
4. Describe one helpful example.
5. Describe one misleading example.
6. Why should memory expire?
7. What should be added next?

Claude may improve spelling and organization, but it must ask for your observations before drafting the conclusion.

The main lesson should be:

> **Remembering that an object exists is easy. Predicting where it moved is harder.**

---

## Task 7 — Commit and Push

Claude must perform all Git work.

Before committing, Claude must show you the planned file list.

Commit:

- The Python program.
- `summary.csv`.
- `final_report.md`.
- The two small charts.
- Your conclusion.

Do not commit:

- The Excel workbook.
- Numbered selection images.
- Comparison-image folders.
- Model weights.
- Python environments or caches.
- Passwords, tokens, or authentication files.

Claude must add ignore rules for those excluded files without deleting earlier results.

Claude must validate the files, create a clear commit, and push the branch to GitHub. If authentication is required, you should complete it privately in the official browser flow.

---

## Rules for Claude

1. The student performs no coding or terminal work.
2. Never guess which box is the selected target.
3. Preserve all first-experiment files.
4. Keep current detections green and memory predictions dashed orange.
5. Never describe memory as current camera evidence.
6. Use the same unchanged-box rule for every sample.
7. Exclude invalid samples with written reasons.
8. Do not add motion prediction or train a model.
9. Show sample sizes beside averages.
10. Do not commit generated image collections or the local workbook.
11. Check staged files for credentials and large data before pushing.

## Success Checklist

- [ ] 3 to 5 clear targets were selected.
- [ ] The same object was confirmed before and after occlusion.
- [ ] Invalid samples were rejected.
- [ ] Green boxes show current detections.
- [ ] Dashed orange boxes show memory predictions.
- [ ] Memory expires after two missing stages.
- [ ] Occlusion and reappearance comparisons were created.
- [ ] Center error and IoU were calculated.
- [ ] Helpful and misleading examples were identified.
- [ ] The report explains both memory's benefit and ghost risk.
- [ ] The student explains why motion prediction should come next.
- [ ] Reproducible code and small results were pushed.

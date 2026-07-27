# Assignment 3: Recreate the Main Idea of the SORT Paper

> This file is both your assignment and your prompt for Claude Code.
>
> Give Claude Code this entire file. You are **not expected to write Python,
> edit configuration files, use the terminal, label bounding boxes, or type Git
> commands**. Claude must do all coding, testing, execution, chart creation, and
> Git work. Your job is to understand the idea, watch the generated videos, and
> answer a few short questions in your own words.

## The Paper

This assignment is based on:

> Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, and Ben Upcroft,
> **“Simple Online and Realtime Tracking,”** ICIP 2016.

- [Read the SORT paper on arXiv](https://arxiv.org/abs/1602.00763)
- [Open the paper PDF](https://arxiv.org/pdf/1602.00763)
- [See the authors' official SORT repository](https://github.com/abewley/sort)
- DOI: [10.1109/ICIP.2016.7533003](https://doi.org/10.1109/ICIP.2016.7533003)

Throughout this assignment, **“the paper” means the SORT paper linked above**.
Claude must mention the paper in every generated explanation and report.

## Why This Paper Fits Your Previous Experiments

Your first experiment showed that YOLO usually loses a target during full
occlusion.

Your second experiment kept the last YOLO box on the screen. That remembered
that the object existed, but the box stayed frozen. The average center error was
about 331 pixels, and three of five remembered boxes had zero overlap with the
object when it reappeared.

The SORT paper provides a natural next idea:

> Instead of assuming that the missing object stayed still, estimate its motion
> and predict where its bounding box should move.

The paper calls this **state estimation** and uses a **Kalman filter**. It then
uses box overlap and the **Hungarian algorithm** to decide which new detection
belongs to which existing track.

Do not worry about those names yet. This assignment will make them visible in
small diagrams, tests, and videos.

## Important Scientific Description

This is a **small paper-inspired replication and extension**, not a complete
reproduction of every result in the SORT paper.

The paper evaluated multiple-object tracking on the MOT benchmark and used
different object detections. This assignment will:

- Use the local nuScenes mini data.
- Use the existing pretrained YOLO detector.
- Implement the central SORT ideas in educational Python.
- Compare SORT motion prediction with the static memory from Assignment 2.
- Use automatically hidden YOLO detections as temporary “pseudo-ground truth.”

Claude must use this exact description in the final report. It must never claim
that this assignment reproduced the paper's published benchmark scores.

---

## Objectives

By the end of the assignment, you should be able to:

1. Explain the difference between object detection and object tracking.
2. Explain why the SORT paper separates the detector from the tracker.
3. Explain a Kalman filter as “predict, observe, and correct.”
4. Explain why the paper uses IoU to compare bounding boxes.
5. Explain why an assignment algorithm is needed when several cars are visible.
6. Compare YOLO-only, static last-seen memory, and SORT motion memory.
7. Use an automatic artificial-occlusion test instead of manually labeling many
   images.
8. Read charts showing how prediction changes as an occlusion becomes longer.
9. Identify at least one limitation of the SORT paper's simple motion model.
10. Complete a paper-based Python experiment on an ordinary laptop by directing
    Claude Code.

## Research Question

> **Does the motion prediction used by SORT keep a missing object's bounding
> box closer to its later position than an unchanged last-seen box?**

## Hypothesis

Claude must place this hypothesis at the top of the generated report before
running the experiment:

> Based on the SORT paper, motion prediction should beat static last-seen memory
> for short detection gaps when an object moves smoothly. Its accuracy should
> decrease during longer gaps, sudden turns, camera motion, or incorrect
> detection-to-track matching.

A result that disagrees with this hypothesis is still a valid result.

---

# Instructions for Claude Code

## Your Role

Complete the assignment autonomously, one task at a time.

- Do all programming and terminal work yourself.
- Never ask the student to write or paste code.
- Never ask the student to number or label bounding boxes.
- Never ask the student to fill a results spreadsheet.
- Prefer automatic checks and saved visual evidence.
- Explain each paper concept using normal high-school language.
- Preserve all files from Assignments 1 and 2.
- Never modify or delete the local nuScenes source data.
- Never train or fine-tune YOLO.
- Use CPU unless a GPU is already available and requires no special setup.
- Stop and explain clearly if the local data does not match the expected
  structure. Do not invent results.

Create all new code under:

`sort_paper_experiment/`

Create all new outputs under:

`results/sort_paper_experiment/`

Do not place generated images, videos, caches, or environments in the Git
repository unless this assignment explicitly asks for them to be committed.

---

## Task 1 — Make a One-Page Map of the SORT Paper

Read at least the paper's abstract, introduction, method section, and conclusion.
Also inspect the authors' repository README.

Create:

`results/sort_paper_experiment/paper_map.md`

Explain these terms in one or two simple sentences each:

- Detection
- Track
- Track ID
- State
- Velocity
- Prediction
- Correction
- Kalman filter
- IoU
- Hungarian assignment
- Track birth
- Track death

Use this everyday analogy:

> If you briefly close your eyes while watching a moving car, you expect the car
> to continue moving. The prediction is your expectation. Opening your eyes
> gives you a new observation, which corrects that expectation.

Include a small pipeline:

```text
camera image
    -> YOLO detections
    -> predict old tracks
    -> match detections to tracks
    -> correct matched tracks
    -> create or remove tracks
    -> boxes with stable track IDs
```

For every pipeline step, state where the same idea appears in the SORT paper.
Do not copy long passages from the paper. Paraphrase them.

### Paper connection

The paper describes SORT as **tracking by detection**. YOLO provides
observations; SORT connects those observations over time.

**Task 1 is complete when:** a beginner can read `paper_map.md` and describe the
paper's overall idea without understanding its equations.

---

## Task 2 — Inspect the Data and Build Small Continuous Clips Automatically

Find the local nuScenes mini dataset. It is expected to contain folders similar
to:

```text
data/nuscenes/
  samples/
  sweeps/
  v1.0-mini/
```

The project may be inside another folder, so discover the dataset instead of
hardcoding an absolute path.

Read the nuScenes metadata, especially `sample_data.json`, and use the camera
frame `prev` and `next` links to create short, time-ordered CAM_FRONT clips.

Use known images from the previous assignments as anchors. Prefer:

- `sample_001`
- `sample_003`
- `sample_006`

Claude may replace an anchor only if its continuous camera frames are
unavailable. Record the reason.

Laptop limits:

- Use no more than **3 clips**.
- Use no more than **36 frames per clip**.
- Prefer approximately 2–4 seconds around each anchor.
- Copy or link only the required frames.
- Never load the full nuScenes dataset into memory.
- Never alter the original dataset.

Create:

- `results/sort_paper_experiment/clip_manifest.csv`
- `results/sort_paper_experiment/data_check.md`
- `results/sort_paper_experiment/clips/<clip_name>/`

The manifest must include:

- Clip name
- Frame number
- Timestamp
- Time since previous frame
- Original image path
- Experiment image path
- Whether the frame is a nuScenes keyframe
- Anchor sample

Check that timestamps increase and files exist. If continuous frames cannot be
created, stop before implementing the experiment and explain the exact problem.

### Paper connection

The paper is an **online** tracker: it receives frames in chronological order
and may use the past but not future frames. The clip builder must therefore
preserve time order.

**Task 2 is complete when:** up to three small, verified, chronological clips
exist without requiring the student to select individual frames.

---

## Task 3 — Run YOLO Once and Cache Every Detection

Use the same lightweight pretrained YOLO model as the earlier assignment when
possible. Run prediction only:

- CPU
- Image size 640
- Confidence threshold 0.05
- Batch size 1 or another memory-safe value
- No training
- No fine-tuning

If a quick five-frame benchmark averages more than 2.5 seconds per frame, reduce
the image size to 480 and record this change. Do not silently change settings.

Save every detection to:

`results/sort_paper_experiment/detections.csv`

Required columns:

- Clip
- Frame number
- Timestamp
- Class
- Confidence
- `x1`, `y1`, `x2`, `y2`
- Inference time in milliseconds

Also create a cache keyed by:

- Image SHA-256
- Model name
- Model weights hash when available
- Image size
- Confidence threshold
- Relevant package versions

If the cache matches, reuse it. Every later experiment must use the cache and
must not rerun YOLO unnecessarily.

Create a contact sheet showing several frames and their YOLO boxes. This is only
a visual sanity check; the student does not need to annotate it.

### Paper connection

The SORT paper emphasizes that tracking quality depends strongly on detection
quality. This cached YOLO file is the detector output that the tracker receives.
Caching also lets us change the tracking experiment without repeatedly paying
the expensive detection cost.

**Task 3 is complete when:** YOLO has been run at most once for each unique
configuration, its output is machine-readable, and the cache is validated.

---

## Task 4 — Implement the Educational SORT Tracker

Do not copy the authors' `sort.py` file. Implement a small educational version
from the ideas in the paper and cite the paper at the top of every main source
file.

Suggested structure:

```text
sort_paper_experiment/
  config.yaml
  run_experiment.py
  src/
    clip_builder.py
    detector_cache.py
    geometry.py
    kalman_box_tracker.py
    assignment.py
    sort_tracker.py
    artificial_occlusion.py
    evaluation.py
    visualization.py
    report.py
  tests/
    test_geometry.py
    test_kalman_prediction.py
    test_assignment.py
    test_track_lifecycle.py
```

Claude may improve this structure, but it must keep the parts understandable.

### Part A — Bounding-box geometry

Implement:

- Box center
- Width and height
- Box area
- IoU
- Center error
- Conversion between box coordinates and tracker state

### Part B — Kalman motion prediction

Implement a small Kalman filter with NumPy. Its state must contain bounding-box
position plus motion. It should follow the paper's constant-velocity idea.

Add comments explaining the loop:

```text
predict where the box should be
observe a new YOLO box if available
correct the prediction using that observation
```

Use real timestamp differences when practical. If fixed frame steps are used,
justify that choice in the final report.

### Part C — Detection-to-track assignment

Create an IoU cost matrix between predicted tracks and current detections.
Use SciPy's Hungarian assignment implementation, or an equivalently tested
implementation, to find the best one-to-one matching.

Only match compatible object classes. Use an explicit IoU threshold stored in
`config.yaml`.

### Part D — Track lifecycle

Implement:

- A unique track ID
- `age`
- `hits`
- `time_since_update`
- `min_hits`
- `max_age`
- Track creation for unmatched detections
- Track removal after too many missed frames

### Required automatic tests

At minimum, test that:

1. Identical boxes have IoU 1.
2. Non-overlapping boxes have IoU 0.
3. A box moving 10 pixels per step continues moving during a missing detection.
4. A new observation corrects an imperfect prediction.
5. Two detections cannot be assigned to the same track.
6. A track survives the allowed missing frames.
7. A track expires after `max_age`.
8. The same input and random seed produce the same output.

Do not continue if tests fail.

### Paper connection

These are the paper's central tracking components: Kalman state estimation,
IoU-based matching, Hungarian assignment, and track management. The assignment
uses a readable implementation so the student can connect each code module to
the paper.

**Task 4 is complete when:** all tests pass and `paper_map.md` links every
concept to the source file that implements it.

---

## Task 5 — Create Three Automatic Baselines

Every evaluation must compare:

### Baseline A — YOLO only

When the chosen detection is removed, YOLO-only has no target box. This baseline
measures detector coverage, not location error during the missing frames.

### Baseline B — Static last-seen memory

Reuse the last observed box without moving it. This is Assignment 2.

### Method C — SORT motion memory

Use the Kalman prediction from the paper-inspired tracker.

All three methods must receive the same detections, image order, and artificial
missing-frame pattern.

### Paper connection

The paper argues that a lightweight motion model and association step can turn
independent detections into tracks. The two baselines show what the paper's
tracking layer adds beyond individual YOLO outputs and frozen memory.

**Task 5 is complete when:** a unit test proves that all three methods are
evaluated on identical input trials.

---

## Task 6 — Make Occlusions Automatically

This task replaces boring manual annotation with a controlled experiment.

### Automatically choose eligible tracks

From the unmodified YOLO detections, automatically find moving car, truck, bus,
or pedestrian tracks that:

- Last for at least 12 frames.
- Have detections before and after the planned gap.
- Have reasonable confidence.
- Do not begin or end at the image boundary.
- Have a class that remains consistent.
- Move enough for static memory and motion prediction to be meaningfully
  different.

Select up to 10 eligible track segments. Use deterministic rules and seed 42.
Save the selection rules and reasons. Do not ask the student to pick tracks.

If fewer than three eligible tracks exist, automatically relax only the minimum
track length, one step at a time, down to 8 frames. Record each relaxation. Do
not relax identity or file-validity checks.

### Artificially hide detections

For each eligible track:

1. Keep the original YOLO boxes in a private evaluation table.
2. Remove that target's detections from the tracker input for a middle gap.
3. Test gap lengths of 1, 2, 3, and 5 frames when the track is long enough.
4. Run YOLO-only, static memory, and SORT on the modified input.
5. Compare their predictions with the withheld YOLO boxes.

The withheld YOLO boxes are **pseudo-ground truth**, not human ground truth.
Claude must use that exact term in every chart caption and report.

This setup answers a clean question: if YOLO had temporarily missed detections
that it originally produced, how well would each memory method bridge the gap?

### Paper connection

The SORT paper studies missed detections and short-term motion through track
prediction and `max_age`. Artificial gaps let us test that idea repeatedly
without asking the student to label hundreds of boxes.

**Task 6 is complete when:** the same deterministic experiment can be recreated
from the configuration and cached detections.

---

## Task 7 — Calculate Results Automatically

For every hidden frame, calculate:

- Whether the method produced a box
- Center error in pixels
- Center error as a percentage of image width
- IoU with the withheld YOLO box
- Whether IoU is at least 0.30
- Track ID before the gap
- Track ID after the gap
- Whether the ID remained continuous
- Prediction time in milliseconds

Save:

- `results/sort_paper_experiment/trials.csv`
- `results/sort_paper_experiment/summary.csv`
- `results/sort_paper_experiment/run_metadata.json`

The summary must group results by:

- Method
- Gap length
- Object class
- Clip

Report mean, median, standard deviation, and number of trials. Never hide the
sample count.

Measure tracker-only speed separately from YOLO-plus-tracker speed. The SORT
paper reports tracker speed separately, so the assignment should make the same
distinction.

### Required comparisons

Calculate:

1. SORT IoU minus static-memory IoU.
2. Percentage change in center error.
3. Prediction coverage during the gap.
4. ID continuity after reappearance.
5. How all metrics change from a one-frame gap to a five-frame gap.

If no method wins, report that honestly.

### Paper connection

The paper evaluates both tracking accuracy and speed. This smaller experiment
uses simpler local measurements because it does not have full manually verified
MOT ground truth.

**Task 7 is complete when:** every number in the report can be traced to a CSV
row and recreated by one command.

---

## Task 8 — Generate Charts and Videos

Create these charts automatically:

1. `mean_iou_by_gap.png`
2. `center_error_by_gap.png`
3. `prediction_coverage_by_gap.png`
4. `id_continuity_by_gap.png`
5. `runtime_comparison.png`

Each chart must:

- Compare all applicable methods.
- Show sample counts.
- Label units.
- Explain that the reference boxes are withheld YOLO detections.
- Include error bars or individual trial points when useful.
- Remain readable to a student who has never taken statistics.

Create at least three short side-by-side MP4 videos:

```text
YOLO only | static last-seen | SORT prediction | withheld YOLO reference
```

Use these colors consistently:

- YOLO observation: green
- Static memory: orange
- SORT prediction: blue
- Withheld reference: magenta

Every prediction must be labeled as a prediction, not a real detection.

Create one slow explanatory video that pauses or adds text when:

- A detection is removed.
- Static memory stays still.
- The SORT paper's motion model moves the predicted box.
- A new detection returns and corrects the track.

### Paper connection

The videos turn the paper's predict–associate–correct loop into something
visible. They should make the Kalman filter understandable without requiring the
student to read matrix equations.

**Task 8 is complete when:** the student can watch a video and point out which
box is observed, remembered, predicted, and withheld for evaluation.

---

## Task 9 — Run One Automatic Ablation

An **ablation** means changing one part of a method to see why it matters.

Without rerunning YOLO, compare at least:

- Static memory
- SORT with motion enabled
- SORT with velocity forced to zero

Optionally, if runtime remains small, also test two `max_age` values.

Do not change several settings at once.

Create:

- `results/sort_paper_experiment/ablation.csv`
- `results/sort_paper_experiment/ablation.png`
- A short ablation section in the final report

### Paper connection

The paper's tracker contains several simple components. This ablation isolates
whether motion prediction—not merely retaining a track ID—caused any
improvement.

**Task 9 is complete when:** the report explains what changed, what stayed
fixed, and what the result suggests.

---

## Task 10 — Write the Automatic Final Report

Generate:

`results/sort_paper_experiment/final_report.md`

Use this structure:

1. Paper citation and link
2. What the SORT paper proposed
3. How this small experiment relates to the paper
4. How it differs from a full paper reproduction
5. Hypothesis written before the run
6. Dataset and automatically selected clips
7. Detector settings and cache
8. Educational SORT implementation
9. Artificial-occlusion protocol
10. Baselines
11. Quantitative results
12. Ablation
13. Best and worst visual examples
14. Runtime and laptop suitability
15. Limitations
16. Conclusion
17. Exact reproduction command

Answer these questions directly:

1. Did SORT motion prediction beat static last-seen memory?
2. For which gap lengths did it help most?
3. When did the constant-velocity assumption fail?
4. Did track IDs remain stable after detections returned?
5. How much time belonged to YOLO, and how much belonged to SORT?
6. How do the findings connect to the claims and limitations of the paper?
7. What do the findings add to Assignments 1 and 2?

The limitations section must say:

- Withheld YOLO boxes are pseudo-ground truth, not manual ground truth.
- Artificially removing a detection is not identical to real visual occlusion.
- YOLO can produce inaccurate reference boxes.
- nuScenes includes camera motion, while a basic image-space constant-velocity
  model does not explicitly understand 3D ego-motion.
- The sample is small.
- This is not a reproduction of the paper's MOT benchmark results.

Include the paper citation:

```bibtex
@inproceedings{Bewley2016SORT,
  author    = {Alex Bewley and Zongyuan Ge and Lionel Ott and
               Fabio Ramos and Ben Upcroft},
  title     = {Simple Online and Realtime Tracking},
  booktitle = {2016 IEEE International Conference on Image Processing},
  year      = {2016},
  pages     = {3464--3468},
  doi       = {10.1109/ICIP.2016.7533003}
}
```

### Paper connection

The report must continually compare the implementation, measurements, and
limitations with the SORT paper. It must distinguish paper claims from this
student experiment's findings.

**Task 10 is complete when:** the report can be read without opening the code,
and every factual result is traceable to generated data.

---

## Task 11 — Let the Student Experiment Without Coding

After showing the main results, Claude must ask the student to choose **one**
plain-language experiment:

1. “Let memory survive longer.”
2. “Require boxes to overlap more before matching.”
3. “Test longer artificial occlusions.”

Claude must translate the choice into one configuration change, rerun only the
cheap cached tracking stage, and create:

`results/sort_paper_experiment/student_experiment.md`

That file must record:

- What the student chose
- Claude's configuration change
- The student's prediction before the run
- The result after the run
- Whether the prediction was supported
- How this relates to the SORT paper

The student must not edit the configuration or run a command herself.

**Task 11 is complete when:** the student has made one scientific choice and
observed its consequence without writing code.

---

## Task 12 — Short Student Reflection

Create:

`results/sort_paper_experiment/student_reflection.md`

Ask the student only these five questions, one at a time:

1. In your own words, what is the difference between YOLO and SORT?
2. Why did the static orange box become stale?
3. What did the blue SORT box do differently?
4. Describe one case where the paper's constant-velocity idea helped or failed.
5. What is one reason this experiment cannot prove that SORT always works?

Her answers may be one or two sentences each. Claude may fix spelling but must
not replace her meaning or invent an answer.

This is the only required manual results work.

### Paper connection

The reflection checks whether the student understood the paper's idea rather
than merely running its code.

**Task 12 is complete when:** the five short answers are saved in the student's
own words.

---

## Task 13 — Validate, Document, Commit, and Push

Before committing, Claude must:

1. Run all tests.
2. Run the experiment from a clean command using cached detections.
3. Confirm that every required CSV, chart, video, and report exists.
4. Scan CSV files for blank required fields, NaN, infinity, and impossible box
   coordinates.
5. Confirm that all reported sample counts match the CSV files.
6. Confirm that no generated chart contains a misleading axis.
7. Confirm that predictions are never labeled as detections.
8. Confirm that the paper is linked and cited.
9. Confirm that the limitations are present.
10. Confirm that the repository does not include:
    - The nuScenes dataset
    - Model weights
    - A Python environment
    - Detection cache files too large for GitHub
    - Temporary files
    - Excel lock files

Create or update `.gitignore` where necessary.

Commit:

- Source code
- Tests
- Small configuration files
- Paper map
- Final CSV summaries
- Charts
- Markdown reports
- Student reflection
- A small number of compressed demonstration videos only if each is reasonably
  sized for GitHub

Do not commit:

- The nuScenes data
- Full copied clips
- YOLO model weights
- Virtual environments
- Large caches
- Large videos

Use a clear commit message such as:

`add SORT paper motion-prediction experiment`

Push the completed branch to the student's GitHub repository. If authentication
is required, open the official browser-based flow and let the student complete
it privately. Never ask for a password or authentication code.

### Paper connection

The repository should make the small paper-inspired experiment understandable
and reproducible by another student.

**Task 13 is complete when:** the validated work is committed and pushed, and
Claude gives the student a short summary with links to the paper, report, best
video, charts, and GitHub commit.

---

# Required Final Deliverables

Claude must not call the assignment complete unless these exist:

```text
assignments/3_sort_paper_experiment.md

sort_paper_experiment/
  config.yaml
  run_experiment.py
  src/
  tests/

results/sort_paper_experiment/
  paper_map.md
  data_check.md
  clip_manifest.csv
  detections.csv
  trials.csv
  summary.csv
  run_metadata.json
  ablation.csv
  mean_iou_by_gap.png
  center_error_by_gap.png
  prediction_coverage_by_gap.png
  id_continuity_by_gap.png
  runtime_comparison.png
  ablation.png
  videos/
  final_report.md
  student_experiment.md
  student_reflection.md
```

Large local-only files may be excluded from Git, but their paths, sizes, hashes,
and reproduction instructions must be recorded.

# Laptop Safety Limits

Claude must obey these limits unless the student and mentor explicitly approve
a larger run:

- Maximum 3 clips
- Maximum 36 frames per clip
- Maximum 10 evaluated track segments
- YOLO nano model only
- No model training
- CPU-compatible execution
- One YOLO pass per configuration
- Cached detections for all tracker experiments
- No full-dataset scan beyond metadata and filename checks
- No large file committed to GitHub

If the run is still too slow, reduce the number of clips before weakening the
scientific comparisons. Record every reduction.

# What Success Looks Like

Success does **not** mean SORT must win.

Success means:

- The paper's central idea was implemented clearly.
- The experiment was automatic and repeatable.
- Three methods received identical inputs.
- Results were calculated without repetitive manual annotation.
- The student could see prediction and correction in a video.
- The conclusions matched the evidence.
- Limitations were stated honestly.
- The complete experiment ran on an ordinary laptop.

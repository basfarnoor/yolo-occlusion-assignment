# Assignment: Test How Occlusion Affects YOLO

> This file is both your assignment and your prompt for Claude Code.
>
> Open the project folder in Claude Code and give Claude this entire file. You are **not expected to write, edit, or copy any code or terminal commands yourself**. Claude must create the programs, run the commands, check the results, and explain any problem in simple language.

## Objectives

By the end of this assignment, you should be able to:

1. Explain in simple words what an object detector does.
2. Recognize a bounding box, object class, and confidence score.
3. Observe how partial and full occlusion affect a pretrained YOLO detector.
4. Organize selected nuScenes images into repeatable five-stage samples.
5. Collect experimental observations without training a model.
6. Compare detection rate and confidence across occlusion stages.
7. Identify interesting detector failures and suggest possible explanations.
8. Use GitHub to keep a safe history of the project.
9. Complete a small Python-based experiment by directing Claude Code rather than programming manually.

## What You Will Investigate

Your research question is:

> **How does increasing occlusion affect YOLO's ability to detect the same object?**

Each sample contains five stages:

1. Previous No Occlusion
2. First Partial Occlusion
3. Full Occlusion
4. First Partial Appearance
5. Full Appearance

The experiment will run a small pretrained YOLO model on every image. You will inspect the results and follow one target object through the five stages.

This is an **occlusion sensitivity experiment**. It is not yet a formal measurement of YOLO's overall accuracy or mAP because the dataset does not contain manually verified ground-truth boxes for every visible object.

---

## Task 0 — Create Your GitHub Account

This is the only task that Claude cannot complete for you because the account must belong to you.

1. Go to [GitHub Sign Up](https://github.com/signup).
2. Create your own account using an email address you can access.
3. Choose a professional username that you will be comfortable showing on school work.
4. Verify your email address.
5. Turn on two-factor authentication if GitHub asks you to do so.
6. Send only your GitHub username to your mentor.

Never give Claude, your mentor, or anyone else your GitHub password, email verification code, recovery codes, or two-factor authentication code. If GitHub authentication is required later, Claude should open the official browser-based sign-in flow and let you complete it privately.

**Task 0 is complete when:** you can sign in to GitHub and your email address is verified.

---

## Task 1 — Give This Assignment to Claude Code

1. Open the `frame-allocation` project folder in Claude Code.
2. Give Claude this entire Markdown file.
3. Say: **"Complete the assignment one task at a time. Do all coding and terminal work yourself. Explain what you are doing in simple language."**
4. Allow Claude to inspect the project.
5. Read Claude's short explanation before allowing it to continue to the next major task.

You do not need to create a Python file, edit this Markdown file, type terminal commands, or install packages manually. If Claude needs permission to install something or access a file, it must explain why and ask you to approve the action.

**Task 1 is complete when:** Claude has found the project folder, the `data` folder, and this assignment file.

---

## Task 2 — Place and Check the Excel Workbook

Your Excel workbook should contain one sample per row and columns with these meanings:

| Column | Meaning |
|---|---|
| Sample no. | The number or name of the sample |
| Previous No Occlusion | Image before the target starts becoming hidden |
| First Partial Occlusion | First image where the target is partly hidden |
| Full Occlusion | Image where the target is completely hidden |
| First Partial Appearance | First image where the target begins to return |
| Full Appearance | Image where the target is clearly visible again |

It is okay if the workbook currently uses the spellings `Apparence` or `Appearence`. Claude must recognize these as `Appearance` and must not require you to edit the original workbook.

Claude must search the project for an `.xlsx` or `.xls` workbook. If it cannot find one, it should ask you to drag or copy the workbook into:

`data/occlusion_samples.xlsx`

The `data` folder is local and intentionally excluded from GitHub. The workbook and nuScenes dataset must remain local.

Claude must inspect the workbook and report:

- The workbook filename.
- The worksheet it selected.
- The column names it found.
- The number of non-empty sample rows.
- Empty cells in the five image columns.
- Duplicate sample numbers.
- Values that do not look like image filenames.

Claude must not modify or replace the original workbook.

**Task 2 is complete when:** Claude can read the sample rows and clearly reports any missing or unusual values.

---

## Task 3 — Ask Claude to Organize `occluded_samples`

Claude must create and run a safe Python organizer. You do not write this program.

### Required source and destination

- Source dataset: the existing `data` folder, including the nuScenes camera folders.
- Excel input: the workbook found in Task 2.
- New organized output folder: `occluded_samples`

The source dataset and workbook are read-only inputs. Claude must **copy**, never move, rename, edit, or delete, the original images.

### Required output structure

Claude should create one folder per Excel row:

```text
occluded_samples/
  sample_001/
    1_previous_no_occlusion.jpg
    2_first_partial_occlusion.jpg
    3_full_occlusion.jpg
    4_first_partial_appearance.jpg
    5_full_appearance.jpg
  sample_002/
    ...
```

The actual image extension may remain `.jpg`, `.jpeg`, or `.png`, depending on the source file.

### How Claude must locate each image

For every non-empty Excel cell in the five image columns, Claude must:

1. Read the filename or path from the cell. If the cell is a hyperlink, check both its displayed text and hyperlink target.
2. First try an exact relative path if one is provided.
3. Otherwise search recursively under `data` for an exact filename.
4. If no exact filename exists, try an exact filename stem with a supported image extension.
5. Use a case-insensitive match only as a final fallback.
6. Never select a file using a vague partial-name match.

If there are no matches, Claude must record the image as `missing`.

If there is more than one possible match, Claude must record the image as `ambiguous` and show the possible paths. It must not silently choose one.

### Required safety behavior

The organizer must be safe to run more than once:

- If the destination file already contains the same image, skip it.
- If a different file already exists at the destination, do not overwrite it.
- Record conflicts in the report.
- Preserve the original image bytes and file extension.
- Do not delete old folders automatically.

### Required organization records

Claude must create:

1. `occluded_samples/manifest.csv`
2. `occluded_samples/organization_report.md`

The manifest must have at least these columns:

- `sample_number`
- `excel_row`
- `stage_number`
- `stage_name`
- `excel_value`
- `source_path`
- `destination_path`
- `status`
- `notes`

Allowed status values should include:

- `copied`
- `already_present`
- `missing`
- `ambiguous`
- `conflict`
- `empty_excel_cell`

The report must state:

- Number of workbook sample rows.
- Number of expected images.
- Number successfully organized.
- Number already present.
- Number missing.
- Number ambiguous.
- Number conflicting.
- The exact rows requiring human attention.

Claude must compare copied files with their sources using file size and SHA-256 hashes. A copied image is successful only when the source and destination match.

### Student checkpoint

Claude should pause after organizing the images and explain the report in simple language.

You should then:

1. Open two or three `sample_###` folders.
2. Confirm that the images appear in the correct five-stage order.
3. Tell Claude about any visibly incorrect sample.

You are inspecting images, not writing code.

**Task 3 is complete when:** every workbook cell is marked with a clear status, successful copies are verified, and unresolved rows are listed rather than guessed.

---

## Task 4 — Ask Claude to Prepare the Python YOLO Experiment

Claude must create the complete Python experiment and execute it. You must not be asked to write or edit Python.

Claude must:

1. Create an isolated Python environment inside the project.
2. Install the required packages in that environment.
3. Use the current Ultralytics Python package.
4. Load the pretrained nano detection model `yolo26n.pt`.
5. Use prediction only. Do not train or fine-tune a model.
6. Run on CPU so a dedicated GPU is not required.
7. Process only the organized images in `occluded_samples`.
8. Use an image size of 640.
9. Use a low confidence threshold of `0.05` so weak detections remain available for study.
10. Avoid loading all images into memory at once.

The first use may download the small pretrained model weights. Claude must explain this before the download and reuse the local weights afterward.

The official Ultralytics prediction documentation is available at:

- <https://docs.ultralytics.com/modes/predict>
- <https://docs.ultralytics.com/usage/python>

### Required outputs

Claude must create:

```text
results/
  annotated/
    sample_001/
    sample_002/
  all_detections.csv
  run_summary.md
```

Every annotated image must show YOLO's bounding boxes, class labels, and confidence values.

`all_detections.csv` must contain one row per detected box with at least:

- `sample_number`
- `stage_number`
- `stage_name`
- `image_filename`
- `detected_class`
- `confidence`
- `x1`
- `y1`
- `x2`
- `y2`
- `box_width`
- `box_height`
- `inference_time_ms`

Claude must also record an image-level row or separate table so that an image with zero detections is not accidentally omitted from the experiment.

The run summary must include:

- Model name.
- Package version.
- Python version.
- Device used.
- Number of images processed.
- Number of images that failed.
- Total runtime.
- Average runtime per image.
- Location of the annotated results.

Claude must not claim that the raw detection count is the number of correct detections.

**Task 4 is complete when:** every organized image has either an annotated result or a clearly reported processing error.

---

## Task 5 — Review One Target Object Per Sample

YOLO may detect several cars, pedestrians, or other objects in the same image. The computer does not yet know which object you chose when collecting the sample. For this first experiment, you will identify the target by looking at the images.

Claude must create `results/student_review.xlsx` with one row for every sample stage and these columns:

- Sample number
- Stage number
- Stage name
- Image filename
- Target description
- Expected target class
- Target detected: Yes or No
- Predicted class
- Correct class: Yes, No, or Not applicable
- Target confidence
- Failure type
- Student notes

The five rows belonging to a sample must stay together and in the correct order.

Claude should make the workbook easy to use, with clear headers, filters, frozen header rows, readable column widths, and dropdown choices where appropriate. It must not alter the raw YOLO output CSV.

### What you do manually

For each sample:

1. Choose the same target object across the five stages.
2. Describe it briefly, for example, `white car near the center`.
3. Open the annotated images.
4. Record whether YOLO placed a box on that target.
5. If it did, record its predicted class and confidence.
6. If it did not, enter `No` and use confidence `0`.
7. Add a short note if something interesting happened.

Use one of these failure types:

- `None`
- `Missed while visible`
- `Missed while partially visible`
- `Wrong class`
- `Box on the wrong object`
- `False target detection during full occlusion`
- `Unclear`

During full occlusion, `Target detected: No` is normally expected because the camera cannot see the target. A target-class box placed on the occluder or background is not successful object persistence; it may be a false detection.

You are entering observations in Excel, not programming.

**Task 5 is complete when:** every organized sample has five reviewed rows or a note explaining why it could not be reviewed.

---

## Task 6 — Ask Claude to Analyze the Completed Review

After completing the review workbook, tell Claude:

> **"I finished reviewing the target objects. Please run the analysis and explain the graphs."**

Claude must read the completed review workbook and create:

1. `results/analysis_summary.csv`
2. `results/detection_rate_by_stage.png`
3. `results/confidence_by_stage.png`
4. `results/correct_class_rate_by_stage.png`
5. `results/final_report.md`

### Required calculations

For each of the five stages, calculate:

- Number of reviewed target objects.
- Number detected.
- Target detection rate.
- Number assigned the correct class.
- Correct-class rate among visible reviewed targets.
- Mean target confidence, treating an undetected target as confidence `0`.
- Median target confidence.
- Number of each failure type.

Natural non-detection during full occlusion must be reported separately from failure to detect a visible or partially visible target.

Claude must keep the five stages in their scientific order rather than alphabetical order.

Claude must not invent values for incomplete review cells. It should report incomplete rows and exclude them from calculations that require those values.

### Questions the report must answer

1. Does target detection rate decrease from no occlusion to partial occlusion?
2. Does average confidence decrease as the object becomes hidden?
3. How often does the target become detectable during first partial appearance?
4. Does confidence return near its original value at full appearance?
5. Which failure type occurs most often?
6. Are any apparent detections during full occlusion actually boxes on the occluder or background?
7. What are three especially interesting samples to inspect?
8. What can and cannot be concluded from this small, manually selected dataset?

Claude should explain each graph in everyday language before using technical vocabulary.

**Task 6 is complete when:** the tables, graphs, and report agree with the reviewed workbook and incomplete data is clearly disclosed.

---

## Task 7 — Write Your Short Conclusion

Answer these questions in your own words. Claude may help you improve clarity, but it should ask you for your observations before drafting your conclusion.

1. What happened to YOLO's confidence when the object became partially hidden?
2. At which stage did YOLO most often lose the target?
3. Did YOLO immediately detect the target when it started to reappear?
4. Describe one surprising failure.
5. Why can a single-image detector struggle with occlusion?
6. What information from previous frames might help?

Your conclusion does not need to prove that YOLO is bad or good. A useful experiment reports what actually happened, including unexpected or mixed results.

**Task 7 is complete when:** you can explain the experiment and one important finding without reading the program code.

---

## Task 8 — Save the Work Safely with GitHub

Claude must handle Git commands. You must not be asked to type them.

Claude should:

1. Check the Git repository and current branch.
2. Confirm that local datasets, copied images, model weights, environments, and generated results will not be uploaded.
3. Add appropriate ignore rules for at least:
   - `data/`
   - `occluded_samples/`
   - `results/`
   - `.venv/`
   - Python cache files
   - downloaded model weight files
4. Show you the exact files it plans to commit.
5. Commit only source programs, documentation, and small configuration files.
6. Never commit the nuScenes dataset, Excel workbook, copied sample images, credentials, or account secrets.
7. Use a clear commit message.
8. Push the work to the GitHub repository after browser-based authentication is complete.
9. Report the branch name and commit identifier.

**Task 8 is complete when:** the reproducible project files are on GitHub and the local dataset remains private and untracked.

---

## Rules Claude Code Must Follow

The instructions below are addressed directly to Claude Code.

1. The student has no programming experience. Perform all coding, file creation, package installation, terminal execution, debugging, Git operations, and result generation yourself.
2. Never instruct the student to type or modify Python, shell commands, Git commands, configuration files, or Markdown.
3. Ask the student only for human decisions that cannot safely be inferred, such as locating a missing workbook, identifying the intended target object, resolving an ambiguous filename, approving a download, or completing private browser authentication.
4. Explain each major action using short, ordinary language.
5. Work through the numbered tasks in order and pause at the stated student checkpoints.
6. Inspect the real folders and workbook before writing the organizer. Do not assume the workbook's filename, worksheet name, exact spelling, or cell format.
7. Preserve the workbook and every source image. Copy files; never move or delete them.
8. Never silently guess when an Excel filename is missing or matches multiple images.
9. Make all created programs safe to rerun.
10. Validate outputs instead of assuming a successful command produced correct results.
11. Keep raw inputs, organized data, raw predictions, manual review, and processed analysis clearly separated.
12. Do not train a detector during this assignment.
13. Use CPU prediction and the nano model unless the mentor explicitly changes the scope.
14. Do not describe this exercise as formal mAP evaluation.
15. Never expose, print, store, or commit passwords, tokens, recovery codes, or private authentication information.
16. Do not upload the dataset, workbook, copied samples, results, or model weights to GitHub.
17. Before any Git commit, show the student the planned file list and confirm that it contains no dataset files.
18. If an action fails, diagnose and retry it yourself. Ask the student for help only when a private sign-in or genuinely missing human decision blocks progress.

---

## Final Success Checklist

- [ ] GitHub account created and email verified.
- [ ] Claude found and inspected the Excel workbook.
- [ ] Original workbook and nuScenes images remained unchanged.
- [ ] `occluded_samples` was created from the workbook.
- [ ] Every requested image has a recorded status.
- [ ] Missing and ambiguous filenames were reported instead of guessed.
- [ ] Copied images were verified against their sources.
- [ ] Pretrained nano YOLO ran on CPU.
- [ ] Annotated images and raw detections were saved.
- [ ] One target object per sample was reviewed.
- [ ] Detection-rate and confidence graphs were generated.
- [ ] The student wrote a short evidence-based conclusion.
- [ ] Code and documentation were committed to GitHub.
- [ ] Dataset files, workbook, copied images, results, weights, and secrets were not uploaded.

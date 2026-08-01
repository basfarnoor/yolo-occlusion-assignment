# Paper Map: Simple Online and Realtime Tracking (SORT)

**Paper:** Alex Bewley, Zongyuan Ge, Lionel Ott, Fabio Ramos, Ben Upcroft,
["Simple Online and Realtime Tracking,"](https://arxiv.org/abs/1602.00763)
ICIP 2016. DOI: [10.1109/ICIP.2016.7533003](https://doi.org/10.1109/ICIP.2016.7533003).
Authors' code: [github.com/abewley/sort](https://github.com/abewley/sort).

This file explains the paper's ideas in plain language before any code is written, and links every term to the source file that implements it (filled in as Task 4 progresses).

## The everyday analogy

> If you briefly close your eyes while watching a moving car, you expect the car to continue moving. The prediction is your expectation. Opening your eyes gives you a new observation, which corrects that expectation.

SORT automates exactly that loop: predict where a tracked object should be, then correct that guess whenever a new camera detection arrives.

## Terms, in plain language

**Detection** — one bounding box that an object detector (YOLO here) drew on a single frame, with a class and confidence. A detection has no memory of previous frames.

**Track** — a single physical object followed *across* frames. A track is built from a sequence of detections (and, when detections are missing, predictions) that the algorithm believes belong to the same object over time.

**Track ID** — the number attached to a track so the same object keeps the same label from frame to frame, instead of being treated as a brand-new object every time it's seen.

**State** — the numbers SORT keeps for each track to describe it right now: its box position, size, shape, and how fast each of those is changing. The paper represents a box as center coordinates, area (scale), and aspect ratio, plus the rate of change of each.

**Velocity** — the part of the state that says how quickly the box's position, size, and shape are currently changing per frame. This is what lets SORT guess *where the box is going*, not just where it last was.

**Prediction** — using the current state and velocity to estimate where the track's box should be in the next frame, before seeing any new detection. This is the "eyes closed, car keeps moving" step.

**Correction** — adjusting the predicted state using a new matching detection, blending the prediction with the fresh observation. This is the "eyes open" step.

**Kalman filter** — the specific mathematical tool SORT uses to do predict-then-correct in a principled way. It doesn't just overwrite the prediction with the observation; it combines them, weighted by how much the filter currently trusts each one.

**IoU (Intersection over Union)** — a single number from 0 to 1 that measures how much two boxes overlap: the area they share divided by the total area they cover together. 1 means identical boxes; 0 means no overlap at all. SORT uses IoU to decide how well a predicted box matches a real detection.

**Hungarian assignment** — an algorithm that solves the "who matches with whom" problem: given several predicted tracks and several new detections, it finds the one-to-one pairing that maximizes total overlap (or minimizes total cost), rather than greedily grabbing whichever match looks good first.

**Track birth** — creating a brand-new track when a detection can't be matched to any existing track (e.g. an object just entered the frame).

**Track death** — removing a track that hasn't been matched to a real detection for too many frames in a row (e.g. the object left the frame, or was a one-off false detection).

## The pipeline

```text
camera image
    -> YOLO detections
    -> predict old tracks
    -> match detections to tracks
    -> correct matched tracks
    -> create or remove tracks
    -> boxes with stable track IDs
```

| Pipeline step | Where it appears in the paper |
|---|---|
| YOLO detections | The paper calls this "tracking by detection" -- SORT itself does not detect objects; it consumes whatever a separate detector produces. Detection quality is explicitly identified as the biggest factor in tracking accuracy. |
| Predict old tracks | The Kalman filter's prediction step, using the paper's constant-velocity assumption for each track's state. |
| Match detections to tracks | The paper builds an IoU cost matrix between predicted boxes and new detections, then solves it with the Hungarian algorithm, rejecting matches below an IoU threshold. |
| Correct matched tracks | The Kalman filter's update/correction step for every successfully matched track. |
| Create or remove tracks | The paper's track management rules: unmatched detections start new tentative tracks (confirmed after a minimum number of hits); tracks with too many consecutive missed frames (`max_age`) are deleted. |
| Boxes with stable IDs | The final output SORT is built to produce: the same identity persisted across frames, at real-time speed (the paper reports 260 Hz on its benchmark hardware). |

## Why this fits after Assignments 1 and 2

- **Assignment 1** showed a plain YOLO detector loses a target completely during full occlusion -- no memory at all.
- **Assignment 2** added a memory box, but it never moved: average center error was about 331 pixels, and 3 of 5 remembered boxes had zero overlap with the real object once it reappeared.
- **This assignment** replaces "freeze the last box" with the paper's actual idea: predict *where the box should have moved*, using a Kalman filter, and use IoU + Hungarian matching to reconnect it with the right detection when the object reappears.

## Important scientific framing

This is a **small paper-inspired replication and extension**, not a full reproduction of the SORT paper's published MOT benchmark results. The paper evaluated on the MOT benchmark with its own detections; this assignment uses local nuScenes mini clips, the existing pretrained YOLO nano detector, and an educational Python reimplementation of the paper's central ideas, compared against Assignment 2's static memory and against automatically withheld YOLO detections used as **pseudo-ground truth** (not manually verified ground truth).

## Concept-to-code links

*(Filled in as Task 4 is implemented -- each concept above will link to its source file in `experiment/src/`.)*

| Concept | Source file |
|---|---|
| Box geometry, IoU, center error | `src/geometry.py` |
| Kalman filter (state, predict, correct) | `src/kalman_box_tracker.py` |
| IoU cost matrix + Hungarian assignment | `src/assignment.py` |
| Track lifecycle (birth, age, hits, death) | `src/sort_tracker.py` |
| Artificial occlusion / pseudo-ground truth | `src/artificial_occlusion.py` |
| Metrics (IoU, center error, ID continuity) | `src/evaluation.py` |
| Charts and videos | `src/visualization.py` |
| Report generation | `src/report.py` |

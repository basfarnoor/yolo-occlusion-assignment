# YOLO Occlusion Assignments

This repository contains three connected computer-vision assignments. Each
assignment keeps its instructions, code, results, and supporting material
together.

## Folder guide

- [`assignments/01_yolo_occlusion/`](assignments/01_yolo_occlusion/) — baseline
  YOLO occlusion experiment, organized samples, review material, and presentation.
- [`assignments/02_last_seen_memory/`](assignments/02_last_seen_memory/) — static
  last-seen memory experiment, including its scripts and results.
- [`assignments/03_sort_paper_experiment/`](assignments/03_sort_paper_experiment/)
  — SORT paper-inspired motion-tracking experiment, tests, reports, charts, and
  videos.

The assignments build on one another. Assignment 2 reads the detections and
organized samples from Assignment 1; Assignment 3 also uses Assignment 1
samples as anchors. The private nuScenes dataset remains in the repository-level
`data/` directory and is excluded from Git.

## Where to start

Open the `README.md` inside the assignment you want to review. Its scripts can
be run from that assignment folder, and generated outputs stay in that same
folder.

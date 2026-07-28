"""SORT paper reference: Bewley et al., "Simple Online and Realtime
Tracking," ICIP 2016 (https://arxiv.org/abs/1602.00763).

Task 8: side-by-side comparison videos (YOLO only | static memory | SORT
prediction | withheld reference) plus one slow explanatory video that
pauses and captions the predict-associate-correct loop.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from baselines import run_three_baselines  # noqa: E402
from track_selection import build_natural_tracks  # noqa: E402
from visualization import panel, compose_grid, GREEN, ORANGE, BLUE, MAGENTA, WHITE, BLACK  # noqa: E402

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "sort_paper_experiment"
VIDEOS_ROOT = OUT_ROOT / "videos"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_REG = r"C:\Windows\Fonts\arial.ttf"

CONTEXT_FRAMES = 3  # real frames shown before and after the gap, for orientation


def load_detections_and_frames():
    detections_by_frame = defaultdict(list)
    clip_frame_numbers = defaultdict(set)
    with open(OUT_ROOT / "detections.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clip, frame_no = row["clip"], int(row["frame_number"])
            clip_frame_numbers[clip].add(frame_no)
            detections_by_frame[(clip, frame_no)].append({
                "class": row["class"], "confidence": float(row["confidence"]),
                "x1": float(row["x1"]), "y1": float(row["y1"]),
                "x2": float(row["x2"]), "y2": float(row["y2"]),
            })
    clip_frame_numbers = {c: sorted(v) for c, v in clip_frame_numbers.items()}

    image_path_by_frame = {}
    with open(OUT_ROOT / "clip_manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            image_path_by_frame[(row["clip_name"], int(row["frame_number"]))] = (
                PROJECT_ROOT / row["experiment_image_path"])
    return dict(detections_by_frame), clip_frame_numbers, image_path_by_frame


def get_track(clip, track_id, natural_tracks):
    for t in natural_tracks:
        if t.clip == clip and t.track_id == track_id:
            return t
    raise KeyError(f"track {clip}#{track_id} not found")


def render_comparison_video(track, gap_start_idx, gap_len, image_path_by_frame, out_path):
    n = len(track.frame_numbers)
    lo = max(0, gap_start_idx - CONTEXT_FRAMES)
    hi = min(n, gap_start_idx + gap_len + CONTEXT_FRAMES)

    results = run_three_baselines(track.boxes, track.class_name, gap_start_idx, gap_len)

    frames = []
    for i in range(lo, hi):
        frame_no = track.frame_numbers[i]
        img_path = image_path_by_frame[(track.clip, frame_no)]
        real_box = track.boxes[i]
        in_gap = gap_start_idx <= i < gap_start_idx + gap_len

        if in_gap:
            offset = i - gap_start_idx
            r = results[offset]
            panels = [
                panel(img_path, {"yolo": None}, "YOLO ONLY -- box lost"),
                panel(img_path, {"static": r.static_memory_box}, "STATIC MEMORY (Assignment 2) -- prediction"),
                panel(img_path, {"sort": r.sort_box}, "SORT MOTION PREDICTION -- prediction"),
                panel(img_path, {"withheld": real_box}, "WITHHELD YOLO REFERENCE (pseudo-GT)"),
            ]
        else:
            panels = [
                panel(img_path, {"yolo": real_box}, "YOLO ONLY -- real detection"),
                panel(img_path, {"yolo": real_box}, "STATIC MEMORY -- real detection"),
                panel(img_path, {"yolo": real_box}, "SORT MOTION PREDICTION -- real detection"),
                panel(img_path, {"yolo": real_box}, "WITHHELD YOLO REFERENCE (pseudo-GT)"),
            ]
        frames.append(np.array(compose_grid(panels, cols=2))[:, :, ::-1])  # RGB -> BGR

    h, w = frames[0].shape[:2]
    VIDEOS_ROOT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"Wrote {out_path} ({len(frames)} frames)")


PHASE_CAPTIONS = {
    "before": "Object is visible -- YOLO detects it normally (green box).",
    "removed": "Detection artificially removed -- this frame's YOLO box is withheld to simulate a miss.",
    "static": "Static memory (Assignment 2) freezes the last box in place -- it will not move on its own.",
    "sort": "SORT's Kalman filter predicts the box forward using the object's estimated velocity (blue).",
    "corrected": "A new detection returns -- SORT corrects its estimate; static memory was still frozen.",
}


def render_explanatory_video(track, gap_start_idx, gap_len, image_path_by_frame, out_path):
    n = len(track.frame_numbers)
    lo = max(0, gap_start_idx - CONTEXT_FRAMES)
    hi = min(n, gap_start_idx + gap_len + CONTEXT_FRAMES)
    results = run_three_baselines(track.boxes, track.class_name, gap_start_idx, gap_len)

    font_caption = ImageFont.truetype(FONT_BOLD, 20)
    frame_size = (960, 540)
    out_frames = []

    def make_frame(i, caption, hold=1):
        frame_no = track.frame_numbers[i]
        img_path = image_path_by_frame[(track.clip, frame_no)]
        real_box = track.boxes[i]
        in_gap = gap_start_idx <= i < gap_start_idx + gap_len

        img = Image.open(img_path).convert("RGB")
        scale = frame_size[0] / img.width
        img = img.resize(frame_size)
        draw = ImageDraw.Draw(img)

        def scaled(box):
            return tuple(v * scale for v in box)

        if in_gap:
            offset = i - gap_start_idx
            r = results[offset]
            b = scaled(r.static_memory_box)
            draw.rectangle(b, outline=ORANGE, width=3)
            draw.text((b[0], max(0, b[1] - 22)), "MEMORY (prediction)", font=font_caption, fill=ORANGE)
            b2 = scaled(r.sort_box)
            draw.rectangle(b2, outline=BLUE, width=3)
            draw.text((b2[0], min(frame_size[1] - 24, b2[3] + 4)), "SORT (prediction)", font=font_caption, fill=BLUE)
        else:
            b = scaled(real_box)
            draw.rectangle(b, outline=GREEN, width=3)
            draw.text((b[0], max(0, b[1] - 22)), "YOLO (real detection)", font=font_caption, fill=GREEN)

        # Caption bar
        bar_h = 70
        full = Image.new("RGB", (frame_size[0], frame_size[1] + bar_h), (15, 15, 15))
        full.paste(img, (0, 0))
        d2 = ImageDraw.Draw(full)
        d2.text((10, frame_size[1] + 10), caption, font=font_caption, fill=WHITE)

        arr = np.array(full)[:, :, ::-1]
        return [arr] * hold

    for i in range(lo, gap_start_idx):
        out_frames.extend(make_frame(i, PHASE_CAPTIONS["before"], hold=2))

    for offset in range(gap_len):
        i = gap_start_idx + offset
        caption = PHASE_CAPTIONS["removed"] if offset == 0 else PHASE_CAPTIONS["sort"]
        out_frames.extend(make_frame(i, caption, hold=4))  # pause longer during the gap

    for j, i in enumerate(range(gap_start_idx + gap_len, hi)):
        caption = PHASE_CAPTIONS["corrected"] if j == 0 else PHASE_CAPTIONS["before"]
        out_frames.extend(make_frame(i, caption, hold=4 if j == 0 else 2))

    h, w = out_frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
    for f in out_frames:
        writer.write(f)
    writer.release()
    print(f"Wrote {out_path} ({len(out_frames)} frames)")


def main() -> None:
    detections_by_frame, clip_frame_numbers, image_path_by_frame = load_detections_and_frames()
    natural_tracks = build_natural_tracks(detections_by_frame, clip_frame_numbers)

    examples = [
        ("clip_sample_003", 273, 3, "comparison_clip_sample_003_track273_gap3.mp4"),
        ("clip_sample_006", 430, 5, "comparison_clip_sample_006_track430_gap5.mp4"),
        ("clip_sample_001", 4, 2, "comparison_clip_sample_001_track4_gap2.mp4"),
    ]

    for clip, track_id, gap_len, filename in examples:
        track = get_track(clip, track_id, natural_tracks)
        n = len(track.frame_numbers)
        gap_start_idx = (n - gap_len) // 2
        render_comparison_video(track, gap_start_idx, gap_len, image_path_by_frame, VIDEOS_ROOT / filename)

    # Slow explanatory video on the most dramatic example.
    clip, track_id, gap_len = "clip_sample_006", 430, 5
    track = get_track(clip, track_id, natural_tracks)
    n = len(track.frame_numbers)
    gap_start_idx = (n - gap_len) // 2
    render_explanatory_video(track, gap_start_idx, gap_len, image_path_by_frame,
                              VIDEOS_ROOT / "explanatory_predict_correct_loop.mp4")


if __name__ == "__main__":
    main()

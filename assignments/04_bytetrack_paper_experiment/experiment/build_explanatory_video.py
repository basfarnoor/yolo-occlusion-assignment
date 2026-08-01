"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 13: one slow explanatory video, built on the pedestrian recovery
scenario (clip_sample_011), that pauses on each of the required moments:
confidence crossing below the high threshold, SORT losing the object into
motion-only prediction, ByteTrack's second association matching a low-score
box, and a later high-confidence detection returning.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from run_methods import new_bytetrack_tracker, new_sort_tracker  # noqa: E402
from track import EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE  # noqa: E402
from visualization import BLUE, CYAN, GREEN, ORANGE, PURPLE, WHITE, YELLOW  # noqa: E402

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
VIDEOS_ROOT = OUT_ROOT / "videos"
FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"

CLIP = "clip_sample_011"
INSTANCE_TOKEN = "8c3247533921448abe371a59d4696252"
FRAME_RANGE = list(range(1, 14))
FRAME_SIZE = (960, 540)


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    with open(EXP_ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    high_score_threshold = cfg["tracker"]["high_score_threshold"]

    manifest_rows = load_csv(OUT_ROOT / "clip_manifest.csv")
    detections = load_csv(OUT_ROOT / "detections.csv")

    image_path_by_frame, timestamps = {}, {}
    for row in manifest_rows:
        if row["clip_name"] != CLIP:
            continue
        frame_no = int(row["frame_number"])
        image_path_by_frame[frame_no] = ASSIGNMENT_ROOT / row["experiment_image_path"]
        timestamps[frame_no] = float(row["timestamp"]) / 1_000_000.0

    detections_by_frame = {}
    for d in detections:
        if d["clip"] != CLIP:
            continue
        detections_by_frame.setdefault(int(d["frame_number"]), []).append({
            "class": d["class"], "confidence": float(d["confidence"]),
            "x1": float(d["x1"]), "y1": float(d["y1"]), "x2": float(d["x2"]), "y2": float(d["y2"]),
        })

    sort_tracker = new_sort_tracker(cfg)
    byte_tracker = new_bytetrack_tracker(cfg)

    font_caption = ImageFont.truetype(FONT_BOLD, 18)
    out_frames = []

    def find_target(outputs):
        # The pedestrian is the only "person" class track in this clip near frame 1.
        for o in outputs:
            if o.class_name == "person":
                return o
        return None

    prev_evidence = {"sort": None, "byte": None}

    for frame_no in sorted(detections_by_frame.keys() | image_path_by_frame.keys()):
        if frame_no > max(FRAME_RANGE):
            break
        dets = detections_by_frame.get(frame_no, [])
        sort_outputs = sort_tracker.update(dets, timestamp=timestamps[frame_no])
        byte_outputs = byte_tracker.update(dets, timestamp=timestamps[frame_no])
        if frame_no not in FRAME_RANGE:
            continue

        sort_t = find_target(sort_outputs)
        byte_t = find_target(byte_outputs)

        raw_conf = max((d["confidence"] for d in dets if d["class"] == "person"), default=None)

        captions = []
        hold = 2
        if raw_conf is not None and raw_conf < high_score_threshold:
            captions.append(f"Confidence crosses below the high threshold ({raw_conf:.2f} < {high_score_threshold}).")
            hold = 5
        if sort_t and sort_t.evidence_source not in (EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE) and prev_evidence["sort"] in (EVIDENCE_HIGH_SCORE, EVIDENCE_LOW_SCORE, None):
            captions.append("High-confidence SORT has no second chance -- it discards the weak box.")
            hold = max(hold, 5)
        if byte_t and byte_t.evidence_source == EVIDENCE_LOW_SCORE:
            captions.append("ByteTrack's second association matches the weak box to the existing track.")
            hold = max(hold, 5)
        if raw_conf is not None and raw_conf >= high_score_threshold and prev_evidence["byte"] not in (EVIDENCE_HIGH_SCORE, None):
            captions.append("A later high-confidence detection returns -- both methods reconnect.")
            hold = max(hold, 4)
        if not captions:
            captions.append("Object visible; both methods track it normally.")

        if sort_t:
            prev_evidence["sort"] = sort_t.evidence_source
        if byte_t:
            prev_evidence["byte"] = byte_t.evidence_source

        img = Image.open(image_path_by_frame[frame_no]).convert("RGB")
        scale = FRAME_SIZE[0] / img.width
        img = img.resize(FRAME_SIZE)
        draw = ImageDraw.Draw(img)

        def scaled(box):
            return tuple(v * scale for v in box)

        for d in dets:
            if d["class"] != "person":
                continue
            color = GREEN if d["confidence"] >= high_score_threshold else YELLOW
            b = scaled((d["x1"], d["y1"], d["x2"], d["y2"]))
            draw.rectangle(b, outline=color, width=2)

        if sort_t:
            b = scaled(sort_t.box)
            draw.rectangle(b, outline=BLUE, width=3)
            draw.text((b[0], max(0, b[1] - 40)), "SORT", font=font_caption, fill=BLUE)
        if byte_t:
            b = scaled(byte_t.box)
            color = CYAN if byte_t.evidence_source == EVIDENCE_HIGH_SCORE else (
                ORANGE if byte_t.evidence_source == EVIDENCE_LOW_SCORE else PURPLE)
            draw.rectangle(b, outline=color, width=3)
            draw.text((b[0], min(FRAME_SIZE[1] - 20, b[3] + 4)), "ByteTrack", font=font_caption, fill=color)

        bar_h = 90
        full = Image.new("RGB", (FRAME_SIZE[0], FRAME_SIZE[1] + bar_h), (15, 15, 15))
        full.paste(img, (0, 0))
        d2 = ImageDraw.Draw(full)
        d2.text((10, FRAME_SIZE[1] + 8), f"Frame {frame_no}", font=font_caption, fill=WHITE)
        for i, cap in enumerate(captions):
            d2.text((10, FRAME_SIZE[1] + 30 + i * 24), cap, font=font_caption, fill=WHITE)

        arr = np.array(full)[:, :, ::-1]
        out_frames.extend([arr] * hold)

    h, w = out_frames[0].shape[:2]
    out_path = VIDEOS_ROOT / "explanatory_bytetrack_second_association.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
    for f in out_frames:
        writer.write(f)
    writer.release()
    print(f"Wrote {out_path} ({len(out_frames)} frames)")


if __name__ == "__main__":
    main()

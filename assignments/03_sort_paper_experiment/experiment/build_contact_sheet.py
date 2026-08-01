"""Task 3 visual sanity check: a grid of frames with their cached YOLO boxes.
Not for student annotation -- just a quick look that detection looks sane."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = PROJECT_ROOT / "results"
DETECTIONS_CSV = OUT_ROOT / "detections.csv"
MANIFEST_CSV = OUT_ROOT / "clip_manifest.csv"
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"

CONF_DISPLAY_THRESHOLD = 0.3
THUMB_W = 480


def main() -> None:
    frames = list(csv.DictReader(open(MANIFEST_CSV, encoding="utf-8")))
    dets_by_frame = defaultdict(list)
    for row in csv.DictReader(open(DETECTIONS_CSV, encoding="utf-8")):
        key = (row["clip"], row["frame_number"])
        dets_by_frame[key].append(row)

    clips = sorted({f["clip_name"] for f in frames})
    picks = []
    for clip in clips:
        clip_frames = [f for f in frames if f["clip_name"] == clip]
        n = len(clip_frames)
        for idx in (0, n // 2, n - 1):
            picks.append(clip_frames[idx])

    font = ImageFont.truetype(FONT_PATH, 14)
    thumbs = []
    for f in picks:
        img_path = PROJECT_ROOT / f["experiment_image_path"]
        img = Image.open(img_path).convert("RGB")
        scale = THUMB_W / img.width
        img = img.resize((THUMB_W, int(img.height * scale)))
        draw = ImageDraw.Draw(img)
        key = (f["clip_name"], f["frame_number"])
        for d in dets_by_frame.get(key, []):
            if float(d["confidence"]) < CONF_DISPLAY_THRESHOLD:
                continue
            x1, y1 = float(d["x1"]) * scale, float(d["y1"]) * scale
            x2, y2 = float(d["x2"]) * scale, float(d["y2"]) * scale
            draw.rectangle([x1, y1, x2, y2], outline=(0, 200, 0), width=2)
            draw.text((x1, max(0, y1 - 14)), f"{d['class']} {float(d['confidence']):.2f}",
                       font=font, fill=(0, 255, 0))
        draw.rectangle([0, 0, img.width, 18], fill=(0, 0, 0))
        draw.text((4, 2), f"{f['clip_name']} frame {f['frame_number']}", font=font, fill=(255, 255, 255))
        thumbs.append(img)

    cols = 3
    rows = (len(thumbs) + cols - 1) // cols
    cell_w, cell_h = thumbs[0].width, thumbs[0].height
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), (30, 30, 30))
    for i, thumb in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet.paste(thumb, (c * cell_w, r * cell_h))

    out_path = OUT_ROOT / "contact_sheet.png"
    sheet.save(out_path)
    print(f"Wrote {out_path} ({len(thumbs)} frames, {CONF_DISPLAY_THRESHOLD}+ confidence shown)")


if __name__ == "__main__":
    main()

"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 4: draws the accepted projected ground-truth boxes onto their keyframe
images (results/projection_overlays/<clip>/frame_XXX.jpg) and assembles a
contact sheet of clear and difficult examples for the required visual review.
Local-only output (not committed -- see .gitignore).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
OVERLAY_ROOT = OUT_ROOT / "projection_overlays"

MAGENTA = (230, 0, 200)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def load_clip_manifest() -> dict[tuple[str, str], str]:
    path_by_key = {}
    with open(OUT_ROOT / "clip_manifest.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path_by_key[(row["clip_name"], row["frame_number"])] = row["experiment_image_path"]
    return path_by_key


def main() -> None:
    path_by_key = load_clip_manifest()

    boxes_by_frame: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with open(OUT_ROOT / "projected_ground_truth.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["rejected"] in ("True", "true"):
                continue
            boxes_by_frame[(row["clip_name"], row["frame_number"])].append(row)

    font = ImageFont.truetype(FONT_PATH, 12)
    contact_thumbs = []

    for (clip_name, frame_number), boxes in sorted(boxes_by_frame.items()):
        rel_path = path_by_key.get((clip_name, frame_number))
        if rel_path is None:
            continue
        img_path = ASSIGNMENT_ROOT / rel_path
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            box = (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]))
            draw.rectangle(box, outline=MAGENTA, width=3)
            label = f"{b['category'].split('.')[-1]} ({b['visibility_level']})"
            ty = max(0, box[1] - 14)
            draw.rectangle([box[0] - 1, ty - 1, box[0] + 7 * len(label), ty + 12], fill=MAGENTA)
            draw.text((box[0], ty), label, font=font, fill=(255, 255, 255))

        out_dir = OVERLAY_ROOT / clip_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{int(frame_number):03d}.jpg"
        img.save(out_path, quality=90)
        contact_thumbs.append((clip_name, frame_number, out_path, len(boxes)))

    # Build a contact sheet: one keyframe per clip (24 candidates -> pick a
    # spread of 20+ across clips, favoring the clearest and busiest frames).
    contact_thumbs.sort(key=lambda t: -t[3])
    chosen = contact_thumbs[:20] if len(contact_thumbs) >= 20 else contact_thumbs
    chosen.sort(key=lambda t: (t[0], int(t[1])))

    cols = 4
    thumb_w, thumb_h = 400, 225
    rows = (len(chosen) + cols - 1) // cols
    sheet = Image.new("RGB", (thumb_w * cols, thumb_h * rows), (20, 20, 20))
    sheet_font = ImageFont.truetype(FONT_PATH, 14)
    sdraw = ImageDraw.Draw(sheet)
    for i, (clip_name, frame_number, path, n_boxes) in enumerate(chosen):
        thumb = Image.open(path).resize((thumb_w, thumb_h))
        r, c = divmod(i, cols)
        sheet.paste(thumb, (c * thumb_w, r * thumb_h))
        label = f"{clip_name} f{frame_number} ({n_boxes} boxes)"
        sdraw.rectangle([c * thumb_w, r * thumb_h, c * thumb_w + 7 * len(label), r * thumb_h + 16], fill=(0, 0, 0))
        sdraw.text((c * thumb_w + 2, r * thumb_h + 1), label, font=sheet_font, fill=(255, 255, 255))

    contact_path = OUT_ROOT / "projection_overlays" / "contact_sheet.png"
    sheet.save(contact_path)

    print(f"Wrote {len(contact_thumbs)} overlay images under {OVERLAY_ROOT}")
    print(f"Wrote contact sheet ({len(chosen)} frames): {contact_path}")


if __name__ == "__main__":
    main()

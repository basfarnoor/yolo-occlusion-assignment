"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 9: draws before/event/after triptychs (with the projected ground-truth
box) for each selected natural event, for the required visual review.
Local-only output.
"""
from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

EXP_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_ROOT = EXP_ROOT.parent
OUT_ROOT = ASSIGNMENT_ROOT / "results"
SHEET_DIR = OUT_ROOT / "natural_event_contact_sheets"

MAGENTA = (230, 0, 200)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    events = load_csv(OUT_ROOT / "natural_event_manifest.csv")
    if not events:
        print("No events to render.")
        return

    clip_paths = {(r["clip_name"], r["frame_number"]): r["experiment_image_path"]
                  for r in load_csv(OUT_ROOT / "clip_manifest.csv")}
    gt_by_key = {}
    for r in load_csv(OUT_ROOT / "projected_ground_truth.csv"):
        gt_by_key.setdefault((r["clip_name"], r["frame_number"], r["instance_token"]), r)

    font = ImageFont.truetype(FONT_PATH, 16)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    thumb_w, thumb_h = 350, 197

    for idx, e in enumerate(events, start=1):
        panels = []
        for role, frame_key in (("BEFORE", "before_frame"), ("EVENT", "event_frame"), ("AFTER", "after_frame")):
            frame_no = e[frame_key]
            rel_path = clip_paths.get((e["clip_name"], frame_no))
            img = Image.open(ASSIGNMENT_ROOT / rel_path).convert("RGB").resize((thumb_w, thumb_h))
            draw = ImageDraw.Draw(img)
            gt = gt_by_key.get((e["clip_name"], frame_no, e["instance_token"]))
            if gt:
                scale_x = thumb_w / 1600
                scale_y = thumb_h / 900
                box = (float(gt["x1"]) * scale_x, float(gt["y1"]) * scale_y,
                       float(gt["x2"]) * scale_x, float(gt["y2"]) * scale_y)
                draw.rectangle(box, outline=MAGENTA, width=3)
            conf_key = {"before_frame": "before_confidence", "event_frame": "event_confidence",
                        "after_frame": "after_confidence"}[frame_key]
            conf = e[conf_key]
            conf_str = f"{float(conf):.2f}" if conf not in ("", "None") else "none"
            label = f"{role} f{frame_no} conf={conf_str}"
            draw.rectangle([0, 0, thumb_w, 22], fill=(0, 0, 0))
            draw.text((3, 3), label, font=font, fill=(255, 255, 255))
            panels.append(img)

        sheet = Image.new("RGB", (thumb_w * 3, thumb_h), (20, 20, 20))
        for i, p in enumerate(panels):
            sheet.paste(p, (i * thumb_w, 0))
        out_path = SHEET_DIR / f"event_{idx:02d}_{e['clip_name']}_{e['category'].split('.')[-1]}.png"
        sheet.save(out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

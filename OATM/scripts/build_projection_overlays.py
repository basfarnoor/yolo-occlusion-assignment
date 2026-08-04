"""Phase 2 (Task 3): draws accepted projected ground-truth boxes onto their
keyframe images for the required visual review (>= 50 overlays). Local-only
output -- not committed, regenerable any time.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
MAGENTA = (230, 0, 200)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
N_REVIEW_FRAMES = 50
CONTACT_SHEET_COLS = 5
CONTACT_SHEET_ROWS_PER_SHEET = 2  # 10 frames per sheet -> 5 sheets for 50 frames


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)

    pgt = pd.read_parquet(config.artifacts_dir / "projected_ground_truth.parquet")
    frame_index = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")
    path_by_sd_token = dict(zip(frame_index["sample_data_token"], frame_index["image_path"]))

    boxes_by_frame: dict[str, list[dict]] = defaultdict(list)
    for _, row in pgt.iterrows():
        boxes_by_frame[row["sample_data_token"]].append(row.to_dict())

    # Deterministic, evenly-spaced sample of keyframes-with-boxes across all scenes.
    frame_tokens = sorted(boxes_by_frame.keys())
    if len(frame_tokens) > N_REVIEW_FRAMES:
        step = len(frame_tokens) / N_REVIEW_FRAMES
        sample_tokens = [frame_tokens[int(i * step)] for i in range(N_REVIEW_FRAMES)]
    else:
        sample_tokens = frame_tokens

    overlay_dir = config.artifacts_dir / "projection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, 12)

    overlay_paths = []
    for token in sample_tokens:
        rel_path = path_by_sd_token.get(token)
        if rel_path is None:
            continue
        img = Image.open(config.data_root / rel_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        n_boxes = 0
        for b in boxes_by_frame[token]:
            box = (b["x1"], b["y1"], b["x2"], b["y2"])
            draw.rectangle(box, outline=MAGENTA, width=3)
            label = f"{b['evaluation_class']}"
            ty = max(0, box[1] - 14)
            draw.rectangle([box[0] - 1, ty - 1, box[0] + 7 * len(label), ty + 12], fill=MAGENTA)
            draw.text((box[0], ty), label, font=font, fill=(255, 255, 255))
            n_boxes += 1
        out_path = overlay_dir / f"{token}.jpg"
        img.save(out_path, quality=85)
        overlay_paths.append((token, out_path, n_boxes))

    # Build contact sheets (10 frames each) for compact visual review.
    thumb_w, thumb_h = 320, 180
    frames_per_sheet = CONTACT_SHEET_COLS * CONTACT_SHEET_ROWS_PER_SHEET
    sheet_font = ImageFont.truetype(FONT_PATH, 13)
    n_sheets = 0
    for sheet_start in range(0, len(overlay_paths), frames_per_sheet):
        chunk = overlay_paths[sheet_start:sheet_start + frames_per_sheet]
        rows = (len(chunk) + CONTACT_SHEET_COLS - 1) // CONTACT_SHEET_COLS
        sheet = Image.new("RGB", (thumb_w * CONTACT_SHEET_COLS, thumb_h * rows), (20, 20, 20))
        sdraw = ImageDraw.Draw(sheet)
        for i, (token, path, n_boxes) in enumerate(chunk):
            thumb = Image.open(path).resize((thumb_w, thumb_h))
            r, c = divmod(i, CONTACT_SHEET_COLS)
            sheet.paste(thumb, (c * thumb_w, r * thumb_h))
            label = f"{token[:8]} ({n_boxes})"
            sdraw.rectangle([c * thumb_w, r * thumb_h, c * thumb_w + 7 * len(label), r * thumb_h + 15],
                             fill=(0, 0, 0))
            sdraw.text((c * thumb_w + 2, r * thumb_h + 1), label, font=sheet_font, fill=(255, 255, 255))
        sheet_path = overlay_dir / f"contact_sheet_{n_sheets + 1:02d}.png"
        sheet.save(sheet_path)
        n_sheets += 1

    print(f"Wrote {len(overlay_paths)} overlay images and {n_sheets} contact sheets under {overlay_dir}")


if __name__ == "__main__":
    main()

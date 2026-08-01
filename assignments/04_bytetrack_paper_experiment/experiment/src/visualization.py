"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 13: shared drawing helpers for the comparison and explanatory videos.
Colors follow the assignment's fixed scheme so every box's meaning is visible
without reading code: high-confidence detection green, low-confidence
detection yellow, SORT track blue, ByteTrack-high-score cyan,
ByteTrack-low-score orange, motion-only prediction dashed purple, offline
ground-truth reference magenta (dotted). Reused drawing primitives (dashed
rectangle, labeling, grid composition) from the Assignment 3 SORT experiment
-- see reuse_audit.md.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

FONT_REG = r"C:\Windows\Fonts\arial.ttf"

GREEN = (0, 200, 0)             # high-confidence detection
YELLOW = (230, 200, 0)          # low-confidence detection
BLUE = (30, 120, 255)           # SORT track (matched)
CYAN = (0, 210, 210)            # ByteTrack track updated by a high-score detection
ORANGE = (255, 140, 0)          # ByteTrack track updated by a low-score detection
PURPLE = (150, 60, 220)         # motion-only prediction (either method)
MAGENTA = (230, 0, 200)         # offline ground-truth reference
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def draw_dashed_rect(draw: ImageDraw.ImageDraw, box, color, width=4, dash=14):
    x1, y1, x2, y2 = box
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dash, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dash, x2), y2)], fill=color, width=width)
        x += dash * 2
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dash, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dash, y2))], fill=color, width=width)
        y += dash * 2


def draw_dotted_rect(draw: ImageDraw.ImageDraw, box, color, width=4, dot=6, gap=8):
    x1, y1, x2, y2 = box
    x = x1
    while x < x2:
        draw.line([(x, y1), (min(x + dot, x2), y1)], fill=color, width=width)
        draw.line([(x, y2), (min(x + dot, x2), y2)], fill=color, width=width)
        x += dot + gap
    y = y1
    while y < y2:
        draw.line([(x1, y), (x1, min(y + dot, y2))], fill=color, width=width)
        draw.line([(x2, y), (x2, min(y + dot, y2))], fill=color, width=width)
        y += dot + gap


def label(draw: ImageDraw.ImageDraw, box, text, color, font, above=True):
    x1, y1, y2 = box[0], box[1], box[3]
    y = max(0, y1 - 16) if above else min(y2 + 2, 1e6)
    bbox = draw.textbbox((x1, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=color)
    text_color = BLACK if color in (YELLOW, CYAN) else WHITE
    draw.text((x1, y), text, font=font, fill=text_color)


def panel_from_boxes(image_path, boxes: list[tuple], panel_label: str, size=(400, 225)) -> Image.Image:
    """boxes: list of (box, color, text, style) where style is 'solid', 'dashed', or 'dotted'."""
    img = Image.open(image_path).convert("RGB")
    scale = size[0] / img.width
    img = img.resize(size)
    draw = ImageDraw.Draw(img)
    font_small = ImageFont.truetype(FONT_REG, 11)

    def scaled(box):
        return tuple(v * scale for v in box)

    for box, color, text, style in boxes:
        b = scaled(box)
        if style == "dashed":
            draw_dashed_rect(draw, b, color, width=2)
        elif style == "dotted":
            draw_dotted_rect(draw, b, color, width=2)
        else:
            draw.rectangle(b, outline=color, width=2)
        if text:
            label(draw, b, text, color, font_small)

    draw.rectangle([0, 0, size[0], 18], fill=(0, 0, 0))
    draw.text((3, 2), panel_label, font=font_small, fill=WHITE)
    return img


def compose_grid(panels: list[Image.Image], cols=2) -> Image.Image:
    w, h = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid.paste(p, (c * w, r * h))
    return grid

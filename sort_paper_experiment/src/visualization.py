"""SORT paper reference: Bewley et al., ICIP 2016 (SORT), arxiv.org/abs/1602.00763.

Task 8: shared drawing helpers for the comparison and explanatory videos.
Colors are fixed per the assignment: YOLO observation green, static memory
orange, SORT prediction blue, withheld reference magenta. Every predicted
box is labeled as a prediction, never as a real detection.
"""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

FONT_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
FONT_REG = r"C:\Windows\Fonts\arial.ttf"

GREEN = (0, 200, 0)
ORANGE = (255, 140, 0)
BLUE = (30, 120, 255)
MAGENTA = (230, 0, 200)
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
    y = max(0, y1 - 24) if above else min(y2 + 4, 1e6)
    bbox = draw.textbbox((x1, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=color)
    text_color = BLACK if color in (ORANGE,) else WHITE
    draw.text((x1, y), text, font=font, fill=text_color)


def panel(image_path, boxes: dict, panel_label: str, size=(640, 360)) -> Image.Image:
    """boxes: {"yolo": box_or_None, "static": box_or_None, "sort": box_or_None, "withheld": box_or_None}"""
    img = Image.open(image_path).convert("RGB")
    scale = size[0] / img.width
    img = img.resize(size)
    draw = ImageDraw.Draw(img)
    font_small = ImageFont.truetype(FONT_REG, 13)
    font_tiny = ImageFont.truetype(FONT_REG, 11)

    def scaled(box):
        return tuple(v * scale for v in box)

    if boxes.get("yolo") is not None:
        b = scaled(boxes["yolo"])
        draw.rectangle(b, outline=GREEN, width=3)
        label(draw, b, "YOLO", GREEN, font_tiny)
    if boxes.get("static") is not None:
        b = scaled(boxes["static"])
        draw_dashed_rect(draw, b, ORANGE, width=3)
        label(draw, b, "MEMORY (prediction)", ORANGE, font_tiny)
    if boxes.get("sort") is not None:
        b = scaled(boxes["sort"])
        draw.rectangle(b, outline=BLUE, width=3)
        label(draw, b, "SORT (prediction)", BLUE, font_tiny, above=False)
    if boxes.get("withheld") is not None:
        b = scaled(boxes["withheld"])
        draw_dotted_rect(draw, b, MAGENTA, width=2)
        label(draw, b, "withheld reference", MAGENTA, font_tiny, above=False)

    draw.rectangle([0, 0, size[0], 20], fill=(0, 0, 0))
    draw.text((4, 3), panel_label, font=font_small, fill=WHITE)
    return img


def compose_grid(panels: list[Image.Image], cols=2) -> Image.Image:
    w, h = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    grid = Image.new("RGB", (w * cols, h * rows), (20, 20, 20))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        grid.paste(p, (c * w, r * h))
    return grid

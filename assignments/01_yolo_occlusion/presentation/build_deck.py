"""Build the OATM project presentation (python-pptx, no Node/Canva dependency).

Import the resulting .pptx into Canva via "Import PPTX" for further editing.
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.oxml.ns import qn
import copy

NAVY = RGBColor(0x1E, 0x27, 0x61)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
AMBER = RGBColor(0xFF, 0xA6, 0x30)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x5B, 0x6B, 0x8C)
DARKGRAY = RGBColor(0x44, 0x44, 0x44)

HEADER_FONT = "Cambria"
BODY_FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide(bg_color):
    slide = prs.slides.add_slide(BLANK)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = bg_color
    return slide


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def add_text(slide, left, top, width, height, text, size, color, bold=False,
             italic=False, font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=None,
             line_spacing=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    if anchor is not None:
        tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color
    return box


def add_bullets(slide, left, top, width, height, items, size, color,
                 font=BODY_FONT, space_after=10, bold_lead=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(space_after)
        run = p.add_run()
        run.text = f"•  {item}"
        run.font.size = Pt(size)
        run.font.font = None
        run.font.name = font
        run.font.color.rgb = color
    return box


def add_icon_circle(slide, emoji, cx, cy, diameter, circle_color, emoji_size=28):
    left = cx - diameter // 2
    top = cy - diameter // 2
    shape = slide.shapes.add_shape(MSO_SHAPE.OVAL, left, top, diameter, diameter)
    shape.fill.solid()
    shape.fill.fore_color.rgb = circle_color
    shape.line.fill.background()
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = False
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = emoji
    run.font.size = Pt(emoji_size)
    return shape


def add_icon_row(slide, left, top, width, emoji, text, size=15, color=DARKGRAY,
                  circle_color=NAVY, row_h=Inches(0.85), icon_d=Inches(0.55)):
    add_icon_circle(slide, emoji, left + icon_d // 2, top + row_h // 2, icon_d, circle_color, emoji_size=18)
    text_left = left + icon_d + Inches(0.25)
    text_width = width - icon_d - Inches(0.25)
    box = slide.shapes.add_textbox(text_left, top, text_width, row_h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.name = BODY_FONT
    run.font.color.rgb = color
    return box


def add_rounded_box(slide, left, top, width, height, fill_color, line=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    try:
        shape.adjustments[0] = 0.08
    except Exception:
        pass
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line:
        shape.line.color.rgb = fill_color
    else:
        shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def style_chart_simple(chart, color_hex, show_data_labels=True, num_fmt="0%"):
    chart.has_legend = False
    plot = chart.plots[0]
    plot.has_data_labels = show_data_labels
    if show_data_labels:
        dls = plot.data_labels
        dls.number_format = num_fmt
        dls.number_format_is_linked = False
        dls.font.size = Pt(13)
        dls.font.bold = True
        dls.font.color.rgb = NAVY
        dls.position = XL_LABEL_POSITION.OUTSIDE_END
    series = plot.series[0]
    series.format.fill.solid()
    series.format.fill.fore_color.rgb = RGBColor.from_string(color_hex)
    chart.category_axis.tick_labels.font.size = Pt(12)
    chart.category_axis.tick_labels.font.color.rgb = DARKGRAY
    chart.category_axis.format.line.color.rgb = RGBColor(0xCC, 0xCC, 0xCC)
    chart.value_axis.visible = False
    chart.value_axis.has_major_gridlines = False
    chart.category_axis.has_major_gridlines = False


# ---------------------------------------------------------------- Slide 1
s = add_slide(NAVY)
add_icon_circle(s, "\U0001F441", SLIDE_W // 2, Inches(1.7), Inches(1.5), AMBER, emoji_size=44)
add_text(s, Inches(1), Inches(2.7), Inches(11.33), Inches(1.2),
         "Occlusion-Adaptive Temporal Memory", 40, WHITE, bold=True,
         font=HEADER_FONT, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.5), Inches(3.75), Inches(10.33), Inches(0.7),
         "Giving self-driving cars short-term memory for hidden objects", 20, ICE,
         italic=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(1.5), Inches(6.6), Inches(10.33), Inches(0.5),
         "An nuScenes-based autonomous-vehicle perception research project", 14, MUTED,
         align=PP_ALIGN.CENTER)
add_notes(s, "Introduce the project: this is about helping self-driving car cameras keep track "
             "of road users even when something briefly blocks the view. Set up that this is an "
             "active research project, not just a class exercise.")

# ---------------------------------------------------------------- Slide 2
s = add_slide(WHITE)
add_text(s, Inches(0.6), Inches(0.5), Inches(8), Inches(0.9),
          "Single-frame detectors forget what they can't see", 28, NAVY, bold=True, font=HEADER_FONT)
add_bullets(s, Inches(0.6), Inches(1.7), Inches(6.6), Inches(3.5), [
    "Most object detectors look at one camera image at a time",
    "If a truck briefly blocks a pedestrian, the model treats them as gone",
    "When the pedestrian reappears, it's logged as a brand-new person, not the same one",
], 17, DARKGRAY, space_after=16)

add_icon_circle(s, "❓", Inches(10.6), Inches(2.0), Inches(1.5), NAVY, emoji_size=40)
callout = add_rounded_box(s, Inches(7.7), Inches(3.3), Inches(4.9), Inches(2.4), ICE)
tf = callout.text_frame
tf.word_wrap = True
tf.margin_left = Inches(0.35)
tf.margin_right = Inches(0.35)
tf.margin_top = Inches(0.3)
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.LEFT
run = p.add_run()
run.text = "“Imagine watching someone walk behind a parked bus — you don't assume they vanished.”"
run.font.size = Pt(16)
run.font.italic = True
run.font.name = BODY_FONT
run.font.color.rgb = NAVY
add_notes(s, "Explain the core problem with a simple analogy. Point to the callout: humans naturally "
             "remember what was there a moment ago; most detectors don't. That's the gap this project "
             "is trying to close.")

# ---------------------------------------------------------------- Slide 3
s = add_slide(NAVY)
add_text(s, Inches(1), Inches(2.0), Inches(11.33), Inches(0.5),
          "THE CORE RESEARCH QUESTION", 15, AMBER, bold=True, align=PP_ALIGN.CENTER, font=BODY_FONT)
add_text(s, Inches(1.3), Inches(2.6), Inches(10.73), Inches(2.2),
          "Can short-term visual memory help a camera keep track of temporarily hidden "
          "road users — without inventing false ones?", 30, WHITE, bold=True,
          font=HEADER_FONT, align=PP_ALIGN.CENTER, line_spacing=1.15)
add_text(s, Inches(1.8), Inches(5.4), Inches(9.73), Inches(1.3),
          "Hypothesis: occlusion-aware temporal memory preserves more valid tracks through short "
          "occlusions than single-frame detection or conventional tracking — without creating "
          "an unacceptable number of false, ghost tracks.", 16, ICE, align=PP_ALIGN.CENTER,
          italic=True)
add_notes(s, "State the research question plainly and pause on it. This is the question the whole "
             "project is trying to answer, and the hypothesis is what we expect but haven't yet proven.")

# ---------------------------------------------------------------- Slide 4
s = add_slide(WHITE)
add_text(s, Inches(0.6), Inches(0.5), Inches(8), Inches(0.9),
          "Step One: Measuring the Baseline", 28, NAVY, bold=True, font=HEADER_FONT)
rows = [
    ("\U0001F4F7", "Dataset: nuScenes CAM_FRONT — 8 real occlusion sequences"),
    ("\U0001F9E0", "Model: pretrained YOLO nano (yolo26n) — prediction only, no training"),
    ("\U0001F5A5", "CPU only, image size 640, confidence threshold 0.05"),
    ("\U0001F501", "5 stages per object: before → first partial occlusion → full occlusion → "
                    "first reappearance → full appearance"),
]
y = Inches(1.7)
for emoji, text in rows:
    add_icon_row(s, Inches(0.6), y, Inches(6.5), emoji, text, size=15)
    y += Inches(1.05)

pic_left, pic_top, pic_w = Inches(7.5), Inches(1.7), Inches(5.2)
frame = add_rounded_box(s, pic_left - Inches(0.12), pic_top - Inches(0.12),
                         pic_w + Inches(0.24), Inches(3.15), ICE)
s.shapes.add_picture(
    r"C:\Users\HP\OneDrive\Desktop\srsi\results\annotated\sample_001\1_previous_no_occlusion.jpg",
    pic_left, pic_top, width=pic_w)
add_text(s, pic_left, pic_top + Inches(3.15), pic_w, Inches(0.5),
          "Real YOLO detections on a nuScenes frame", 12, MUTED, italic=True, align=PP_ALIGN.CENTER)
add_notes(s, "Walk through the method: we didn't build anything fancy yet, just measured how a plain, "
             "off-the-shelf detector behaves as an object goes through five occlusion stages. Point out "
             "the real annotated example image.")

# ---------------------------------------------------------------- Slide 5
s = add_slide(WHITE)
add_text(s, Inches(0.6), Inches(0.4), Inches(11), Inches(0.9),
          "Detection Collapses at Full Occlusion", 28, NAVY, bold=True, font=HEADER_FONT)

stages = ["No\nOcclusion", "First Partial\nOcclusion", "Full\nOcclusion",
          "First Partial\nAppearance", "Full\nAppearance"]
detection_rates = [1.0, 1.0, 0.125, 0.5, 1.0]
mean_conf = [0.774, 0.587, 0.029, 0.425, 0.845]

cd1 = CategoryChartData()
cd1.categories = stages
cd1.add_series("Detection rate", detection_rates)
gframe1 = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(1.5),
                              Inches(6.0), Inches(5.3), cd1)
chart1 = gframe1.chart
chart1.has_title = True
chart1.chart_title.text_frame.text = "Detection rate by stage"
chart1.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(15)
chart1.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
chart1.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
style_chart_simple(chart1, "1E2761", num_fmt="0%")

cd2 = CategoryChartData()
cd2.categories = stages
cd2.add_series("Mean confidence", mean_conf)
gframe2 = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(6.9), Inches(1.5),
                              Inches(6.0), Inches(5.3), cd2)
chart2 = gframe2.chart
chart2.has_title = True
chart2.chart_title.text_frame.text = "Mean confidence by stage (0 = undetected)"
chart2.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(15)
chart2.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True
chart2.chart_title.text_frame.paragraphs[0].runs[0].font.color.rgb = NAVY
style_chart_simple(chart2, "FFA630", num_fmt="0.00")

add_text(s, Inches(0.6), Inches(6.95), Inches(12.1), Inches(0.5),
          "Partial-occlusion and first-reappearance stages have small sample sizes (n=2-3) — "
          "interpret with caution.", 12, MUTED, italic=True, align=PP_ALIGN.CENTER)
add_notes(s, "Show the two charts together. Detection rate stays perfect through partial occlusion "
             "then falls off a cliff at full occlusion. Confidence tells a smoother story: it erodes "
             "before detection technically fails, and recovers once the object is visible again.")

# ---------------------------------------------------------------- Slide 6
s = add_slide(NAVY)
add_text(s, Inches(1), Inches(0.6), Inches(11.33), Inches(0.9),
          "What We Learned", 32, WHITE, bold=True, font=HEADER_FONT, align=PP_ALIGN.CENTER)

stats = [
    ("100% → 12%", "detection rate at full occlusion"),
    ("0.77 → 0.03", "mean confidence at full occlusion"),
    ("100%", "correct class whenever something was detected"),
]
col_w = Inches(3.8)
gap = Inches(0.3)
total_w = col_w * 3 + gap * 2
start_x = (SLIDE_W - total_w) // 2
for i, (big, small) in enumerate(stats):
    x = start_x + i * (col_w + gap)
    add_text(s, x, Inches(2.2), col_w, Inches(1.1), big, 42, AMBER, bold=True,
              font=HEADER_FONT, align=PP_ALIGN.CENTER)
    add_text(s, x, Inches(3.3), col_w, Inches(1.1), small, 15, ICE, align=PP_ALIGN.CENTER)

add_text(s, Inches(1.6), Inches(5.3), Inches(10.13), Inches(1.3),
          "The detector doesn't fade out gradually — it's either fully seen or fully gone. "
          "That gap is exactly what temporal memory can fill.", 18, WHITE, italic=True,
          align=PP_ALIGN.CENTER, line_spacing=1.2)
add_notes(s, "Summarize the three headline numbers. The key insight: confidence decays smoothly but "
             "detection itself is almost binary. That binary drop is the opportunity for a memory-based "
             "system to smooth over.")

# ---------------------------------------------------------------- Slide 7
s = add_slide(WHITE)
add_text(s, Inches(0.6), Inches(0.45), Inches(9), Inches(0.8),
          "The Proposed Fix: OATM", 28, NAVY, bold=True, font=HEADER_FONT)
add_text(s, Inches(0.6), Inches(1.15), Inches(9), Inches(0.4),
          "Occlusion-Adaptive Temporal Memory", 14, MUTED, italic=True)

rows = [
    ("\U0001F9E9", "Dual memory: appearance (last clear view) + motion (position & velocity)"),
    ("\U0001F6A6", "Occlusion classifier: VISIBLE / OCCLUDED / LOST / EXITED"),
    ("⏳", "Confidence decays smoothly over time — not a fixed countdown"),
    ("\U0001F6AB", "Anti-ghost termination stops phantom tracks from lingering"),
]
y = Inches(2.0)
for emoji, text in rows:
    add_icon_row(s, Inches(0.6), y, Inches(6.6), emoji, text, size=14.5, row_h=Inches(1.05))
    y += Inches(1.15)

# simple state-flow diagram on the right
box_w, box_h = Inches(2.3), Inches(0.7)
diag_x = Inches(8.1)
states = [("VISIBLE", NAVY), ("OCCLUDED", AMBER), ("LOST / EXITED", MUTED)]
ys = [Inches(2.1), Inches(3.5), Inches(4.9)]
centers = []
for (label, color), yy in zip(states, ys):
    shp = slide_shape = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, diag_x, yy, box_w, box_h)
    shp.adjustments[0] = 0.15
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = BODY_FONT
    centers.append((diag_x + box_w // 2, yy, yy + box_h))

for i in range(len(centers) - 1):
    cx, _, y_bottom = centers[i]
    _, y_top_next, _ = centers[i + 1]
    conn = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, cx, y_bottom, cx, y_top_next)
    conn.line.color.rgb = DARKGRAY
    conn.line.width = Pt(2)

add_text(s, diag_x, Inches(5.75), box_w, Inches(0.9),
          "reappearance → back to VISIBLE", 11, MUTED, italic=True, align=PP_ALIGN.CENTER)
add_notes(s, "Introduce OATM's core mechanism: it doesn't just keep every missing object alive, it "
             "explicitly classifies whether something is occluded, lost, or has exited frame, and "
             "decays confidence smoothly rather than on a fixed timer.")

# ---------------------------------------------------------------- Slide 8
s = add_slide(WHITE)
add_text(s, Inches(0.6), Inches(0.45), Inches(9), Inches(0.8),
          "How OATM Compares", 28, NAVY, bold=True, font=HEADER_FONT)

table_data = [
    ("Method", "How it works", "Main weakness"),
    ("Single-frame detection", "Looks only at the current image", "Hidden object instantly disappears"),
    ("SORT", "Predicts motion from recent pattern", "Struggles with turns or ego-motion"),
    ("ByteTrack", "Links strong and weak detections", "Still needs some visible evidence"),
    ("Fixed temporal memory", "Keeps missing objects for N frames", "May forget early or keep ghosts too long"),
    ("OATM (proposed)", "Appearance + motion + occlusion-aware decay", "More complex; needs careful tuning"),
]
n_rows, n_cols = len(table_data), 3
tbl_shape = s.shapes.add_table(n_rows, n_cols, Inches(0.6), Inches(1.5), Inches(12.1), Inches(5.2))
table = tbl_shape.table
table.columns[0].width = Inches(3.0)
table.columns[1].width = Inches(5.3)
table.columns[2].width = Inches(3.8)

for r, row in enumerate(table_data):
    for c, val in enumerate(row):
        cell = table.cell(r, c)
        cell.text = val
        cell.margin_left = Inches(0.15)
        cell.margin_right = Inches(0.15)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        run = para.runs[0]
        run.font.name = BODY_FONT
        if r == 0:
            run.font.bold = True
            run.font.size = Pt(14)
            run.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
        else:
            run.font.size = Pt(13)
            run.font.color.rgb = DARKGRAY
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xE0) if r == n_rows - 1 else WHITE
            if r == n_rows - 1:
                run.font.bold = True
                run.font.color.rgb = NAVY
add_notes(s, "Position OATM against the alternatives already used in the field. The point isn't that "
             "the others are bad — it's that none of them explicitly reason about *why* an object "
             "disappeared, which is what OATM adds.")

# ---------------------------------------------------------------- Slide 9
s = add_slide(NAVY)
add_text(s, Inches(1), Inches(0.6), Inches(11.33), Inches(0.8),
          "What's Next", 32, WHITE, bold=True, font=HEADER_FONT, align=PP_ALIGN.CENTER)

rows = [
    ("\U0001F9EA", "Build natural + controlled occlusion test sets from nuScenes"),
    ("\U0001F4CA", "Compare 5 systems: single-frame, SORT, ByteTrack, fixed memory, OATM"),
    ("\U0001F4CF", "Metrics: occluded-object recall, identity preservation, ghost-track rate, recovery time"),
    ("\U0001F3AC", "Split train / validation / test by scene, not by frame, to avoid leakage"),
]
y = Inches(1.9)
for emoji, text in rows:
    add_icon_row(s, Inches(1.6), y, Inches(10.1), emoji, text, size=16, color=WHITE,
                  circle_color=AMBER, row_h=Inches(0.95))
    y += Inches(1.05)

add_text(s, Inches(1.6), Inches(6.5), Inches(10.13), Inches(0.7),
          "Thank you — questions welcome", 20, AMBER, bold=True, align=PP_ALIGN.CENTER,
          font=HEADER_FONT)
add_notes(s, "Close with the roadmap: this baseline experiment was step one. Next comes building proper "
             "test sets, implementing OATM itself, and comparing it fairly against existing tracking "
             "methods using metrics that specifically capture occlusion handling and false-track risk.")

out_path = r"C:\Users\HP\OneDrive\Desktop\srsi\presentation\OATM_Project_Presentation.pptx"
prs.save(out_path)
print("Saved:", out_path)

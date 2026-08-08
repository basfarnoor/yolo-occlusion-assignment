# OATM paper-figure drawing brief

## Communication goal

Show, in one glance, how a causal camera-only tracker turns current visual
evidence and prior track memory into one of three explicitly labeled outcomes:
an observed track, a prediction-only hidden track, or a terminated track.

The figure must also make two scientific boundaries unmistakable:

1. A hidden-frame prediction is not presented as a current camera detection.
2. LiDAR-supported annotations, recorded ego pose, and nuScenes visibility
   labels are offline supervision or evaluation evidence, never online OATM
   input.

## Recommended composition

Use a two-column-wide figure (182 mm / 7.16 in) with three panels.

### Panel (a): online causal pipeline

Draw a dominant left-to-right flow:

1. **Camera frames up to time `t`** — show `I(t-1)` and `I(t)` and state that
   no future frame is used.
2. **Detector + features** — output box, class, confidence, and appearance
   embedding. Explicitly distinguish strong, weak, and absent evidence.
3. **Two-stage association** — match strong detections first, then use weak
   evidence to recover plausible track matches.
4. **OATM state controller** — combine detection strength, possible-occluder
   overlap, a camera-derived depth-order proxy, trajectory consistency,
   boundary evidence, track age, and uncertainty.
5. Split into three labeled branches:
   - **Observed update:** correct the motion state; refresh the appearance
     memory only from reliable strong evidence.
   - **Hidden-state recovery:** apply causal camera-motion compensation,
     predict the track state, search an uncertainty region, decay existence
     and identity confidence, and run anti-ghost checks.
   - **Lost / exited:** retire the track and emit no persistent object.
6. Label outputs by evidence source: `OBSERVED`, `PREDICTED_HIDDEN`, and
   `NO OUTPUT`.

Place a paired memory bank below the main flow:

- **Appearance memory `A(clear)`** stores the last clearly visible crop or
  embedding and is not overwritten while the object is hidden.
- **Motion memory `M(t-1)`** stores position, velocity, scale, age, time since
  the last reliable observation, localization uncertainty, existence
  confidence, and identity confidence.

Use dashed feedback arrows from memory into association and the state
controller. Use a separate camera-motion arrow into hidden-state recovery.

### Panel (b): track-state controller

Show the states as a compact transition graph:

- `OBSERVED_STRONG <-> OBSERVED_WEAK`
- `OBSERVED_WEAK -> PREDICTED_HIDDEN` when visual association fails but
  occlusion evidence supports persistence
- `PREDICTED_HIDDEN -> OBSERVED_STRONG/WEAK` when a reappearance candidate
  agrees with the predicted region, `A(clear)`, class, scale, and direction
- `PREDICTED_HIDDEN -> LOST` when uncertainty or persistence risk exceeds its
  stop rule
- `PREDICTED_HIDDEN -> EXITED` when the predicted path crosses the field of
  view boundary

### Panel (c): privileged offline evidence

Use a visually separate dashed enclosure. Connect nuScenes 3D boxes,
visibility labels, LiDAR-supported geometry, and recorded ego pose to projected
camera ground truth and evaluation metrics. Do not connect this lane to the
online inference path.

## Visual grammar

- Use rectangles for computational modules, a paired enclosure for memory,
  and explicit state labels for decisions.
- Preserve one primary left-to-right reading direction. Avoid a hub-and-spoke
  layout because it hides execution order.
- Use solid blue for current visual observations, dashed amber for
  prediction-only persistence, dotted gray for termination, and purple only
  for offline privileged evidence.
- Pair every color with text and line style so the figure remains meaningful
  when printed in grayscale.
- Keep arrows behind nodes and route feedback outside the main causal path.
- Avoid shadows, gradients, decorative icons, and long prose inside boxes.
- Use 8–10 pt sans-serif text at final print size and approximately 0.6–1.0 pt
  strokes.

## Export requirements

- Keep an editable vector master in SVG.
- Export the submission figure as PDF (or EPS if the venue requires it).
- Size the graphic at its final paper width before checking legibility.
- If a raster fallback is required, use at least 300 dpi for color or grayscale
  graphics and 600 dpi for monochrome line art.
- Check a grayscale printout and a one-column thumbnail before submission.

IEEE recommends vector graphics for resizing, identifies 3.5 in and 7.16 in as
typical one- and two-column widths, and recommends more than 300 dpi for color
or grayscale raster graphics and more than 600 dpi for monochrome line art:
[IEEE resolution and size guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/).
Its accessibility guidance also recommends combining color with shape or line
style rather than relying on color alone:
[IEEE graphics guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/).

## Published figure conventions used

- **MeMOT, Figure 2:** a strong left-to-right architecture, a clearly bounded
  memory module, and a visible memory-update feedback path. This supports using
  one dominant causal flow with memory underneath rather than placing OATM at
  the center of four peer boxes.
  [MeMOT paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html)
- **PermaTrack, Figure 3:** sequential frames enter a backbone, recurrent memory
  remains visually distinct, and outputs are decoded at the right. This
  supports showing temporal input, memory, and visibility/output semantics as
  separate visual roles.
  [PermaTrack paper](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html)
- **ByteTrack, Figure 2 and method description:** high- and low-confidence
  detections are visually differentiated and the second association stage is
  motivated by recovering true occluded objects from weak detections. This
  supports the strong/weak/absent evidence encoding before OATM's occlusion
  decision.
  [ByteTrack paper](https://arxiv.org/abs/2110.06864)

These conventions are synthesized for OATM; the figure does not copy any one
paper's architecture or artwork.

## Draft caption

**Figure X. Occlusion-Adaptive Temporal Memory (OATM).** At time `t`, the
camera-only causal pipeline combines current-frame strong or weak detections
with per-track appearance and motion memories. An occlusion-aware controller
labels each track as observed, prediction-only hidden, lost, or exited. Hidden
tracks use camera-motion-compensated state prediction, uncertainty-region
search, calibrated confidence decay, and anti-ghost termination; their output
is explicitly marked as a temporal prediction. Reappearance candidates are
matched using predicted location, uncertainty, the last clear appearance
anchor, class, scale, and motion direction. nuScenes LiDAR-supported boxes,
visibility labels, and recorded ego pose are used only for offline supervision,
oracle diagnostics, or evaluation and are not supplied to online OATM.

## LaTeX inclusion

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/oatm_pipeline.pdf}
  \caption{Occlusion-Adaptive Temporal Memory (OATM). At time $t$, the
  camera-only causal pipeline combines current visual evidence with per-track
  appearance and motion memories; prediction-only hidden outputs are explicitly
  distinguished from current observations. Privileged nuScenes evidence is
  used offline only.}
  \label{fig:oatm_pipeline}
\end{figure*}
```

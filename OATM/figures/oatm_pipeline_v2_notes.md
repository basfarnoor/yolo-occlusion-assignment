# OATM compact paper figure — version 2

## Design choices

This revision intentionally uses a smaller visual vocabulary than version 1:

- One real three-frame nuScenes sequence at the top.
- One straight left-to-right method path at the bottom.
- One short vertical memory connection.
- No curved feedback loops, separate state-machine panel, or bullet-heavy
  component descriptions.
- Solid cyan boxes denote current visual evidence. Dashed amber boxes denote a
  prediction-only hidden region. Gray denotes termination.

The three images are unmodified CAM_FRONT frames from `sample_003`. The target
annotations and predicted hidden region are explanatory overlays. They are not
presented as measured OATM outputs because the proposed system has not yet been
implemented and evaluated.

## Published conventions used

- **OC-SORT, Figure 2:** uses real temporal frames, solid detection/track boxes,
  dashed prediction boxes, and a horizontal temporal reading direction. The
  OATM revision adopts the real-frame sequence and solid-versus-dashed evidence
  distinction while removing OC-SORT's denser internal routing.
  [OC-SORT paper](https://openaccess.thecvf.com/content/CVPR2023/html/Cao_Observation-Centric_SORT_Rethinking_SORT_for_Robust_Multi-Object_Tracking_CVPR_2023_paper.html)
- **PermaTrack, Figure 3:** places sequential frames before a compact recurrent
  memory and output path. This supports making the real temporal example the
  dominant visual evidence rather than surrounding the method with many
  abstract boxes.
  [PermaTrack paper](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html)
- **MeMOT, Figure 2:** keeps the main processing direction horizontal and
  identifies memory as a distinct source of track information. The OATM figure
  retains one explicit dual-memory connection.
  [MeMOT paper](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html)
- **IEEE author guidance:** favors vector output, standard final-width sizing,
  and encodings that remain understandable without color.
  [IEEE resolution and size guidance](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/resolution-and-size/)

## Draft caption

**Figure X. Compact overview of Occlusion-Adaptive Temporal Memory (OATM).**
(a) A real nuScenes CAM_FRONT sequence illustrates the intended OATM states.
The target is visually observed before occlusion, represented by a
prediction-only region while hidden, and reconnected to its original identity
after reappearance. These overlays illustrate the proposed behavior and are not
measured OATM results. (b) At time `t`, the causal camera-only pipeline first
associates strong and weak detections with active tracks. An occlusion gate then
selects an observed update, uncertainty-aware hidden-state prediction, or track
termination. The update reads per-object appearance memory anchored to the last
reliable observation and motion memory containing the previous state and its
uncertainty. Outputs explicitly distinguish current observations from temporal
predictions. nuScenes LiDAR-supported annotations are used only for offline
evaluation and are not supplied to OATM at inference.

## Recommended use

Use `oatm_pipeline_v2.pdf` as a two-column-wide `figure*`. Keep the sentence
stating that panel (a) is illustrative until actual OATM outputs replace the
overlays.

```latex
\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/oatm_pipeline_v2.pdf}
  \caption{Compact overview of Occlusion-Adaptive Temporal Memory (OATM).
  Solid cyan boxes indicate current visual evidence; dashed amber boxes indicate
  prediction-only hidden states. The real-frame overlays in (a) illustrate the
  proposed behavior and are not measured OATM outputs.}
  \label{fig:oatm_pipeline}
\end{figure*}
```

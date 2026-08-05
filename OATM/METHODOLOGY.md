# Occlusion-Adaptive Temporal Memory for Autonomous Vehicle Perception

## Project concept

**Working title:** Occlusion-Adaptive Temporal Memory for Object Persistence in Autonomous Vehicle Perception

**Original title:** Temporal Frame Recovery for Occluded Object Detection in Autonomous Vehicle Perception

When an object is temporarily blocked from a vehicle camera by another road user, many single-frame detection systems behave as if the hidden object has disappeared. This project investigates whether causal temporal information from previous frames can preserve the object's identity and estimate its location until it becomes visible again.

The phrase *frame recovery* can imply reconstructing missing image pixels. The proposed system instead recovers an object's state: its identity, position, velocity, visibility status, and confidence. For that reason, *object persistence* or *temporal memory* describes the research more precisely.

## Research question

Can an occlusion-aware, uncertainty-calibrated temporal memory system improve object recall and identity continuity during temporary camera occlusions, compared with single-frame detection and conventional tracking methods, without producing an unacceptable number of false persistent tracks?

## Hypothesis

Starting from a ByteTrack-style baseline, explicit separation of strong visual
evidence, weak visual evidence, and prediction-only persistence, followed by
camera-derived motion compensation and uncertainty-aware termination, should
improve the tradeoff between short-occlusion recovery and false persistence.
Appearance memory and explicit occlusion classification are secondary hypotheses
that must earn their place through ablation rather than being assumed beneficial.

## Why using previous frames is not sufficient novelty

Temporal information is already widely used in video detection and tracking:

- A **single-frame detector** processes each image independently and loses objects when visual evidence disappears.
- **Tracking-by-detection** methods associate detections between frames and use motion models to bridge short gaps.
- **Temporal feature aggregation** aligns and combines visual features from nearby frames.
- **Memory-based trackers** preserve object representations over longer time intervals.
- **Streaming bird's-eye-view systems** maintain temporal scene representations for autonomous-vehicle perception.
- **Multisensor and cooperative perception** systems use LiDAR, radar, or information from other vehicles when a camera view is obstructed.

Flow-Guided Feature Aggregation established that motion-aligned features from neighboring frames can improve video object detection. More recent systems such as MeMOT, MeMOTR, PermaTrack, and mask-guided temporal aggregation also maintain temporal object information. Therefore, the contribution cannot simply be "YOLO plus previous frames" or "YOLO plus a tracker."

The proposed contribution is an **occlusion-specific decision and memory system** that determines:

1. Whether a missing object is probably occluded, outside the image, or genuinely lost.
2. Which past appearance should be retained as the object's reliable visual memory.
3. How its location should be predicted while hidden.
4. How prediction confidence should change with time and uncertainty.
5. When the system must terminate the prediction to avoid a ghost object.

## Evidence from the four preliminary assignments

The completed assignments motivate OATM but do not validate the complete method.
YOLO frequently lost fully hidden targets in eight selected examples. Frozen
last-seen memory preserved a report but produced severe localization drift in
five examples. A controlled SORT study showed that constant-velocity prediction
can help selected smooth, moving tracks, but its original evaluation was not an
independent end-to-end tracker validation. A small ByteTrack study showed that
weak detections can improve continuity, including a held-out natural-event
result of 3/6 recoveries versus 2/6 for high-confidence SORT. These are pilot
signals, not effect-size estimates. They justify staged experiments in which
each new OATM component is tested independently.

## Comparison with existing methodological families

| Method | Main behavior | Limitation under occlusion |
|---|---|---|
| Single-frame detector | Detects each image independently | Immediately loses a fully hidden object |
| Kalman filter or SORT | Extrapolates location from recent motion | Assumes simple motion and can mishandle ego-motion or turns |
| ByteTrack-style tracking | Associates both strong and weak detections | Still benefits from visible evidence and does not explicitly reason about the occluder |
| Temporal feature aggregation | Combines aligned features from multiple frames | Can mix stale object features with irrelevant background |
| Transformer memory tracking | Maintains long-term track representations | More computationally demanding and not necessarily calibrated for hidden-object uncertainty |
| LiDAR, radar, or V2X fusion | Obtains evidence unavailable to the blocked camera | Requires additional sensors, vehicles, or infrastructure |
| Proposed OATM system | Activates dual memory and prediction specifically during likely occlusions | Must control uncertainty and false persistence carefully |

## Proposed methodology: OATM

The proposed method is called **OATM: Occlusion-Adaptive Temporal Memory**. It is a causal system: at time `t`, it may use the current frame and earlier frames, but not future frames. This keeps the experiment relevant to real-time autonomous driving.

### 1. Current-frame detection

A pretrained detector processes the current camera image and produces:

- Bounding boxes.
- Object classes.
- Detection confidence.
- Object-level appearance embeddings or visual features.

A pretrained YOLO-family or RT-DETR model is sufficient. Training a large detector from scratch is not necessary because the research contribution is the temporal recovery mechanism.

Low confidence is treated as weak visual support, not proof of occlusion. It can
also result from distance, blur, illumination, truncation, class confusion, or a
false box. The detector therefore runs with a documented low confidence floor,
and the temporal system records whether evidence is strong, weak, or absent.

### 2. Per-object dual memory

For every active track, OATM stores two complementary memories.

**Appearance memory** contains a feature representation from the most recent frame in which the object was clearly visible. The system should not continually overwrite this anchor with features dominated by an occluding truck, barrier, or neighboring vehicle.

**Motion memory** contains:

- Recent image or world coordinates.
- Estimated velocity and direction.
- Bounding-box size changes.
- Track age and time since the last reliable observation.
- Prediction uncertainty.

This design separates the questions "What object was this?" and "Where should it be now?"

### 3. Occlusion-event classification

When a detection weakens or disappears, the system estimates whether the object is hidden rather than automatically keeping every missing track alive.

An occlusion score can combine:

\[
O_t = w_1(1-C_t) + w_2 I_{\text{overlap}} + w_3 D_{\text{ordering}} + w_4 T_{\text{consistency}}
\]

where:

- \(C_t\) is current detection confidence.
- \(I_{\text{overlap}}\) measures overlap with a potential foreground occluder.
- \(D_{\text{ordering}}\) is a camera-derived depth-ordering proxy, not privileged
  nuScenes depth at inference.
- \(T_{\text{consistency}}\) measures agreement between the disappearance and the previous trajectory.

This expression is an initial feature inventory, not a calibrated probability
model. Every term must be computable causally from camera frames and tracker
history. The classifier must be evaluated separately on true occlusions,
field-of-view exits, ordinary detector misses, poor viewing conditions, and
false-positive tracks.

The state machine assigns one of the following states:

- `OBSERVED_STRONG`: supported by a current high-confidence detection.
- `OBSERVED_WEAK`: supported by an associated current low-confidence detection.
- `PREDICTED_HIDDEN`: no current detection, but persistence is supported by
  temporal and occlusion evidence.
- `LOST`: insufficient evidence to continue the track.
- `EXITED`: predicted to have left the camera field of view.

The state distinction is important for safety. A completely hidden object should be reported as a temporal prediction, not as something the camera currently sees.

### 4. Ego-motion-compensated state recovery

If a track is classified as `PREDICTED_HIDDEN`, OATM:

1. Predicts its location from motion history.
2. Compensates for camera motion estimated causally from consecutive camera
   images, excluding tracked foreground regions where practical.
3. Produces an uncertainty region around the predicted location.
4. Searches that region for weak visual evidence.
5. Compares candidate features with the last clear appearance memory.
6. Updates the predicted box, identity confidence, and uncertainty.

Recorded nuScenes ego pose is permitted only as privileged offline supervision,
evaluation evidence, or a clearly labeled oracle diagnostic. It is never an
input to the headline camera-only method. The first implementation compares a
stationary model and timestamp-aware constant-velocity Kalman model before
adding camera-motion compensation. A learned motion model is outside the MVP.

### 5. Adaptive confidence decay

OATM must not collapse detector confidence, object-existence probability,
identity confidence, and localization uncertainty into one score. It records
them separately. A simple initial persistence model uses a non-negative hazard:

\[
h_t = \beta + \alpha\,\Delta U_t, \qquad
P^{\mathrm{exist}}_t = P^{\mathrm{exist}}_{t-1}\exp(-h_t\Delta t)
\]

where:

- \(P^{\mathrm{exist}}_t\) is the estimated probability that the object remains
  relevant in the camera scene.
- \(\Delta U_t\) is incremental localization-uncertainty growth.
- \(\Delta t\) is elapsed time in seconds.
- \(\alpha\) and \(\beta\) are tuned on development scenes only.

Confidence cannot increase without new evidence. Calibration is measured on
held-out scenes, and fixed-lifetime baselines are compared at matched ghost-risk
operating points. This formulation is an MVP policy, not a claim that the
hazard equation is intrinsically novel.

### 6. Reappearance and identity recovery

When an object becomes visible again, OATM compares the new detection with:

- The predicted location and uncertainty region.
- The stored clear-frame appearance embedding.
- The expected class, scale, and motion direction.

The system then either reconnects the original identity or starts a new track. Successful reconnection without an identity switch is one of the main experimental outcomes.

The promoted relational implementation also retains a mature but unsupported
identity internally for one silent frame after its ordinary miss grace expires.
This `DORMANT` state emits no box and has zero output existence confidence; it
can reconnect only through a stricter relation-style association threshold. It
is therefore an identity-recovery opportunity, not an extra prediction frame.

### 7. Anti-ghost termination

Temporal recovery creates a safety risk if a system continues predicting an object that has actually left. OATM terminates or downgrades a track when:

- Its outward-moving predicted path reaches an image boundary and no more than
  the development-selected fraction of its box remains visible; touching the
  boundary alone is not an exit.
- Its uncertainty exceeds a threshold.
- The object should have reappeared from behind an occluder but does not.
- The predicted path conflicts with scene geometry or the visible occluder's motion.
- Appearance matching fails when a candidate object appears.
- Persistence confidence falls below a calibrated threshold.

The balance between occluded-object recall and ghost-track rate is central to the project.

## Proposed novelty

The defensible contribution is the combination of:

- Explicit classification of occlusion, loss, and field-of-view exit.
- Appearance memory anchored to the last clearly visible observation.
- Ego-motion-compensated trajectory prediction.
- Occlusion-dependent confidence decay.
- Uncertainty-aware track termination.
- Separate strong-observed, weak-observed, and predicted-hidden output states.
- An occlusion-specific evaluation subset derived from nuScenes.
- Controlled experiments that measure both successful persistence and harmful ghost predictions.

This should initially be described as a **proposed novel combination**. Claiming that the method is globally unprecedented would require a more exhaustive literature review.

## Dataset methodology with nuScenes

nuScenes is suitable because it provides:

- 1,000 driving scenes of approximately 20 seconds each.
- Six cameras plus LiDAR, radar, GPS, and IMU.
- Persistent object instance identifiers within each scene.
- 3D bounding boxes and object visibility categories.
- Ego-pose and camera calibration information.
- Camera images at approximately 12 Hz, with official annotated keyframes at 2 Hz.

### Recommended data pipeline

1. Start with the `CAM_FRONT` stream.
2. Project annotated 3D boxes into the front-camera image.
3. Connect an object's boxes over time using its `instance_token`.
4. Identify visibility transitions such as high visibility, low visibility, and high visibility again.
5. Extract candidate natural-occlusion clips.
6. Manually inspect a smaller evaluation subset to confirm that the visibility change is caused by an actual occluder.
7. Split training, validation, and testing by scene rather than by frame.

Splitting by scene prevents nearly identical neighboring frames from leaking between training and evaluation.

### Camera-only input with privileged ground truth

The deployed system should receive camera images only. LiDAR-supported 3D annotations, visibility labels, and ego poses may be used to construct ground truth and evaluate whether a hidden object physically remains in the scene. This must be reported clearly: the additional sensor information supervises or evaluates the experiment but is not supplied as live perception evidence to the final camera-only method.

### CAM_FRONT LiDAR-supported offline evaluation protocol

The online and offline paths must be implemented as separate stages. First,
the detector and tracker run causally over every `CAM_FRONT` frame and save
their outputs without loading projected annotations. Only after those outputs
exist may the evaluator load official nuScenes 3D annotations, visibility,
calibration, instance tokens, ego pose, or LiDAR/radar point counts.

Headline localization and identity metrics use only official annotated
keyframes. Unannotated camera sweeps are still processed online but are never
treated as empty ground truth, and interpolated boxes are not official
headline labels. Matching is class-aware and one-to-one under fixed IoU gates;
observed detections and predicted-hidden outputs are reported separately.

The primary population retains zero-LiDAR-point and truncated annotations so
that difficult occlusions are not removed selectively. Visibility, distance,
point support, and truncation are sensitivity strata. False outputs, identity
switches, fragmentation, and unsupported-keyframe tracks are reported along
with recall. Because annotations are sparse, an unsupported-keyframe track is
explicitly labeled as a ghost proxy rather than a verified sweep-frame ghost
duration. Whole scenes, never neighboring frames, define development and
validation populations.

### Natural and controlled occlusions

Official annotations at 2 Hz may miss very short occlusions. The evaluation should therefore contain two separately reported subsets:

**Natural occlusions** come from real nuScenes sequences and are manually validated. They demonstrate real-world relevance but can be difficult to label consistently.

**Controlled occlusions** are created by placing realistic vehicle-shaped masks or segmented foreground objects over visible targets for known durations and severities. They provide exact control over when an object becomes hidden and when it returns.

Controlled occlusions should vary by:

- Occlusion duration.
- Percentage of the target covered.
- Target class and size.
- Occluder class and motion.
- Relative target and camera motion.

Results from natural and controlled occlusions must not be merged into a single unexplained score.

Controlled visual occlusion must modify the input pixels and rerun the detector;
editing confidence values or deleting detection rows is instead called a
`detector-intervention` experiment. Detector interventions remain useful for
isolating tracker behavior, but they do not demonstrate visual occlusion
handling. Every controlled image records its source frame, target, mask,
coverage, duration, seed, and transformation parameters, while original
nuScenes files remain untouched.

## Experimental comparison

The experiment should compare at least five systems:

1. Pretrained single-frame detector.
2. Detector plus SORT or a Kalman-filter tracker.
3. Detector plus ByteTrack.
4. A fixed-window temporal aggregation baseline.
5. The complete OATM method.

Large transformer or bird's-eye-view models can be discussed as related work without being reimplemented. The experimental comparison should remain computationally achievable and focused on the proposed mechanism.

## Evaluation metrics

Standard detection accuracy alone is insufficient. Report:

- **Occluded-object recall:** Fraction of hidden ground-truth objects maintained correctly.
- **Visible-object precision and recall:** Confirms that temporal memory does not damage ordinary detection.
- **Identity preservation:** Fraction of reappearing objects that retain the same identity.
- **Identity switches:** Number of times a track is assigned to the wrong object.
- **Localization error during occlusion:** Distance between predicted and ground-truth positions.
- **Ghost-track rate:** Predictions retained after the object has exited or should be considered lost.
- **Maximum recovered occlusion duration:** Longest gap bridged reliably.
- **Recovery time:** Frames required to reconnect an object after reappearance.
- **Confidence calibration:** Whether predicted confidence reflects actual correctness.
- **Runtime:** Frames per second or latency per frame.

Results should also be stratified by object class, visibility level, distance, and occlusion duration.

The primary presentation is a tradeoff rather than a single recall score:

- Hidden-object recall versus ghost frames per scene or track.
- Identity preservation versus wrong-object association rate.
- Localization error versus occlusion duration.
- Risk-coverage and calibration curves as persistence thresholds vary.

A ghost is reported both as an event and a duration: a predicted-hidden output
that does not correspond to the intended ground-truth instance, or that remains
after a verified exit/loss condition. Wrong-object associations, initial false
tracks, and stale post-exit predictions are counted separately. Frame rows are
never treated as independent objects; uncertainty is computed at track/event
level and clustered by scene where sample size permits.

## Ablation study

An ablation study isolates which components create the improvement:

| Variant | Question answered |
|---|---|
| Motion memory only | How much can trajectory prediction recover? |
| Appearance memory only | How much does a clear visual anchor preserve identity? |
| Dual memory | Are appearance and motion complementary? |
| No ego-motion compensation | How much does observer movement matter? |
| No occlusion classifier | Is explicit occlusion reasoning better than keeping every missing track? |
| Fixed confidence lifetime | Does adaptive uncertainty-based decay help? |
| No anti-ghost mechanism | What recall is gained at the cost of false persistence? |
| Complete OATM | What is the combined result? |

This study is more scientifically persuasive than reporting only that the final model outperforms the single-frame detector.

## Difficulty assessment

| Scope | Difficulty | Assessment |
|---|---:|---|
| Pretrained detector plus basic tracker | 5/10 | Feasible, but offers limited novelty |
| Recommended OATM post-processing system | **7.5/10** | Ambitious but realistic and experimentally defensible |
| Learned temporal feature-fusion network | 8.5/10 | Requires greater training, compute, and debugging |
| End-to-end multi-camera 3D temporal transformer | 9.5/10 | Likely too broad for the intended project |

The recommended version uses a pretrained detector and concentrates original work on:

- Building an occlusion-specific dataset and evaluation protocol.
- Implementing dual temporal memory.
- Distinguishing occlusion from exit or ordinary detector failure.
- Quantifying uncertainty.
- Preventing ghost tracks.
- Conducting controlled baselines and ablations.

The most difficult part is not obtaining images. It is constructing trustworthy occlusion ground truth and demonstrating that improved persistence does not simply come from predicting stale objects for longer.

## Recommended implementation scope

### Minimum viable study

- nuScenes `CAM_FRONT` only.
- Cars and pedestrians only.
- Pretrained detector.
- Detector-only, static-memory, SORT, and ByteTrack baselines on identical
  observations.
- Explicit `OBSERVED_STRONG`, `OBSERVED_WEAK`, `PREDICTED_HIDDEN`, `LOST`, and
  `EXITED` outputs.
- Timestamp-aware stationary and constant-velocity motion baselines with
  localization uncertainty.
- A small rule-based occlusion/exit/loss gate using only camera-derived evidence.
- Fixed-lifetime and uncertainty-aware termination compared at matched risk.
- Natural and controlled test subsets.

Camera-derived ego-motion and appearance memory enter only after the simpler
motion/state/termination MVP passes its gates. This ordering prevents a large
combined system from hiding which component caused an improvement or failure.

### Optional extension

After the minimum system works, train a small model to predict:

- Probability that a missing track is occluded.
- Persistence confidence.
- Correction to the motion-predicted box.

This creates a learned extension without requiring an end-to-end video transformer.

## Expected scientific contribution

The strongest project claim would not be "our tracker remembers objects." It would be:

> An occlusion-aware, uncertainty-calibrated temporal memory improves camera-only object persistence through short autonomous-driving occlusions, while explicit state classification and anti-ghost termination control the safety cost of retaining invisible objects.

The project is successful even if the proposed method does not dominate every metric. A rigorous result showing the tradeoff between recovered occlusions and false persistence would still be scientifically useful.

## References

1. Caesar, H. et al. [nuScenes: A Multimodal Dataset for Autonomous Driving](https://arxiv.org/abs/1903.11027).
2. nuScenes. [Dataset overview](https://www.nuscenes.org/).
3. nuScenes. [Data format and visibility annotations](https://www.nuscenes.org/data-format).
4. Zhu, X. et al. [Flow-Guided Feature Aggregation for Video Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html).
5. Wang, S. et al. [Fully Motion-Aware Network for Video Object Detection](https://openaccess.thecvf.com/content_ECCV_2018/html/Shiyao_Wang_Fully_Motion-Aware_Network_ECCV_2018_paper.html).
6. Cai, J. et al. [MeMOT: Multi-Object Tracking With Memory](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_MeMOT_Multi-Object_Tracking_With_Memory_CVPR_2022_paper.html).
7. Gao, R. and Wang, L. [MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html).
8. Tokmakov, P. et al. [Learning to Track With Object Permanence](https://openaccess.thecvf.com/content/ICCV2021/html/Tokmakov_Learning_To_Track_With_Object_Permanence_ICCV_2021_paper.html).
9. Hashmi, K. A. et al. [Beyond Boxes: Mask-Guided Spatio-Temporal Feature Aggregation for Video Object Detection](https://openaccess.thecvf.com/content/WACV2025/html/Hashmi_Beyond_Boxes_Mask-Guided_Spatio-Temporal_Feature_Aggregation_for_Video_Object_Detection_WACV_2025_paper.html).

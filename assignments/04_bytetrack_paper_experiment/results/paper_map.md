# Paper Map: ByteTrack

**Paper:** Yifu Zhang, Peize Sun, Yi Jiang, Dongdong Yu, Fucheng Weng, Zehuan Yuan,
Ping Luo, Wenyu Liu, Xinggang Wang,
["ByteTrack: Multi-Object Tracking by Associating Every Detection Box,"](https://arxiv.org/abs/2110.06864)
ECCV 2022. DOI: [10.1007/978-3-031-20047-2_1](https://doi.org/10.1007/978-3-031-20047-2_1).
Authors' code: [github.com/FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack).

This map is written for a beginner. It explains what the paper's terms mean, then
shows a small pipeline and connects every step back to the paper.

## Terms, explained simply

- **Object detection** — a single-frame guess of "what is this and where is its
  box," with no memory of earlier frames. YOLO from Assignments 1-3 is a detector.
- **Detection confidence** — a 0-1 number the detector attaches to a box, roughly
  "how sure am I this box is real." Higher is more sure.
- **High-confidence detection** — a box whose confidence is at or above a chosen
  cutoff (the paper's `high_score_threshold` on unmatched detections). Treated as
  trustworthy evidence.
- **Low-confidence detection** — a box whose confidence sits below the high cutoff
  but at or above a lower floor. The paper's central claim is that many of these
  are still real, partly hidden objects rather than noise.
- **Detection floor** — the absolute minimum confidence kept at all; anything
  below it is discarded before ByteTrack ever sees it. This is different from the
  high-score threshold: the floor decides what exists at all, the threshold
  decides what is trusted immediately.
- **Track and track ID** — a **track** is one object followed across frames; its
  **track ID** is the number that says "this box in frame 10 is the same physical
  object as that box in frame 9."
- **Tracklet** — the paper's word for a track: the short trajectory (sequence of
  boxes) belonging to one track ID so far.
- **Tracked / lost / removed track states** — three stages of a track's life.
  *Tracked*: matched recently, considered active. *Lost*: unmatched this frame,
  kept alive for a limited time in case it returns. *Removed*: unmatched for too
  long; deleted for good.
- **Kalman prediction** — before matching, each track guesses ("predicts") where
  its box should be this frame, using its past motion (same constant-velocity
  idea as Assignment 3's SORT tracker).
- **IoU** (Intersection over Union) — how much two boxes overlap, from 0 (no
  overlap) to 1 (identical). ByteTrack uses IoU to decide if a predicted track
  location and a detection box are "close enough" to be the same object.
- **Hungarian assignment** — an algorithm that picks the best one-to-one pairing
  between two lists (here: predicted track locations and detections) given a cost
  for every possible pair, so no detection is double-claimed.
- **First association** — round 1 of matching: only high-confidence detections are
  matched against all active/predicted tracks.
- **Remaining unmatched track** — a track that predicted a location but found no
  high-confidence detection close enough in round 1. It is *not* deleted yet — it
  gets a second chance.
- **Second association** — round 2: the tracks left over from round 1 are matched
  against the low-confidence detections (using the same IoU + Hungarian idea).
  This is the paper's core contribution, "BYTE."
- **Track initialization** — starting a brand-new track ID. ByteTrack only does
  this from high-confidence detections that remain unmatched after round 1 —
  never from a low-confidence detection, since an unmatched weak box is too
  likely to be background noise or a false detection to trust as a new object.
- **Track buffer** — how many consecutive frames a track is allowed to stay
  "lost" (unmatched) before it becomes "removed." This is what lets a track
  survive a short gap and reconnect later.
- **False positive** — a detection box that does not correspond to a real object
  (e.g., the detector mistook shadow or clutter for something).
- **False negative** — a real object that produced no detection box at all.
- **Identity switch** — a track's ID incorrectly jumps to a different physical
  object, or two tracks swap identities.
- **Trajectory fragmentation** — one real object's history gets split into
  multiple separate track IDs instead of staying as one continuous track.

## Everyday analogy

> A teacher first matches students to clearly readable name tags. If someone is
> still unmatched, the teacher takes a second look at blurry name tags and uses
> where each student was standing to decide whether a blurry tag belongs to an
> existing student. A blurry tag may help identify someone already known, but it
> should not automatically create a brand-new student record.

The "clearly readable name tags" are high-confidence detections. The "blurry name
tags" are low-confidence detections. "Where each student was standing" is the
Kalman motion prediction used to judge whether a blurry tag is plausible for that
particular track.

## Pipeline

```text
camera frame
    -> YOLO detections with confidence scores
    -> split detections into high and low groups
    -> predict existing track locations
    -> first match: tracks with high-score detections
    -> second match: remaining tracks with low-score detections
    -> update matched tracks
    -> mark unmatched tracks lost or remove them after the buffer
    -> start new tracks only from unmatched high-score detections
    -> output current tracked objects with IDs
```

| Pipeline step | Where the same idea appears in the paper |
|---|---|
| Split detections into high/low groups | The paper's detection-score thresholding step feeding the BYTE association algorithm (Sec. 3, "BYTE: a simple yet effective association method") |
| Predict existing track locations | Kalman-filter motion prediction, used the same way as SORT/DeepSORT-style trackers the paper builds on |
| First match: high-score vs. tracks | BYTE's first association stage (high-confidence matching) |
| Second match: remaining tracks vs. low-score detections | BYTE's second association stage — the paper's central contribution, motivated by the claim that low-score boxes often "recover true objects and filter out background" |
| Update matched tracks | Standard tracked-state update after either association round |
| Mark unmatched tracks lost / remove after buffer | Track-buffer / lost-removed lifecycle described alongside the association method |
| Start new tracks only from unmatched high-score detections | The paper explicitly does not initialize new tracks from low-score detections, to avoid trusting weak/noisy boxes as brand-new objects |
| Output current tracked objects with IDs | Final tracked output, evaluated in the paper with MOTA/IDF1/HOTA on MOT17/MOT20 |

## SORT-style baseline vs. ByteTrack

| Question | SORT-style baseline | ByteTrack |
|---|---|---|
| Which detections are considered? | High-confidence only | High first, then low |
| Number of association rounds | One | Two |
| Can a low-score box update an existing track? | No | Yes, in the second round |
| Can an unmatched low-score box start a new track? | No | No |
| Can either method see a fully hidden object? | No | No |

## What the paper reports (for context, not something this assignment reproduces)

The paper reports 80.3 MOTA, 77.3 IDF1, and 63.1 HOTA on the MOT17 test set, and
states that adding the BYTE association strategy to nine different existing
trackers improved their IDF1 scores by roughly 1 to 10 points — evidence that the
"keep low-confidence boxes, match them second" idea is a generic improvement, not
something that only works with one specific detector or tracker.

**Task 1 is complete when:** a beginner can explain why ByteTrack uses two
matching rounds and why it does not trust every weak box equally. The pipeline
table above traces every step to the corresponding idea in the paper.

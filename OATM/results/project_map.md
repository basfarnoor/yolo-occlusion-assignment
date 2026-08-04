# OATM Project Map (Beginner's Guide)

This page explains what OATM is trying to do, in plain language, with no code.
If you've done Assignments 1-4, several of these terms will already feel
familiar — this page connects them to the bigger research project.

## Terms, explained simply

- **Object detection** — a single-frame guess of "what is this, and where is
  its box," with no memory of earlier frames. This is what plain YOLO does
  (Assignment 1).
- **Bounding box** — the rectangle drawn around an object, given as its
  left/top/right/bottom pixel coordinates.
- **Confidence score** — a 0-1 number the detector attaches to a box, roughly
  "how sure am I this box is real."
- **Strong and weak detection** — a *strong* detection has high confidence and
  is trusted immediately. A *weak* detection has low confidence — it might
  still be a real, partly-hidden object (this was the whole idea behind
  Assignment 4's ByteTrack experiment), or it might be noise. OATM treats weak
  detections as a hint, not proof.
- **Track and track ID** — a *track* is one object followed across many
  frames; its *track ID* is the number saying "this box in frame 10 is the
  same physical object as that box in frame 9."
- **Temporal memory** — anything the system remembers about an object from
  earlier frames (its last position, its speed, what it looked like) so it
  isn't starting from zero every single frame.
- **Occlusion** — a real object is still physically present in the scene, but
  something else (a bus, a pole, another car) is blocking the camera's view of
  it right now.
- **Ordinary detector miss** — the object was actually visible, but the
  detector simply failed to notice it (bad lighting, an odd angle, a model
  limitation) — not because anything was blocking it.
- **Field-of-view exit** — the object didn't get hidden, it genuinely drove
  or walked out of the camera's picture (turned a corner, went behind the
  car, etc.). A system that keeps "remembering" an object that has actually
  left is being unsafe, not helpful.
- **Motion prediction** — guessing where an object should be *right now*
  based on how it was moving a moment ago (its recent speed and direction).
  This is the idea Assignment 3's SORT experiment tested.
- **Localization uncertainty** — a measure of how unsure the system is about
  a *predicted* (not observed) position. This should grow the longer an
  object stays hidden, because a guess about where something is gets less
  reliable the longer you haven't actually seen it.
- **Identity switch** — a mistake where the tracker either swaps two objects'
  IDs, or attaches an object's ID to the wrong physical thing.
- **Ghost track** — the system keeps reporting an object that isn't really
  there anymore (it already left, or was never a real object to begin with).
  This is the main safety risk temporal memory can introduce.
- **Causal or online inference** — at the current frame, the system may only
  use *this* frame and *earlier* frames — never a future frame it hasn't
  "seen" yet. This matters because a real self-driving car can never see the
  future; an experiment that cheats by looking ahead would be measuring
  something that could never work in an actual vehicle.
- **Privileged ground truth** — extra information (like official 3D box
  labels, or LiDAR) that the research team is allowed to use to build the
  test and grade the results, but that the live camera-only system itself is
  never allowed to see while running. It's like an answer key used to grade a
  test, not something the student gets to peek at while taking it.
- **Natural occlusion** — a real, already-existing moment in the recorded
  video where one real object actually blocked another. Found by
  searching real footage, not created artificially.
- **Controlled visual occlusion** — the researchers deliberately paste a
  realistic mask or shape over a real object in a copy of the image (never
  touching the original file), specifically to test the system with an exact,
  known occlusion duration and severity.
- **Detector intervention** — a cheaper, more artificial test: instead of
  covering up the image, the researchers just delete or weaken a detection
  score directly in the data. Useful for testing the tracker's logic in
  isolation, but this is *not* the same as testing real visual occlusion, and
  must never be described as if it were.
- **Development, validation, and test split** — dividing the driving scenes
  into three separate groups: one for trying things out and tuning settings
  (development), one for double-checking those settings on unseen scenes
  before the very final run (validation), and one that is only opened at the
  very end for the final, honest result (test). Splitting by whole scene
  (never by nearby frames) stops the system from "cheating" by training and
  testing on nearly-identical images.
- **Ablation study** — an experiment where you remove or disable one piece of
  the system at a time (e.g., "what if we turn off motion prediction?") to
  find out which specific piece is actually responsible for an improvement,
  rather than just saying "the whole system works better" without knowing why.

## The pipeline

```text
current camera frame
    -> frozen object detector
    -> strong and weak detections
    -> associate detections with existing tracks
    -> update motion and uncertainty
    -> decide whether an unmatched object is hidden, lost, or exited
    -> keep or terminate the prediction
    -> reconnect the same identity if compatible evidence returns
    -> save detections and predictions with different labels
```

Every box on this page that reaches the final saved output must be clearly
labeled as either **currently seen** (a real detection right now) or
**predicted** (a memory-based guess while hidden) — never something in
between or ambiguous.

## How the methods compare

| Method | What it remembers | Main limitation |
|---|---|---|
| YOLO only | Nothing between frames | Loses objects when evidence disappears |
| Static last-seen memory | Last box | Does not move with the object or camera |
| SORT | Simple motion | Can drift and does not explicitly reason about occlusion |
| ByteTrack | Strong and weak detections plus motion | Still needs visible evidence and can retain wrong tracks |
| OATM | Evidence type, motion, uncertainty, and occlusion state | More rules to validate and a risk of ghost predictions |

**Task 0 is complete when:** this page exists and you can explain the
difference between current visual evidence and temporal prediction in your
own words — see the checkpoint questions below.

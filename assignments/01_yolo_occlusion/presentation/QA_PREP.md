# Q&A Prep — Plain-English Answers

Read these like you'd explain them to a friend, not like a textbook. Each
one has: the simple version, and (where useful) a one-line "if they push
further" follow-up.

---

## The things you specifically asked about

### IoU (Intersection over Union)
**It's how much two boxes overlap, on a 0-to-1 scale.** 1 = perfect overlap,
0 = they don't touch at all. When we say "IoU 0.34," that means the
predicted box and the real box only overlap by about a third — not great,
not zero.
*If pushed:* "It's the standard way computer vision measures 'how close was
this box to the right answer,' used everywhere from object detection to
tracking."

### SORT
**It's one of the simplest tracking methods that exists: detect, predict
where it'll move using constant velocity, match new detections to existing
tracks by box overlap.** No deep learning in the tracking part itself — it's
old-school and lightweight. We used it because it's the simplest possible
version of "predict motion instead of freezing," so it was the right first
thing to prototype before building something more complex.

### ByteTrack
**It's SORT's smarter cousin.** Its one big idea: don't throw away
low-confidence detections — keep them and try to match them too, because a
half-occluded object often gets a low-confidence box, and that's still
useful signal, not noise. Its limitation for us: it still needs *some*
visible detection to latch onto. It doesn't reason about an object that's
100% invisible, which is exactly our problem.

### The Kalman-style predict → observe → correct loop
**Analogy: predicting where a rolling ball will be one second from now.**
You don't need to keep watching it — you just need where it is now and how
fast it's moving. That's the whole idea:
- **Predict**: use last known position + velocity to guess where it is now.
- **Observe**: if a real detection shows up, note it.
- **Correct**: blend the prediction and the real observation into an
  updated estimate.
During occlusion, there's no observation, so it just keeps predicting
forward using velocity alone — that's what lets it do better than a frozen
box.

### "Fixed to use an independent reference and real track-expiry"
Two separate fixes, both about honesty in how we tested SORT:
1. **Independent reference** — our first test accidentally graded SORT's
   guesses against an answer key that was partly made by SORT itself (see
   the circular-evaluation explanation you already have). Fixing it means
   grading against LiDAR instead, a completely separate sensor.
2. **Real track-expiry** — every real tracker has a rule like "if I haven't
   seen this object in 3 frames, assume it's gone and stop guessing." Our
   first test accidentally let it keep guessing forever during long gaps,
   which isn't how it would actually behave once deployed. Fixing it means
   testing it the way it would really run.

### Track-expiry / max_age (if they use the technical term)
**Just the "give up" rule.** Without it, a tracker would keep reporting an
object that's long gone — a ghost. `max_age: 3` means: after 3 missed
frames with no detection, delete the track.

### Why only the front camera?
**To isolate one variable.** Real self-driving cars fuse multiple cameras,
LiDAR, and radar together. We deliberately used just one camera so the
occlusion-memory question wouldn't get tangled up with multi-sensor fusion
complexity — this is a controlled first experiment testing one idea
cleanly, not a claim that a real car should only use one camera.

### Next steps (Roadmap)
**In order:** fix how we grade SORT (use LiDAR, not another camera
tracker), build a bigger and more controlled occlusion test set, actually
build OATM (right now it's a design, not code), benchmark it against SORT/
ByteTrack/static memory using that fair LiDAR-based grading, then extend
from cars to pedestrians.

---

## Other things a sharp, AI-literate judge might ask

### "MeMOT already does memory-based tracking well. How is OATM different?"
MeMOT proved memory-based tracking reduces ID switches — but it's a heavy
transformer trained end-to-end on huge labeled datasets, and it's a black
box: it doesn't explicitly say *why* it thinks an object is still there.
OATM is a lighter, rule-based layer that explicitly classifies visible /
occluded / lost / exited — the tradeoff is interpretability and low compute
cost vs. MeMOT's raw performance ceiling.

### "PermaTrack already predicts through occlusion. Why not just use that?"
PermaTrack is trained end-to-end to hallucinate occluded positions, which
requires synthetic data with perfect occlusion labels (since real data
can't label what's invisible), plus careful fine-tuning to not forget that
skill on real data. OATM avoids that whole synthetic-to-real training
pipeline by using explicit states and confidence decay instead of a learned
hallucination network — more tractable for a project at this scale.

### "What's the Hungarian algorithm, and why does SORT need it?"
When you have several tracked objects and several new detections in one
frame, you need to decide which detection belongs to which track. The
Hungarian algorithm finds the single best one-to-one pairing (by box
overlap) that no other combination could beat. It's just an optimal
assignment solver, not anything occlusion-specific.

### "Is any of this fast enough to run in real time?"
The tracking math itself (Kalman filter, matching) is extremely cheap —
milliseconds per frame. The real cost is the YOLO detector, which already
runs in real time on modest hardware. So yes, the design goal is to stay
lightweight, unlike heavier transformer-based competitors like MeMOT.

### "How exactly do you decide VISIBLE vs OCCLUDED vs LOST vs EXITED?"
Be honest: **that's the part that isn't built yet.** Right now it's a
proposed classifier, not implemented code. The plan is to use signals like
how long the object's been missing, whether it was near the image edge,
and how confident the last real detection was — but the exact rule or model
for this is explicitly future work.

### "What happens with multiple occluded objects at once, or crowded scenes?"
Untested limitation — everything so far uses single-target sequences.
Crowded scenes are harder because of the assignment problem (which object
is which when several are hidden at once). Worth saying plainly: that's
real future work, not something OATM already handles.

### "Why nuScenes and not KITTI?"
KITTI is older and doesn't have as rich per-object occlusion labeling.
nuScenes has synchronized LiDAR + camera plus a built-in 1-4 visibility
scale for every object — exactly what's needed both for the occlusion
experiments and for independent LiDAR-based evaluation later.

### "Isn't 9 tracks / 5 samples way too small to conclude anything?"
Yes — and that's exactly why the deck calls this exploratory evidence, not
proof. It's a small pilot to check the idea is worth building further, not
a benchmark result. Step one of the roadmap is building a proper, larger
test set specifically because of this.

### "What's a 'ghost track' / 'ghost detection,' concretely?"
When the tracker keeps reporting an object that isn't there anymore — like
a memory that never lets go. Dangerous in a real car because it might brake
or swerve for something that's no longer actually present.

### "What does 'appearance memory' actually store — a whole image?"
No — a small crop or compact feature representation of the object's last
clear view, so if it reappears, the system can check "does this new
detection look like the thing I lost," not just rely on position alone.

### "Is confidence decay just a countdown timer?"
No, and that's the point of calling it out — a fixed timer drops confidence
the same way regardless of context. The design decays confidence based on
the occlusion classifier's state, so an object hidden behind a large truck
for a long stretch should decay faster than one that flickered out for a
single frame.

---

## If you get asked something you genuinely don't know

Say so plainly: *"That's a great question — it's actually one of the things
we've flagged as future work / haven't nailed down yet."* Judges who know
AI respect an honest "not yet solved" far more than a bluffed answer that
falls apart under one follow-up. Your whole deck's credibility is built on
that honesty already (the SORT caveats, the "not measured yet" labels) —
staying consistent with that in the room is a strength, not a weakness.

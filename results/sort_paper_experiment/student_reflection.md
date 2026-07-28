# Student Reflection (Task 12)

Answers are the student's own words, with spelling/grammar lightly cleaned only. Meaning has not
been replaced or invented, per the assignment's rule. Where Claude flagged a possible mix-up or
asked a follow-up and the student chose to move on instead of revising, that is noted honestly below
rather than silently corrected.

**1. In your own words, what is the difference between YOLO and SORT?**

> YOLO keeps a box where the object was before, while SORT tries to predict [where the object is now].

*Note: Claude flagged that this description (freezing a box where the object used to be) matches
Assignment 2's static-memory fix rather than plain YOLO, which shows no box at all once it loses an
object. The student did not revise this after the flag, so the answer is recorded as given.*

**2. Why did the static orange box become stale?**

> Because it's a frozen box for YOLO [-- it doesn't move once it's set, so it stops matching reality
> as the object keeps moving].

**3. What did the blue SORT box do differently?**

> It predicted where the object was going to appear, instead of just keeping a box where the object
> was before.

**4. Describe one case where the constant-velocity idea helped or failed.**

> The sample I put in my PPT.

*Note: Claude flagged that this reference points to the earlier project-overview presentation, not a
specific case from this SORT experiment, and offered two concrete in-experiment examples (the
`clip_sample_006` track 430 success case, and the gap=10 case where SORT's IoU advantage vanished --
see `student_experiment.md`). The student did not pick one or provide an alternative, so the original
answer is recorded as given rather than substituted.*

**5. What is one reason this experiment cannot prove that SORT always works?**

> Student asked Claude to explain the question rather than answering directly, then asked to proceed
> without confirming or restating an answer in their own words. Claude's explanation, offered but
> **not confirmed by the student**, was: the experiment only tested 9 objects across 3 short clips,
> and it artificially deleted detections YOLO had already gotten right, rather than testing real
> detection failures -- so it's a small, controlled test, not proof that SORT handles every real-world
> situation (sharp turns, crowded scenes, genuinely bad detections, etc.).
>
> **This question does not have a confirmed student answer in their own words.**

# Student Conclusion: Last-Seen Memory Experiment

These are the student's own observations from reviewing the comparison images and the analysis in `final_report.md`.

**1. What happened when YOLO lost the target?**

A memory box appeared in its place, instead of the target simply vanishing with no trace.

**2. What did the memory system add?**

A bounding box that persisted through the occlusion -- carried forward unchanged from the last real detection, so the target's existence wasn't immediately forgotten.

**3. Was the old box still accurate when the object reappeared?**

Not really. Center error ranged from 83px up to 589px across the 5 samples, and in 3 of the 5 the memory box had zero overlap (IoU 0.00) with the real object once it reappeared.

**4. Describe one helpful example.**

`sample_003` (the dark sedan reappearing next to a bus) was the closest match: only 83px of center error and the highest IoU of the set (0.34). The memory box landed close enough to the real object that it would have been genuinely useful.

**5. Describe one misleading example.**

`sample_006` (the silver sedan driving through the intersection) was the worst case: 589px of drift and 0.00 IoU. The memory box was left pointing at empty road while the real car had already moved well past it -- exactly the kind of stale prediction that could mislead a system relying on it.

**6. Why should memory expire?**

So the system doesn't keep carrying a memory of a car it's never going to see again. If the box never expired, it would keep getting drawn indefinitely for objects that have long since left the scene -- a stale, meaningless prediction instead of real information.

**7. What should be added next?**

The memory box should use velocity to predict where the object has moved, instead of assuming it stayed in the exact same spot. This is exactly what samples like `sample_006` show: the object kept moving while the box stayed frozen, so factoring in motion would close a lot of that gap.

---

*This directly motivates the next step in the bigger OATM research project ([`OATM/METHODOLOGY.md`](../../../OATM/METHODOLOGY.md)): ego-motion-compensated motion prediction instead of a static last-seen box.*

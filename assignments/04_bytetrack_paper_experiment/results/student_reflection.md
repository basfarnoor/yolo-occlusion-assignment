# Student Reflection

Answered one question at a time, in the student's own words. Where an answer showed a genuine
misunderstanding, the concept was explained and the student was asked to try again; the final
response is recorded below, with a note where that happened.

## 1. Why can a low-confidence detection still be useful?

> Because bytetrack will still keep track of it knowing it might be partly occluded.

## 2. What happens in ByteTrack's first association?

*(The student asked for a plain-language explanation first; the answer below is her restatement
after that explanation.)*

> Bytetrack first looks only at the detections its fairly confident about (the clearly visible
> boxes) and compares those to where it predicted each existing tracked object should be, and
> matches them up.

## 3. What happens in its second association?

*(Explained after an initial "idk"; answer below is the student's restatement.)*

> After the first round the tracks that are still unmatched -- bytetrack takes a second look and
> checks if any of the low-confidence boxes match the undetected ones.

## 4. Why should a weak unmatched box not start a new track?

*(This one took two tries -- the first answer confused it with keeping the same ID across frames,
and the second attempt was a verbatim copy of the explanation rather than her own phrasing. The
answer below is her own, shorter restatement on the third attempt.)*

> Because it means that nothing is recognized to a tracked object and the detector isn't really
> confident -- if bytetrack started a brand new track from it anyway, it might just be background
> noise.

## 5. Describe one case where ByteTrack helped or made a mistake.

> It helped in the pedestrian scene -- bytetrack kept the weak detection to her and kept the same
> ID when she got clearly visible again.

## 6. Why can ByteTrack not fully solve complete occlusion?

*(Explained after an initial answer that had the mechanism backwards; answer below is the
student's restatement.)*

> It needs a detection box to match against, even a weak one.

## 7. What is one reason this small experiment cannot prove ByteTrack always works?

*(The first attempt answered a different question -- about why ByteTrack is needed at all, not
about this experiment's sample size. Refocused, and the answer below is her corrected response.)*

> Because the samples are limited.

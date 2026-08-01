"""ByteTrack paper reference: Zhang et al., ECCV 2022 (https://arxiv.org/abs/2110.06864).

Task 11: evaluates high-confidence SORT and ByteTrack on the natural
weak-evidence events (Task 9), against the INDEPENDENT projected nuScenes
ground truth (Task 4) -- never against either tracker's own output. For each
event, at the before/event/after keyframes, records whether each method
produced a box overlapping the ground truth (a recall proxy) and whether the
SAME track ID persisted from before the confidence dip to after it (a
fragmentation/identity-continuity proxy, measured from real tracker output,
never assumed).
"""
from __future__ import annotations

from geometry import center_error, iou
from run_methods import new_bytetrack_tracker, new_sort_tracker

MATCH_IOU_THRESHOLD = 0.3


def _best_match(outputs: list, gt_box: tuple[float, float, float, float]) -> tuple[int | None, float, str]:
    best_iou, best_id, best_evidence = 0.0, None, ""
    for o in outputs:
        i = iou(o.box, gt_box)
        if i > best_iou:
            best_iou, best_id, best_evidence = i, o.track_id, o.evidence_source
    return best_id, best_iou, best_evidence


def evaluate_natural_events(events: list[dict], gt_by_key: dict, clip_frame_numbers: dict,
                              detections_by_frame: dict, timestamps_by_clip: dict, cfg: dict) -> list[dict]:
    """Runs each method fresh, ONCE per clip (not per event -- events sharing a
    clip share the same tracker run, exactly as a real online tracker would
    see the clip), then reads off outputs at each event's before/event/after
    frames."""
    rows = []
    clips_needed = {e["clip_name"] for e in events}

    for clip in clips_needed:
        frame_numbers = clip_frame_numbers[clip]
        timestamps = timestamps_by_clip[clip]

        methods = {"high_confidence_sort": new_sort_tracker(cfg), "bytetrack": new_bytetrack_tracker(cfg)}
        outputs_by_method_frame: dict[str, dict[int, list]] = {name: {} for name in methods}
        for frame_no in frame_numbers:
            dets = detections_by_frame.get((clip, frame_no), [])
            for name, tracker in methods.items():
                outputs_by_method_frame[name][frame_no] = tracker.update(dets, timestamp=timestamps[frame_no])

        clip_events = [e for e in events if e["clip_name"] == clip]
        for e in clip_events:
            gt_key_before = (clip, e["before_frame"], e["instance_token"])
            gt_key_event = (clip, e["event_frame"], e["instance_token"])
            gt_key_after = (clip, e["after_frame"], e["instance_token"])
            gt_before = gt_by_key.get(gt_key_before)
            gt_event = gt_by_key.get(gt_key_event)
            gt_after = gt_by_key.get(gt_key_after)
            if not (gt_before and gt_event and gt_after):
                continue

            for method_name, per_frame_outputs in outputs_by_method_frame.items():
                role_results = {}
                for role, frame_no, gt in (("before", int(e["before_frame"]), gt_before),
                                            ("event", int(e["event_frame"]), gt_event),
                                            ("after", int(e["after_frame"]), gt_after)):
                    gt_box = (float(gt["x1"]), float(gt["y1"]), float(gt["x2"]), float(gt["y2"]))
                    outputs = per_frame_outputs.get(frame_no, [])
                    track_id, matched_iou, evidence = _best_match(outputs, gt_box)
                    recovered = matched_iou >= MATCH_IOU_THRESHOLD
                    matched_output = next((o for o in outputs if o.track_id == track_id), None) if recovered else None
                    role_results[role] = {
                        "frame_number": frame_no, "track_id": track_id if recovered else None,
                        "recovered": recovered, "evidence_source": evidence if recovered else "",
                        "center_error_px": center_error(matched_output.box, gt_box) if matched_output else "",
                        "iou": matched_iou,
                    }

                identity_preserved = (
                    role_results["before"]["track_id"] is not None and
                    role_results["after"]["track_id"] is not None and
                    role_results["before"]["track_id"] == role_results["after"]["track_id"]
                )

                for role in ("before", "event", "after"):
                    r = role_results[role]
                    rows.append({
                        "clip_name": clip,
                        "instance_token": e["instance_token"],
                        "category": e["category"],
                        "split": e["split"],
                        "method": method_name,
                        "role": role,
                        "frame_number": r["frame_number"],
                        "track_id": r["track_id"] if r["track_id"] is not None else "",
                        "recovered": r["recovered"],
                        "evidence_source": r["evidence_source"],
                        "center_error_px": r["center_error_px"],
                        "iou": r["iou"],
                        "identity_preserved_before_to_after": identity_preserved,
                    })
    return rows

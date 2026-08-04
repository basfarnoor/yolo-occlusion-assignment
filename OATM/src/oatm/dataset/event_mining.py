"""Phase 3: mines candidate natural occlusion events from real visibility
transitions in the projected ground truth (Task 3's output). Candidate
ranking may use privileged metadata (this is offline dataset construction,
not online inference -- METHODOLOGY.md's camera-only boundary applies only
to the live tracker, never to how the evaluation set itself is built).

An event is a maximal run of low-visibility keyframes for one instance,
bounded by a high-visibility frame immediately before and after (both still
successfully projected -- i.e. still geometrically inside CAM_FRONT, so this
is a *visibility* drop, not the object leaving the frame). Acceptance as a
candidate additionally requires a plausible occluder: another instance whose
box overlaps the target's in the same frame and which sits closer to the
camera (smaller center_depth_m) -- a second, independent signal beyond the
visibility label alone.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

LOW_VISIBILITY_TOKENS = {"1", "2"}   # v0-40, v40-60
HIGH_VISIBILITY_TOKENS = {"3", "4"}  # v60-80, v80-100
MIN_OCCLUDER_IOU = 0.02
MIN_OCCLUDER_DEPTH_MARGIN_M = 0.3
MAX_TRUNCATION_FOR_NON_EXIT = 0.3  # above this, treat the low-vis frame as a likely exit, not occlusion


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class EventCandidate:
    scene_token: str
    instance_token: str
    evaluation_class: str
    pre_frame: dict
    start_frame: dict
    end_frame: dict
    post_frame: dict
    low_vis_run_length: int
    possible_occluder_instance_token: str | None
    occluder_overlap_iou: float
    rejection_reason: str = ""

    @property
    def has_two_independent_signals(self) -> bool:
        return self.possible_occluder_instance_token is not None


def find_plausible_occluder(target_row: dict, same_frame_rows: list[dict]) -> tuple[str | None, float]:
    target_box = (target_row["x1"], target_row["y1"], target_row["x2"], target_row["y2"])
    best_iou, best_instance = 0.0, None
    for other in same_frame_rows:
        if other["instance_token"] == target_row["instance_token"]:
            continue
        if other["center_depth_m"] >= target_row["center_depth_m"] - MIN_OCCLUDER_DEPTH_MARGIN_M:
            continue  # occluder must plausibly be closer to the camera than the target
        other_box = (other["x1"], other["y1"], other["x2"], other["y2"])
        iou = _box_iou(target_box, other_box)
        if iou > best_iou:
            best_iou, best_instance = iou, other["instance_token"]
    if best_iou >= MIN_OCCLUDER_IOU:
        return best_instance, best_iou
    return None, 0.0


def find_candidate_events(
    rows_by_instance: dict[tuple[str, str], list[dict]],
    rows_by_frame: dict[str, list[dict]],
) -> tuple[list[EventCandidate], list[EventCandidate]]:
    """Returns (accepted_candidates, rejected_candidates) -- rejected ones are
    kept (with a reason) for an honest, traceable selection log, never
    silently dropped."""
    accepted, rejected = [], []

    for (scene_token, instance_token), rows in rows_by_instance.items():
        rows = sorted(rows, key=lambda r: r["frame_index"])
        visibility = [r["visibility_token"] for r in rows]

        i = 0
        while i < len(rows):
            if visibility[i] not in LOW_VISIBILITY_TOKENS:
                i += 1
                continue
            run_start = i
            while i < len(rows) and visibility[i] in LOW_VISIBILITY_TOKENS:
                i += 1
            run_end = i - 1  # inclusive

            if run_start == 0 or run_end == len(rows) - 1:
                continue  # no frame before or after the run -- can't confirm recovery
            if visibility[run_start - 1] not in HIGH_VISIBILITY_TOKENS:
                continue
            if visibility[run_end + 1] not in HIGH_VISIBILITY_TOKENS:
                continue

            pre_row, start_row, end_row, post_row = (
                rows[run_start - 1], rows[run_start], rows[run_end], rows[run_end + 1],
            )
            evaluation_class = start_row["evaluation_class"]

            candidate_kwargs = dict(
                scene_token=scene_token, instance_token=instance_token, evaluation_class=evaluation_class,
                pre_frame=pre_row, start_frame=start_row, end_frame=end_row, post_frame=post_row,
                low_vis_run_length=run_end - run_start + 1,
            )

            if start_row["truncation_fraction"] > MAX_TRUNCATION_FOR_NON_EXIT:
                truncation = start_row["truncation_fraction"]
                rejected.append(EventCandidate(
                    **candidate_kwargs, possible_occluder_instance_token=None, occluder_overlap_iou=0.0,
                    rejection_reason=(
                        f"start frame heavily truncated ({truncation:.2f}) -- likely exit, not occlusion"
                    ),
                ))
                continue

            occluder_token, overlap_iou = find_plausible_occluder(
                start_row, rows_by_frame.get(start_row["sample_data_token"], [])
            )
            candidate = EventCandidate(
                **candidate_kwargs,
                possible_occluder_instance_token=occluder_token,
                occluder_overlap_iou=overlap_iou,
            )
            if candidate.has_two_independent_signals:
                accepted.append(candidate)
            else:
                candidate.rejection_reason = "no plausible closer occluder found in the same frame"
                rejected.append(candidate)

    return accepted, rejected


def rank_candidates(candidates: list[EventCandidate]) -> list[EventCandidate]:
    """Deterministic ranking: longer, more convincing occlusions and
    stronger occluder overlap first; stable tiebreak by token."""
    return sorted(
        candidates,
        key=lambda c: (-c.low_vis_run_length, -c.occluder_overlap_iou, c.scene_token, c.instance_token),
    )


def assign_scene_split(scene_tokens: list[str], seed: int,
                        n_development: int, n_validation: int) -> dict[str, str]:
    """Scene-derived split, assigned BEFORE any event is selected or reviewed
    -- required so no event-selection decision can leak across splits."""
    tokens = sorted(scene_tokens)
    rng = random.Random(seed)
    shuffled = tokens[:]
    rng.shuffle(shuffled)

    split: dict[str, str] = {}
    for i, token in enumerate(shuffled):
        if i < n_development:
            split[token] = "development"
        elif i < n_development + n_validation:
            split[token] = "validation"
        else:
            split[token] = "test"
    return split

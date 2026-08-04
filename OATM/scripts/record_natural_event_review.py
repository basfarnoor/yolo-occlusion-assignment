"""Phase 3 (Task 4), part 2: records the student's actual accept/reject/unsure
review of the mined candidates (from scripts/mine_natural_events.py) into the
immutable manifest. The review answers below are transcribed verbatim from
the chat checkpoint -- never invented or inferred by the assistant.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent

# Verbatim student review, one entry per candidate_number from
# natural_event_candidates.json. Recorded exactly as given in chat.
STUDENT_REVIEW = {
    1: ("unsure", "unclear to my eye"),
    2: ("unsure", "unclear to my eye"),
    3: ("accepted", "the rest are good"),
    4: ("unsure", "unclear to my eye"),
    5: ("unsure", "unclear to my eye"),
    6: ("unsure", "unclear to my eye"),
    7: ("accepted", "the rest are good"),
    8: ("accepted", "the rest are good"),
    9: ("accepted", "the rest are good"),
    10: ("unsure", "more than one occluder -- too much overlap to confidently judge this as one event"),
    11: ("unsure", "more than one occluder -- too much overlap to confidently judge this as one event"),
    12: ("unsure", "unclear to my eye"),
    13: ("accepted", "the rest are good"),
    14: ("rejected", "wrong objects"),
    15: ("accepted", "the rest are good"),
}


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)

    with open(config.artifacts_dir / "natural_event_candidates.json", encoding="utf-8") as f:
        payload = json.load(f)
    candidates = payload["candidates"]

    if set(STUDENT_REVIEW.keys()) != {c["candidate_number"] for c in candidates}:
        raise RuntimeError("STUDENT_REVIEW does not cover exactly the candidates that were shown for review.")

    manifest_rows = []
    for c in candidates:
        review_status, reason = STUDENT_REVIEW[c["candidate_number"]]
        event_id = f"{c['scene_token'][:8]}_{c['instance_token'][:8]}"
        manifest_rows.append({
            "event_id": event_id,
            "scene_token": c["scene_token"],
            "instance_token": c["instance_token"],
            "evaluation_class": c["evaluation_class"],
            "pre_frame_index": c["pre_frame_index"],
            "start_frame_index": c["start_frame_index"],
            "end_frame_index": c["end_frame_index"],
            "post_frame_index": c["post_frame_index"],
            "event_source": "natural",
            "visibility_pattern": f"{c['pre_visibility']}-{c['start_visibility']}-{c['post_visibility']}",
            "low_vis_run_length": c["low_vis_run_length"],
            "occluder_overlap_iou": c["occluder_overlap_iou"],
            "review_status": review_status,
            "rejection_reason": reason if review_status != "accepted" else "",
            "split": c["split"],
        })

    fieldnames = list(manifest_rows[0].keys())
    manifest_path = config.results_dir / "natural_event_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    n_accepted = sum(1 for r in manifest_rows if r["review_status"] == "accepted")
    n_unsure = sum(1 for r in manifest_rows if r["review_status"] == "unsure")
    n_rejected = sum(1 for r in manifest_rows if r["review_status"] == "rejected")

    lines = ["# Natural Event Selection Log\n\n"]
    lines.append(
        "## Automated mining\n\n"
        "Candidates require BOTH an official visibility-label decline-then-recovery "
        "AND a plausible closer occluder (another instance overlapping the target's box, "
        "sitting nearer the camera by `center_depth_m`) in the same frame -- two "
        "independent signals, per this task's requirement.\n\n"
        f"- Instances with a decline-recovery pattern AND a plausible occluder "
        f"(auto-accepted as candidates): **{payload['n_accepted_candidates_total']}**\n"
        f"- Instances with a decline-recovery pattern but NO plausible occluder found "
        f"(auto-rejected, logged, never shown for review): **{payload['n_rejected_candidates_total']}**\n"
        f"- Shortlisted for human review (top {len(candidates)} by run length, then occluder overlap): "
        f"**{len(candidates)}**\n\n"
    )
    lines.append(
        "## Student review (verbatim, recorded exactly as given -- not invented)\n\n"
        f"- Accepted: **{n_accepted}**\n- Unsure: **{n_unsure}**\n- Rejected: **{n_rejected}**\n\n"
    )
    lines.append("| # | Scene (split) | Class | Review | Reason (student's words) |\n|---|---|---|---|---|\n")
    for c in candidates:
        review_status, reason = STUDENT_REVIEW[c["candidate_number"]]
        lines.append(f"| {c['candidate_number']} | `{c['scene_token'][:8]}` ({c['split']}) | "
                     f"{c['evaluation_class']} | **{review_status}** | {reason} |\n")

    lines.append(
        "\nOnly `accepted` events are eligible for OATM MVP evaluation (Task 11). "
        "`unsure` events are kept in the manifest for traceability but excluded from "
        "the accepted evaluation set -- they are not silently promoted to accepted. "
        "This mini result is a **pilot**, not a final statistical conclusion: 15 "
        "reviewed candidates, 6 accepted, is a small sample from one dataset split.\n"
    )

    with open(config.results_dir / "natural_event_selection.md", "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"Wrote {manifest_path} ({len(manifest_rows)} rows: "
          f"{n_accepted} accepted, {n_unsure} unsure, {n_rejected} rejected)")
    print("Wrote natural_event_selection.md")


if __name__ == "__main__":
    main()

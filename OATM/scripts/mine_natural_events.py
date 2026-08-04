"""Phase 3 (Task 4), part 1: mines and ranks candidate natural occlusion
events, assigns the scene-disjoint split, and builds contact-sheet images for
human review. Does NOT write the final manifest -- that happens after the
student's actual accept/reject/unsure review (see
scripts/record_natural_event_review.py).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from oatm.config import find_repo_root, load_config  # noqa: E402
from oatm.dataset.event_mining import assign_scene_split, find_candidate_events, rank_candidates  # noqa: E402

OATM_ROOT = Path(__file__).resolve().parent.parent
MAX_CANDIDATES_FOR_REVIEW = 15
N_DEVELOPMENT_SCENES = 6
N_VALIDATION_SCENES = 2  # remaining scenes become "test"
MVP_CLASSES = {"car", "pedestrian"}
MAGENTA = (230, 0, 200)
CYAN = (0, 220, 220)
FONT_PATH = r"C:\Windows\Fonts\arial.ttf"


def main() -> None:
    repo_root = find_repo_root()
    config = load_config(OATM_ROOT / "configs" / "mini.yaml", repo_root=repo_root)

    pgt = pd.read_parquet(config.artifacts_dir / "projected_ground_truth.parquet")
    frame_index = pd.read_parquet(config.artifacts_dir / "frame_index.parquet")

    frame_lookup = frame_index.set_index("sample_data_token")[["frame_index", "image_path"]]
    pgt = pgt.join(frame_lookup, on="sample_data_token")

    all_rows = pgt.to_dict("records")
    rows_by_frame: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        rows_by_frame[r["sample_data_token"]].append(r)

    rows_by_instance: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in all_rows:
        if r["evaluation_class"] in MVP_CLASSES:
            rows_by_instance[(r["scene_token"], r["instance_token"])].append(r)

    accepted, rejected = find_candidate_events(dict(rows_by_instance), dict(rows_by_frame))
    ranked = rank_candidates(accepted)
    shortlist = ranked[:MAX_CANDIDATES_FOR_REVIEW]

    with open(OATM_ROOT / "configs" / "mini.yaml", encoding="utf-8") as f:
        import yaml
        seed = yaml.safe_load(f)["random_seed"]

    scene_tokens = sorted({r["scene_token"] for r in all_rows})
    split_by_scene = assign_scene_split(scene_tokens, seed, N_DEVELOPMENT_SCENES, N_VALIDATION_SCENES)

    review_dir = config.artifacts_dir / "natural_event_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT_PATH, 13)

    candidate_summaries = []
    panels_for_sheet = []
    for idx, cand in enumerate(shortlist, start=1):
        panels = []
        for role, row in (("PRE", cand.pre_frame), ("START", cand.start_frame),
                          ("END", cand.end_frame), ("POST", cand.post_frame)):
            img = Image.open(config.data_root / row["image_path"]).convert("RGB").resize((300, 169))
            draw = ImageDraw.Draw(img)
            scale_x, scale_y = 300 / 1600, 169 / 900
            box = (row["x1"] * scale_x, row["y1"] * scale_y, row["x2"] * scale_x, row["y2"] * scale_y)
            draw.rectangle(box, outline=MAGENTA, width=3)
            if role == "START" and cand.possible_occluder_instance_token:
                occluder_row = next(
                    (r for r in rows_by_frame[cand.start_frame["sample_data_token"]]
                     if r["instance_token"] == cand.possible_occluder_instance_token),
                    None,
                )
                if occluder_row:
                    obox = (occluder_row["x1"] * scale_x, occluder_row["y1"] * scale_y,
                            occluder_row["x2"] * scale_x, occluder_row["y2"] * scale_y)
                    draw.rectangle(obox, outline=CYAN, width=2)
            label = f"#{idx} {role} f{row['frame_index']} vis={row['visibility_token']}"
            draw.rectangle([0, 0, 300, 16], fill=(0, 0, 0))
            draw.text((2, 1), label, font=font, fill=(255, 255, 255))
            panels.append(img)
        row_img = Image.new("RGB", (300 * 4, 169), (20, 20, 20))
        for i, p in enumerate(panels):
            row_img.paste(p, (i * 300, 0))
        panels_for_sheet.append(row_img)

        candidate_summaries.append({
            "candidate_number": idx,
            "scene_token": cand.scene_token,
            "split": split_by_scene[cand.scene_token],
            "instance_token": cand.instance_token,
            "evaluation_class": cand.evaluation_class,
            "low_vis_run_length": cand.low_vis_run_length,
            "occluder_overlap_iou": round(cand.occluder_overlap_iou, 3),
            "pre_frame_index": cand.pre_frame["frame_index"],
            "start_frame_index": cand.start_frame["frame_index"],
            "end_frame_index": cand.end_frame["frame_index"],
            "post_frame_index": cand.post_frame["frame_index"],
            "pre_visibility": cand.pre_frame["visibility_token"],
            "start_visibility": cand.start_frame["visibility_token"],
            "post_visibility": cand.post_frame["visibility_token"],
        })

    sheet = Image.new("RGB", (300 * 4, 169 * len(panels_for_sheet)), (20, 20, 20))
    for i, row_img in enumerate(panels_for_sheet):
        sheet.paste(row_img, (0, i * 169))
    sheet_path = review_dir / "candidate_review_contact_sheet.png"
    sheet.save(sheet_path)

    with open(config.artifacts_dir / "natural_event_candidates.json", "w", encoding="utf-8") as f:
        json.dump({
            "candidates": candidate_summaries,
            "n_accepted_candidates_total": len(accepted),
            "n_rejected_candidates_total": len(rejected),
            "rejection_reasons": [
                {
                    "scene_token": r.scene_token, "instance_token": r.instance_token,
                    "reason": r.rejection_reason,
                }
                for r in rejected
            ],
            "split_by_scene": split_by_scene,
        }, f, indent=2)

    print(f"Found {len(accepted)} candidates with two independent occlusion signals "
          f"({len(rejected)} rejected -- see natural_event_candidates.json for reasons).")
    print(f"Shortlisted top {len(shortlist)} for review. Contact sheet: {sheet_path}")
    for s in candidate_summaries:
        print(f"  #{s['candidate_number']}: scene={s['scene_token'][:8]} ({s['split']}) "
              f"instance={s['instance_token'][:8]} class={s['evaluation_class']} "
              f"run_length={s['low_vis_run_length']} occluder_iou={s['occluder_overlap_iou']}")


if __name__ == "__main__":
    main()

"""Phase 1: read-only nuScenes mini audit and chronological CAM_FRONT index.

Reconstructs every scene's CAM_FRONT frame chain using the dataset's own
`prev`/`next` links (never by sorting on filename or trusting timestamps
alone), and validates it against a strict checklist before any downstream
phase is allowed to use it. Only ever reads metadata; never writes to the
nuScenes data root.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from oatm.records import FrameIndexRecord

EXPECTED_IMAGE_SIZE = (1600, 900)  # (width, height) -- nuScenes CAM_FRONT native resolution


@dataclass
class SceneAuditResult:
    scene_token: str
    scene_name: str
    n_cam_front_records: int
    n_chain_walked: int
    chain_complete: bool  # walked count matches the scene's actual record count
    n_heads: int  # records with prev == "" -- must be exactly 1
    n_tails: int  # records with next == "" -- must be exactly 1
    strictly_increasing_timestamps: bool
    reciprocal_links_ok: bool
    missing_image_files: list[str] = field(default_factory=list)
    unexpected_image_dimensions: list[str] = field(default_factory=list)
    missing_calibration_refs: list[str] = field(default_factory=list)
    missing_pose_refs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.chain_complete
            and self.n_heads == 1
            and self.n_tails == 1
            and self.strictly_increasing_timestamps
            and self.reciprocal_links_ok
            and not self.missing_image_files
            and not self.unexpected_image_dimensions
            and not self.missing_calibration_refs
            and not self.missing_pose_refs
        )


@dataclass
class DatasetAuditResult:
    dataset_version: str
    n_scenes: int
    n_keyframes: int
    n_cam_front_records: int
    scene_results: list[SceneAuditResult]

    @property
    def all_scenes_ok(self) -> bool:
        return all(s.ok for s in self.scene_results)


def _load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class NuscenesMiniMetadata:
    """In-memory index over the raw nuScenes JSON tables. Read-only."""

    def __init__(self, data_root: Path, dataset_version: str):
        meta_dir = data_root / dataset_version
        self.data_root = data_root
        self.scenes = _load_json(meta_dir / "scene.json")
        self.samples = {s["token"]: s for s in _load_json(meta_dir / "sample.json")}
        self.sample_data = {r["token"]: r for r in _load_json(meta_dir / "sample_data.json")}
        self.sensors = _load_json(meta_dir / "sensor.json")
        self.calibrated_sensors = {c["token"]: c for c in _load_json(meta_dir / "calibrated_sensor.json")}
        self.ego_poses = {e["token"]: e for e in _load_json(meta_dir / "ego_pose.json")}

    def cam_front_calibrated_sensor_tokens(self) -> set[str]:
        cam_front_sensor_tokens = {s["token"] for s in self.sensors if s["channel"] == "CAM_FRONT"}
        return {
            token for token, cs in self.calibrated_sensors.items()
            if cs["sensor_token"] in cam_front_sensor_tokens
        }

    def scene_token_for_sample_data(self, sd_record: dict) -> str | None:
        sample = self.samples.get(sd_record["sample_token"])
        return sample["scene_token"] if sample else None


def _walk_chain(sample_data: dict[str, dict], head_token: str) -> list[dict]:
    chain = []
    token = head_token
    seen = set()
    while token:
        if token in seen:
            break  # cycle guard -- should never happen in valid data
        seen.add(token)
        rec = sample_data.get(token)
        if rec is None:
            break
        chain.append(rec)
        token = rec["next"]
    return chain


def audit_scene(
    scene: dict, cam_front_records: list[dict], meta: NuscenesMiniMetadata
) -> tuple[SceneAuditResult, list[dict]]:
    heads = [r for r in cam_front_records if r["prev"] == ""]
    tails = [r for r in cam_front_records if r["next"] == ""]

    chain: list[dict] = []
    chain_complete = False
    strictly_increasing = False
    reciprocal_ok = True

    if len(heads) == 1:
        chain = _walk_chain(meta.sample_data, heads[0]["token"])
        chain_complete = len(chain) == len(cam_front_records)

        timestamps = [r["timestamp"] for r in chain]
        strictly_increasing = all(b > a for a, b in zip(timestamps, timestamps[1:]))

        for i, rec in enumerate(chain):
            if i > 0 and rec["prev"] != chain[i - 1]["token"]:
                reciprocal_ok = False
            if i < len(chain) - 1 and rec["next"] != chain[i + 1]["token"]:
                reciprocal_ok = False

    missing_files, bad_dims, missing_calib, missing_pose = [], [], [], []
    for rec in chain:
        image_path = meta.data_root / rec["filename"]
        if not image_path.is_file():
            missing_files.append(rec["token"])
        if (rec.get("width"), rec.get("height")) != EXPECTED_IMAGE_SIZE:
            bad_dims.append(rec["token"])
        if rec["calibrated_sensor_token"] not in meta.calibrated_sensors:
            missing_calib.append(rec["token"])
        if rec["ego_pose_token"] not in meta.ego_poses:
            missing_pose.append(rec["token"])

    result = SceneAuditResult(
        scene_token=scene["token"], scene_name=scene["name"],
        n_cam_front_records=len(cam_front_records), n_chain_walked=len(chain),
        chain_complete=chain_complete, n_heads=len(heads), n_tails=len(tails),
        strictly_increasing_timestamps=strictly_increasing, reciprocal_links_ok=reciprocal_ok,
        missing_image_files=missing_files, unexpected_image_dimensions=bad_dims,
        missing_calibration_refs=missing_calib, missing_pose_refs=missing_pose,
    )
    return result, chain


def build_frame_index(data_root: Path, dataset_version: str = "v1.0-mini"
                       ) -> tuple[list[FrameIndexRecord], DatasetAuditResult]:
    meta = NuscenesMiniMetadata(data_root, dataset_version)
    cam_front_cs_tokens = meta.cam_front_calibrated_sensor_tokens()

    cam_front_by_scene: dict[str, list[dict]] = {}
    for rec in meta.sample_data.values():
        if rec["calibrated_sensor_token"] not in cam_front_cs_tokens:
            continue
        scene_token = meta.scene_token_for_sample_data(rec)
        if scene_token is None:
            continue
        cam_front_by_scene.setdefault(scene_token, []).append(rec)

    all_records: list[FrameIndexRecord] = []
    scene_results: list[SceneAuditResult] = []

    for scene in meta.scenes:
        scene_records = cam_front_by_scene.get(scene["token"], [])
        result, chain = audit_scene(scene, scene_records, meta)
        scene_results.append(result)

        for frame_index, rec in enumerate(chain):
            all_records.append(FrameIndexRecord(
                scene_token=scene["token"],
                sample_token=rec["sample_token"] or None,
                sample_data_token=rec["token"],
                timestamp_us=rec["timestamp"],
                frame_index=frame_index,
                is_keyframe=rec["is_key_frame"],
                image_path=rec["filename"],
                prev_token=rec["prev"] or None,
                next_token=rec["next"] or None,
                calibrated_sensor_token=rec["calibrated_sensor_token"],
                ego_pose_token=rec["ego_pose_token"],
            ))

    audit = DatasetAuditResult(
        dataset_version=dataset_version,
        n_scenes=len(meta.scenes),
        n_keyframes=len(meta.samples),
        n_cam_front_records=sum(len(v) for v in cam_front_by_scene.values()),
        scene_results=scene_results,
    )
    return all_records, audit
